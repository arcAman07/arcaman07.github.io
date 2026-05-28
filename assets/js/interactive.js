/* Shared interactive utilities for IRL blog posts.
   Vanilla JS, no dependencies. */

(function (global) {
'use strict';

const NS = 'http://www.w3.org/2000/svg';

// ---------- tiny SVG helpers ----------
function svgEl(name, attrs = {}, parent = null) {
  const el = document.createElementNS(NS, name);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(el);
  return el;
}
function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

// ---------- math helpers ----------
function softmax(xs, tau = 1) {
  const m = Math.max(...xs);
  const exps = xs.map(x => Math.exp((x - m) / Math.max(tau, 1e-6)));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / s);
}
function logsumexp(xs) {
  const m = Math.max(...xs);
  return m + Math.log(xs.reduce((acc, x) => acc + Math.exp(x - m), 0));
}
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function norm(a) { return Math.sqrt(dot(a, a)); }
function sub(a, b) { return a.map((v, i) => v - b[i]); }
function add(a, b) { return a.map((v, i) => v + b[i]); }
function scale(a, k) { return a.map(v => v * k); }
function unit(a) { const n = norm(a); return n < 1e-9 ? a.map(_ => 0) : a.map(v => v / n); }
function argmax(xs) { let bi = 0, bv = -Infinity; xs.forEach((v, i) => { if (v > bv) { bv = v; bi = i; } }); return bi; }

// ---------- 1D corridor MDP ----------
// states 0..N-1; actions L/R; bumping wall keeps state; rewards r[s]; horizon T.
function buildCorridor(N) {
  return {
    N,
    transition: (s, a) => {
      if (a === 0) return Math.max(0, s - 1);   // L
      if (a === 1) return Math.min(N - 1, s + 1); // R
      return s;
    }
  };
}
function hardValueIteration(mdp, reward, T, goalState = -1) {
  const V = Array.from({ length: T + 1 }, () => new Array(mdp.N).fill(0));
  const pi = Array.from({ length: T }, () => new Array(mdp.N).fill(0));
  for (let t = T - 1; t >= 0; t--) {
    for (let s = 0; s < mdp.N; s++) {
      if (s === goalState) { V[t][s] = 0; pi[t][s] = 1; continue; }
      const qL = reward[s] + V[t + 1][mdp.transition(s, 0)];
      const qR = reward[s] + V[t + 1][mdp.transition(s, 1)];
      if (qR >= qL) { V[t][s] = qR; pi[t][s] = 1; } else { V[t][s] = qL; pi[t][s] = 0; }
    }
  }
  return { V, pi };
}
function softValueIteration(mdp, reward, T, goalState = -1) {
  // soft Bellman: V_t(s) = logsumexp_a [r(s) + V_{t+1}(next(s,a))], with V at goal absorbed at 0
  const V = Array.from({ length: T + 1 }, () => new Array(mdp.N).fill(0));
  const piR = Array.from({ length: T }, () => new Array(mdp.N).fill(0));
  for (let t = T - 1; t >= 0; t--) {
    for (let s = 0; s < mdp.N; s++) {
      if (s === goalState) { V[t][s] = 0; piR[t][s] = 0.5; continue; }
      const qL = reward[s] + V[t + 1][mdp.transition(s, 0)];
      const qR = reward[s] + V[t + 1][mdp.transition(s, 1)];
      V[t][s] = logsumexp([qL, qR]);
      const p = softmax([qL, qR]);
      piR[t][s] = p[1]; // probability of R
    }
  }
  return { V, piR };
}
function forwardVisitation(mdp, piR, T, start, goalState = -1) {
  // D[t][s] = probability of being in state s at time t
  const D = Array.from({ length: T + 1 }, () => new Array(mdp.N).fill(0));
  D[0][start] = 1;
  for (let t = 0; t < T; t++) {
    for (let s = 0; s < mdp.N; s++) {
      if (D[t][s] === 0) continue;
      if (s === goalState) { D[t + 1][s] += D[t][s]; continue; }
      const pR = piR[t][s];
      const pL = 1 - pR;
      D[t + 1][mdp.transition(s, 0)] += D[t][s] * pL;
      D[t + 1][mdp.transition(s, 1)] += D[t][s] * pR;
    }
  }
  return D;
}
function sampleTrajectory(mdp, piR, T, start, rng) {
  const path = [start];
  let s = start;
  for (let t = 0; t < T; t++) {
    const u = rng();
    const a = u < piR[t][s] ? 1 : 0;
    s = mdp.transition(s, a);
    path.push(s);
  }
  return path;
}
function seededRng(seed) {
  let st = seed >>> 0;
  return function () {
    st = (st + 0x6D2B79F5) >>> 0;
    let t = st;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------- 2D gridworld (4-directional, deterministic) ----------
// Used by classical_irl 12x12 and MMP demos
function gridShortestPath(terrain, cost, start, goal) {
  // Dijkstra. Returns {path, dist}.
  const R = terrain.length, C = terrain[0].length;
  const dist = Array.from({ length: R }, () => new Array(C).fill(Infinity));
  const prev = Array.from({ length: R }, () => new Array(C).fill(null));
  dist[start[0]][start[1]] = 0;
  // simple priority queue (array-backed)
  const pq = [[0, start[0], start[1]]];
  while (pq.length) {
    pq.sort((a, b) => a[0] - b[0]);
    const [d, r, c] = pq.shift();
    if (d > dist[r][c]) continue;
    if (r === goal[0] && c === goal[1]) break;
    const moves = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    for (const [dr, dc] of moves) {
      const nr = r + dr, nc = c + dc;
      if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
      const nd = d + cost[terrain[nr][nc]];
      if (nd < dist[nr][nc]) {
        dist[nr][nc] = nd;
        prev[nr][nc] = [r, c];
        pq.push([nd, nr, nc]);
      }
    }
  }
  const path = [];
  let cur = goal;
  if (!prev[goal[0]][goal[1]] && !(goal[0] === start[0] && goal[1] === start[1])) {
    return { path: [], dist: Infinity };
  }
  while (cur) {
    path.push(cur);
    if (cur[0] === start[0] && cur[1] === start[1]) break;
    cur = prev[cur[0]][cur[1]];
  }
  path.reverse();
  return { path, dist: dist[goal[0]][goal[1]] };
}

// ---------- bar chart ----------
function drawBars(svg, opts) {
  const {
    values, target = null, labels = null, colors = null,
    yMax = null, width = 280, height = 140, padding = 24
  } = opts;
  clear(svg);
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  const n = values.length;
  const top = padding * 0.6, bot = height - padding;
  const left = padding, right = width - padding * 0.4;
  const bw = (right - left) / n * 0.7;
  const gap = (right - left) / n * 0.3;
  const maxV = yMax !== null ? yMax : Math.max(1, Math.max(...values), target ? Math.max(...target) : 0);
  svgEl('line', { x1: left, y1: bot, x2: right, y2: bot, stroke: '#999', 'stroke-width': 1 }, svg);
  values.forEach((v, i) => {
    const x = left + i * ((right - left) / n) + gap / 2;
    const h = Math.max(0, (v / maxV) * (bot - top));
    svgEl('rect', {
      class: 'bar', x, y: bot - h, width: bw, height: h,
      fill: colors ? colors[i] : '#3a6dd2', stroke: '#222', 'stroke-width': 0.5
    }, svg);
    if (target !== null) {
      const ht = (target[i] / maxV) * (bot - top);
      svgEl('line', {
        x1: x - 1, y1: bot - ht, x2: x + bw + 1, y2: bot - ht,
        stroke: '#d23a3a', 'stroke-width': 2, 'stroke-dasharray': '3,2'
      }, svg);
    }
    svgEl('text', {
      class: 'bar-value', x: x + bw / 2, y: bot - h - 4, 'text-anchor': 'middle'
    }, svg).textContent = (Math.abs(v) < 100 ? v.toFixed(v < 10 ? 1 : 0) : v.toFixed(0));
    if (labels) {
      svgEl('text', {
        class: 'bar-label', x: x + bw / 2, y: bot + 12, 'text-anchor': 'middle'
      }, svg).textContent = labels[i];
    }
  });
}

// ---------- export ----------
global.IRL = {
  NS, svgEl, clear,
  softmax, logsumexp, dot, norm, sub, add, scale, unit, argmax,
  buildCorridor, hardValueIteration, softValueIteration, forwardVisitation,
  sampleTrajectory, seededRng,
  gridShortestPath, drawBars
};

})(window);
