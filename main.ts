// Sokoban solver, push-optimal room by room: a box pushed into the hallway never returns, so a level's
// optimum is the sum of its rooms' optima. Each room: breadth-first search over pushes (player region
// normalised, dead squares and frozen 2x2 blocks pruned). Budgets count states, never time.
import { mkdirSync as M, readdirSync as R, readFileSync as F, writeFileSync as W } from 'node:fs';
import { join as J } from 'node:path';
const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const D = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];
const P = (t) => {
  const r = t.split('\n').filter((s) => s.length);
  if (!r.length) return null;
  const w = Math.max(...r.map((s) => s.length));
  const h = r.length;
  const V = new Uint8Array(w * h).fill(1);
  const G = [];
  const B = [];
  let p = -1;
  r.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const c = row[x];
      const i = y * w + x;
      if (c === '#') continue;
      V[i] = 0;
      if ('.*+'.includes(c)) G.push(i);
      if ('$*'.includes(c)) B.push(i);
      if ('@+'.includes(c)) p = i;
    }
  });
  for (let x = 0; x < w; x++) V[x] = V[(h - 1) * w + x] = 1;
  for (let y = 0; y < h; y++) V[y * w] = V[y * w + w - 1] = 1;
  B.sort((a, b) => a - b);
  G.sort((a, b) => a - b);
  return [w, V, G, B, p];
};
const L = (w, V, G) => {
  const l = new Uint8Array(V.length);
  const q = [...G];
  for (const g of G) l[g] = 1;
  while (q.length) {
    const x = q.pop();
    for (const [a, b] of D) {
      const d = b * w + a;
      const y = x - d;
      if (!V[y] && !V[y - d] && !l[y]) {
        l[y] = 1;
        q.push(y);
      }
    }
  }
  return l;
};
const MK = (n) => {
  const s = new Int32Array(n);
  let t = 0;
  const f = (w, V, occ, from) => {
    t++;
    let low = from;
    const st = [from];
    s[from] = t;
    while (st.length) {
      const p = st.pop();
      if (p < low) low = p;
      for (const q of [p - w, p + w, p - 1, p + 1])
        if (!V[q] && !occ[q] && s[q] !== t) {
          s[q] = t;
          st.push(q);
        }
    }
    return low;
  };
  f.has = (x) => s[x] === t;
  return f;
};
const WK = (w, V, occ, f, t) => {
  if (f === t) return '';
  const pv = new Int32Array(V.length).fill(-1);
  const hw = [];
  const q = [f];
  pv[f] = f;
  for (let i = 0; i < q.length; i++) {
    const p = q[i];
    if (p === t) break;
    for (const [a, b, c] of D) {
      const x = p + b * w + a;
      if (!V[x] && !occ[x] && pv[x] < 0) {
        pv[x] = p;
        hw[x] = c;
        q.push(x);
      }
    }
  }
  let s = '';
  for (let p = t; p !== f; p = pv[p]) s = hw[p] + s;
  return s;
};
const SR = (w, V, G, B0, p0) => {
  const n = B0.length;
  const sz = V.length;
  const gl = new Uint8Array(sz);
  for (const g of G) gl[g] = 1;
  const lv = L(w, V, G);
  if (B0.some((b) => !lv[b])) return null;
  const RA = MK(sz);
  const RB = MK(sz);
  const occ = new Uint8Array(sz);
  const SB = [B0.slice()];
  const SP = [p0];
  const PA = [-1];
  const PU = [-1];
  const seen = new Set();
  const key = (B, rep) => {
    let k = rep;
    for (const b of B) k = k * sz + b;
    return k;
  };
  for (const b of B0) occ[b] = 1;
  seen.add(key(B0, RB(w, V, occ, p0)));
  for (const b of B0) occ[b] = 0;
  let fd = B0.every((b) => gl[b]) ? 0 : -1;
  for (let hd = 0; fd < 0 && hd < SB.length; hd++) {
    if (hd > 400000) return null;
    const B = SB[hd];
    for (const b of B) occ[b] = 1;
    RA(w, V, occ, SP[hd]);
    for (let bi = 0; bi < n && fd < 0; bi++) {
      const b = B[bi];
      for (let k = 0; k < 4; k++) {
        const d = D[k][1] * w + D[k][0];
        const t = b + d;
        if (!RA.has(b - d) || V[t] || occ[t] || !lv[t]) continue;
        occ[b] = 0;
        occ[t] = 1;
        let fz = 0;
        for (const a of [t, t - 1, t - w, t - w - 1]) {
          const s = [a, a + 1, a + w, a + w + 1];
          if (s.every((x) => V[x] || occ[x]) && s.some((x) => occ[x] && !gl[x])) {
            fz = 1;
            break;
          }
        }
        const rp = fz ? -1 : RB(w, V, occ, b);
        occ[t] = 0;
        occ[b] = 1;
        if (fz) continue;
        const nx = B.slice();
        nx[bi] = t;
        nx.sort((x, y) => x - y);
        const kk = key(nx, rp);
        if (seen.has(kk)) continue;
        seen.add(kk);
        SB.push(nx);
        SP.push(b);
        PA.push(hd);
        PU.push(b * 4 + k);
        if (nx.every((x) => gl[x])) fd = SB.length - 1;
      }
    }
    for (const b of B) occ[b] = 0;
  }
  if (fd < 0) return null;
  const path = [];
  for (let s = fd; s > 0; s = PA[s]) path.push(PU[s]);
  path.reverse();
  let mv = '';
  let pl = p0;
  const cur = B0.slice();
  const oc = new Uint8Array(sz);
  for (const ps of path) {
    const fr = ps >> 2;
    const k = ps & 3;
    const d = D[k][1] * w + D[k][0];
    for (const b of cur) oc[b] = 1;
    mv += WK(w, V, oc, pl, fr - d);
    for (const b of cur) oc[b] = 0;
    mv += D[k][3];
    cur[cur.indexOf(fr)] = fr + d;
    pl = fr;
  }
  return mv;
};
const SV = (t) => {
  const q = P(t);
  if (!q) return '';
  const [w, V, G, B, p0] = q;
  if (p0 < 0 || !B.length) return '';
  const lv = L(w, V, G);
  const row = (p0 / w) | 0;
  const cut = new Uint8Array(V.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!V[i] && !lv[i]) cut[i] = 1;
  if (B.some((b) => cut[b])) return '';
  const part = new Int32Array(V.length).fill(-1);
  const comps = [];
  for (let s = 0; s < V.length; s++) {
    if (V[s] || cut[s] || part[s] >= 0) continue;
    const id = comps.length;
    const sq = [s];
    part[s] = id;
    for (let i = 0; i < sq.length; i++)
      for (const [a, b] of D) {
        const x = sq[i] + b * w + a;
        if (!V[x] && !cut[x] && part[x] < 0) {
          part[x] = id;
          sq.push(x);
        }
      }
    comps.push([[], []]);
  }
  for (const b of B) comps[part[b]][0].push(b);
  for (const g of G) {
    if (part[g] < 0) return '';
    comps[part[g]][1].push(g);
  }
  const rooms = comps.filter((c) => c[0].length);
  if (rooms.some((c) => c[0].length !== c[1].length)) return '';
  let mv = '';
  let p = p0;
  const dn = rooms.map(() => 0);
  for (let ri = 0; ri < rooms.length; ri++) {
    const rw = V.slice();
    rooms.forEach((o, oi) => {
      if (oi !== ri) for (const s of dn[oi] ? o[1] : o[0]) rw[s] = 1;
    });
    const r = SR(w, rw, rooms[ri][1], rooms[ri][0], p);
    if (r === null) return '';
    mv += r;
    for (const c of r.toLowerCase()) p += c === 'u' ? -w : c === 'd' ? w : c === 'l' ? -1 : 1;
    dn[ri] = 1;
  }
  return mv;
};
for (const id of R(I).sort()) {
  let mv = '';
  try {
    mv = SV(F(J(I, id, 'level.txt'), 'utf8'));
  } catch {
    mv = '';
  }
  M(J(O, id), { recursive: true });
  W(J(O, id, 'solution.txt'), mv + '\n');
}
