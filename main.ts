// Sokoban solver: push-optimal, room by room.
//
// A generated level is a one-wide hallway with one to four rooms off it. A box pushed out of a room into
// the hallway can never come back, so no optimal solution moves a box out of its room, and a level's push
// optimum is the sum of its rooms' optima. So: split the level into its rooms and solve each with an A*
// search over pushes — a state is the room's boxes plus the player's reachable region (named by its lowest
// square), walking is free, only pushes count. The estimate is a minimum matching of boxes to goals over
// precomputed lone-box push distances, which never overestimates, so the first solution found is optimal.
// Pushes onto squares no goal is reachable from, and pushes freezing a 2x2 block of walls and boxes around
// a box off its goal, are pruned. Budgets count expanded states, never time, and iteration orders are
// fixed, so two runs write the same bytes. If the optimal search ever ran out of budget (it does not on
// generated rooms, which the generator verified with a plain breadth-first search), a greedy search still
// returns a valid solution for the room rather than none.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The four directions in a fixed order: [dx, dy, walk letter, push letter]. */
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];
/** Expanded-state budgets: per room, for a whole level at once, and across the whole run. */
const ROOM_BUDGET = 800_000;
const LEVEL_BUDGET = 2_000_000;
let workLeft = 5_000_000;

interface Level {
  w: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** A level in XSB text; squares outside the rows and on the outermost frame count as walls. */
function parseLevel(text: string): Level | null {
  const rows = text.split('\n').filter((r) => r.length > 0);
  if (rows.length === 0) return null;
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let x = 0; x < w; x++) {
    wall[x] = 1;
    wall[(h - 1) * w + x] = 1;
  }
  for (let y = 0; y < h; y++) {
    wall[y * w] = 1;
    wall[y * w + w - 1] = 1;
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, wall, goals, boxes, player };
}

/** Flood-fill scratch: `seen` stamped per fill, `stack` the frontier. */
interface Scratch {
  seen: Int32Array;
  stamp: number;
  stack: Int32Array;
}
const scratch = (size: number): Scratch => ({ seen: new Int32Array(size), stamp: 0, stack: new Int32Array(size) });

/** Mark the region the player reaches from `from` around walls and the boxes in `occ`; return its lowest square. */
function flood(sc: Scratch, w: number, wall: Uint8Array, occ: Uint8Array, from: number): number {
  const stamp = ++sc.stamp;
  let low = from;
  let n = 0;
  sc.stack[n++] = from;
  sc.seen[from] = stamp;
  while (n > 0) {
    const p = sc.stack[--n];
    if (p < low) low = p;
    for (const q of [p - w, p + w, p - 1, p + 1]) {
      if (!wall[q] && !occ[q] && sc.seen[q] !== stamp) {
        sc.seen[q] = stamp;
        sc.stack[n++] = q;
      }
    }
  }
  return low;
}

/** Per goal, the fewest pushes bringing a lone box from any square onto it (walls only), -1 where impossible. */
function boxDistances(w: number, wall: Uint8Array, goals: number[]): Int32Array[] {
  const size = wall.length;
  return goals.map((g) => {
    const dist = new Int32Array(size).fill(-1);
    if (wall[g]) return dist;
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const b = queue[qi];
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const prev = b - d;
        if (prev < 0 || prev >= size || dist[prev] !== -1 || wall[prev] || wall[prev - d]) continue;
        dist[prev] = dist[b] + 1;
        queue.push(prev);
      }
    }
    return dist;
  });
}

/** Every ordering of 0..n-1, in a fixed order. */
function permutations(n: number): number[][] {
  const out: number[][] = [];
  const cur = Array.from({ length: n }, (_, i) => i);
  const build = (k: number): void => {
    if (k === n) {
      out.push(cur.slice());
      return;
    }
    for (let i = k; i < n; i++) {
      [cur[k], cur[i]] = [cur[i]!, cur[k]!];
      build(k + 1);
      [cur[k], cur[i]] = [cur[i]!, cur[k]!];
    }
  };
  build(0);
  return out;
}

/** Whether the box pushed from `from` to `t` (`occ`: boxes before the push) is frozen in a 2x2 block. */
function frozen(w: number, wall: Uint8Array, occ: Uint8Array, goalSet: Uint8Array, from: number, t: number): boolean {
  const solid = (s: number): boolean => wall[s] === 1 || s === t || (occ[s] === 1 && s !== from);
  const badBox = (s: number): boolean => (s === t || (occ[s] === 1 && s !== from)) && goalSet[s] === 0;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (solid(a) && solid(b) && solid(c) && solid(d) && (badBox(a) || badBox(b) || badBox(c) || badBox(d))) return true;
  }
  return false;
}

/** The shortest walk from `from` to `to` around walls and boxes, lower-case, in fixed direction order. */
function walk(w: number, wall: Uint8Array, occ: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occ[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

/**
 * Solve one push problem (a room, or a whole level as fallback): a solution with the fewest pushes when
 * `optimal` holds, else the first solution a greedy search finds; null when the budget runs out or the
 * problem has no solution. `wall` already counts foreign boxes as walls.
 */
function search(
  w: number,
  wall: Uint8Array,
  goals: number[],
  boxes0: number[],
  player0: number,
  budget: number,
  optimal: boolean,
): string | null {
  const size = wall.length;
  const n = boxes0.length;
  if (n === 0) return '';
  if (goals.length !== n) return null;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const dists = boxDistances(w, wall, goals);
  const perms = permutations(n);
  // The estimate: a minimum matching of boxes to goals over the lone-box distances. Admissible, because
  // every box must travel at least its matched distance and one push moves one box one square.
  const estimate = (boxes: number[]): number => {
    let best = Infinity;
    for (const perm of perms) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const d = dists[perm[i]!]![boxes[i]!]!;
        if (d < 0) {
          sum = Infinity;
          break;
        }
        sum += d;
      }
      if (sum < best) best = sum;
    }
    return best;
  };
  const h0 = estimate(boxes0);
  if (h0 === Infinity) return null;

  // Two flood scratches: one holds the region of the state being expanded, the other computes successors'.
  const sc = scratch(size);
  const scRep = scratch(size);
  const occ = new Uint8Array(size);
  for (const b of boxes0) occ[b] = 1;
  const rep0 = flood(scRep, w, wall, occ, player0);
  for (const b of boxes0) occ[b] = 0;
  // A state's key: its region's lowest square and its boxes, packed in base `size` while that stays exact.
  const numeric = Math.pow(size, n + 1) < 2 ** 53;
  const keyOf = (boxes: number[], rep: number): number | string => {
    if (!numeric) return String.fromCharCode(rep, ...boxes);
    let k = rep;
    for (const b of boxes) k = k * size + b;
    return k;
  };

  const sBoxes: number[][] = [boxes0.slice()];
  const sPlayer: number[] = [player0];
  const sParent: number[] = [-1];
  const sPush: number[] = [-1];
  const sG: number[] = [0];
  const sH: number[] = [h0];
  const best = new Map<number | string, number>([[keyOf(boxes0, rep0), 0]]);
  // The open list: a binary heap of state indexes, (f, h, insertion) for A*, (h, -g, insertion) greedy.
  const heap: number[] = [0];
  const before = (a: number, b: number): boolean => {
    if (optimal) {
      const fa = sG[a]! + sH[a]!;
      const fb = sG[b]! + sH[b]!;
      if (fa !== fb) return fa < fb;
      if (sH[a] !== sH[b]) return sH[a]! < sH[b]!;
    } else {
      if (sH[a] !== sH[b]) return sH[a]! < sH[b]!;
      if (sG[a] !== sG[b]) return sG[a]! > sG[b]!;
    }
    return a < b;
  };
  const heapPush = (s: number): void => {
    let i = heap.length;
    heap.push(s);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!before(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && before(heap[c + 1]!, heap[c]!)) c++;
        if (!before(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  // The region representative of a stored state, recomputed from its boxes and player square.
  function repOf(s: number): number {
    const bs = sBoxes[s]!;
    for (const b of bs) occ[b] = 1;
    const rep = flood(scRep, w, wall, occ, sPlayer[s]!);
    for (const b of bs) occ[b] = 0;
    return rep;
  }

  let expanded = 0;
  let found = -1;
  while (heap.length > 0) {
    const s = heapPop();
    if (best.get(keyOf(sBoxes[s]!, repOf(s))) !== s) continue;
    const boxes = sBoxes[s]!;
    if (boxes.every((b) => goalSet[b] === 1)) {
      found = s;
      break;
    }
    if (expanded >= budget || workLeft <= 0) return null;
    expanded++;
    workLeft--;
    for (const b of boxes) occ[b] = 1;
    flood(sc, w, wall, occ, sPlayer[s]!);
    const here = sc.stamp;
    for (let bi = 0; bi < n; bi++) {
      const b = boxes[bi]!;
      for (let k = 0; k < 4; k++) {
        const d = DIRS[k]![1] * w + DIRS[k]![0];
        const t = b + d;
        if (sc.seen[b - d] !== here || wall[t] || occ[t]) continue;
        if (frozen(w, wall, occ, goalSet, b, t)) continue;
        const next = boxes.slice();
        next[bi] = t;
        next.sort((x, y) => x - y);
        const h = estimate(next);
        if (h === Infinity) continue;
        occ[b] = 0;
        occ[t] = 1;
        const rep = flood(scRep, w, wall, occ, b);
        occ[t] = 0;
        occ[b] = 1;
        const key = keyOf(next, rep);
        const known = best.get(key);
        if (known !== undefined && sG[known]! <= sG[s]! + 1) continue;
        const idx = sBoxes.length;
        best.set(key, idx);
        sBoxes.push(next);
        sPlayer.push(b);
        sParent.push(s);
        sPush.push(b * 4 + k);
        sG.push(sG[s]! + 1);
        sH.push(h);
        heapPush(idx);
      }
    }
    for (const b of boxes) occ[b] = 0;
  }
  if (found < 0) return null;
  // Replay the push path from the start, walking the player to each push: the answer's moves.
  const path: number[] = [];
  for (let s = found; s > 0; s = sParent[s]!) path.push(sPush[s]!);
  path.reverse();
  let moves = '';
  let player = player0;
  const cur = boxes0.slice();
  for (const push of path) {
    const from = Math.floor(push / 4);
    const k = push % 4;
    const d = DIRS[k]![1] * w + DIRS[k]![0];
    for (const b of cur) occ[b] = 1;
    moves += walk(w, wall, occ, player, from - d);
    for (const b of cur) occ[b] = 0;
    moves += DIRS[k]![3];
    cur[cur.indexOf(from)] = from + d;
    player = from;
  }
  return moves;
}

/**
 * The level's rooms: cut the player's row's squares that no box can be pushed to a goal from (the hallway)
 * out of the floor; each part left that holds a box is a room. Null when the level is no set of rooms.
 */
function roomsOf(level: Level): { boxes: number[]; goals: number[] }[] | null {
  const { w, wall, goals, boxes, player } = level;
  const size = wall.length;
  const dists = boxDistances(w, wall, goals);
  const live = new Uint8Array(size);
  for (let s = 0; s < size; s++) if (dists.some((d) => d[s]! >= 0)) live[s] = 1;
  const row = Math.floor(player / w);
  const cut = new Uint8Array(size);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(size).fill(-1);
  const comps: { boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const idx = comps.length;
    const squares = [s];
    part[s] = idx;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i]!;
      for (const [dx, dy] of DIRS) {
        const q = p + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = idx;
          squares.push(q);
        }
      }
    }
    comps.push({ boxes: [], goals: [] });
  }
  for (const b of boxes) comps[part[b]!]!.boxes.push(b);
  for (const g of goals) {
    if (part[g]! < 0) return null;
    comps[part[g]!]!.goals.push(g);
  }
  if (comps.some((c) => c.boxes.length !== c.goals.length)) return null;
  return comps.filter((c) => c.boxes.length > 0);
}

/** The square the player ends on after `moves` from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/** The moves that solve a level in as few pushes as the rooms allow; '' when nothing was found. */
function solve(text: string): string {
  const level = parseLevel(text);
  if (level === null || level.player < 0) return '';
  if (level.boxes.length === 0) return '';
  const rooms = roomsOf(level);
  if (rooms === null) {
    return (
      search(level.w, level.wall, level.goals, level.boxes, level.player, LEVEL_BUDGET, true) ??
      search(level.w, level.wall, level.goals, level.boxes, level.player, LEVEL_BUDGET, false) ??
      ''
    );
  }
  let player = level.player;
  let moves = '';
  const done = rooms.map(() => false);
  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri]!;
    const wall = level.wall.slice();
    rooms.forEach((other, oi) => {
      if (oi === ri) return;
      for (const s of done[oi] ? other.goals : other.boxes) wall[s] = 1;
    });
    const solved =
      search(level.w, wall, room.goals, room.boxes, player, ROOM_BUDGET, true) ??
      search(level.w, wall, room.goals, room.boxes, player, ROOM_BUDGET, false);
    if (solved === null) return '';
    moves += solved;
    player = endOf(level.w, player, solved);
    done[ri] = true;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
