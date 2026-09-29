#!/usr/bin/env node
/**
 * Synthetic development data for the viewer.
 *
 * Writes a fake sequence `mock_crossing1_demo`, `index.json` and `map_crossing1.json` into
 * web/public/data/ in exactly the format described in docs/DATA_FORMAT.md.
 *
 * The scene is a 4-way intersection with four pole-mounted LiDARs. Point clouds are produced
 * by ray-casting a simple spinning-LiDAR beam pattern against the ground, curbs, buildings,
 * trees, parked cars and moving road users, so objects are only sampled on their visible
 * surfaces and occlude each other like in a real scan.
 *
 * Usage: node scripts/make-mock-data.mjs [--frames 200] [--az 900] [--channels 48] [--out <dir>] [--force]
 *
 * Refuses to write into a data folder that already holds real (non-mock) sequences unless --force.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const outArg = process.argv.indexOf('--out');
const OUT = outArg >= 0 && process.argv[outArg + 1] ? path.resolve(process.argv[outArg + 1]) : path.resolve(HERE, '../public/data');
{
  const idx = path.join(OUT, 'index.json');
  if (fs.existsSync(idx) && !process.argv.includes('--force')) {
    let real = [];
    try {
      real = (JSON.parse(fs.readFileSync(idx, 'utf8')).sequences ?? []).filter((s) => !String(s.id).startsWith('mock_'));
    } catch { /* unreadable index: treat as foreign data */ real = [{ id: '?' }]; }
    if (real.length) {
      console.error(`${OUT} already contains real sequences (${real.map((s) => s.id).join(', ')}).\n` +
        'Refusing to overwrite. Use --out <dir> to write the mock elsewhere, or --force.');
      process.exit(1);
    }
  }
}
const SEQ = 'mock_crossing1_demo';

function arg(name, def) {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? Number(process.argv[i + 1]) : def;
}
const N_FRAMES = arg('--frames', 200);
const N_AZ = arg('--az', 900);
const N_CH = arg('--channels', 48);
const FPS = 10;
const DT = 1 / FPS;
const T0 = 1732640002.0;

const GROUND_Z = 0.45;
const WALK_Z = 0.6;
const ROAD_HW = 9; // road half-width incl. parking lane
const WALK_OUT = 13; // outer edge of walkway
const CORNER_R = 4;
const ARM_END = 78;
const BOUND = 80;
const SCALE = 0.01;

// ------------------------------------------------------------------------------------------
// deterministic RNG
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
let rand = mulberry32(42);
const setSeed = (s) => (rand = mulberry32(s));
const uni = (a, b) => a + (b - a) * rand();
function gauss() {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

// ------------------------------------------------------------------------------------------
// ground regions
function isRoad(x, y) {
  const ax = Math.abs(x);
  const ay = Math.abs(y);
  if (ax <= ROAD_HW || ay <= ROAD_HW) return true;
  if (ax < WALK_OUT && ay < WALK_OUT) return Math.hypot(ax - WALK_OUT, ay - WALK_OUT) > CORNER_R;
  return false;
}
const groundAt = (x, y) => (isRoad(x, y) ? GROUND_Z : WALK_Z);

/** Arm-local coordinates: u = distance along arm, v = lateral (incoming lanes have v > 0). */
function toArm(x, y) {
  const ax = Math.abs(x);
  const ay = Math.abs(y);
  if (ax >= ay) return x >= 0 ? [x, y] : [-x, -y];
  return y >= 0 ? [y, -x] : [-y, x];
}
const ARMS = {
  E: (u, v) => [u, v],
  W: (u, v) => [-u, -v],
  N: (u, v) => [-v, u],
  S: (u, v) => [v, -u],
};

function markingIntensity(x, y) {
  const [u, v] = toArm(x, y);
  const av = Math.abs(v);
  if (av >= ROAD_HW) return 0;
  if (u >= 19.5) {
    if (av < 0.08) return 1;
    if (Math.abs(av - 3.5) < 0.07 && u % 9 < 3) return 1;
    if (Math.abs(av - 7) < 0.07 && u > 22) return 1;
  }
  if (u >= 19.2 && u <= 19.8 && v >= 0 && v <= 7) return 1;
  if (u >= 14 && u <= 18 && ((v + 55) % 1.1) < 0.55) return 1;
  return 0;
}

// ------------------------------------------------------------------------------------------
// static world
const lidars = [
  { name: 'crossing1_11_lidar', index: 0, position: [24.2, 23.1, 7.6], yawOff: 0.01 },
  { name: 'crossing1_12_lidar', index: 1, position: [-25.3, 24.4, 7.4], yawOff: 0.23 },
  { name: 'crossing1_13_lidar', index: 2, position: [-23.6, -25.2, 7.7], yawOff: 0.41 },
  { name: 'crossing1_14_lidar', index: 3, position: [25.1, -24.3, 7.5], yawOff: 0.57 },
];

/** Axis-aligned boxes: [x0, x1, y0, y1, z0, z1, intensityBase, kind] */
const aabbs = [];
setSeed(7);
for (const sx of [1, -1]) {
  for (const sy of [1, -1]) {
    // along the E-W road, two blocks
    const h1 = uni(11, 19);
    const h2 = uni(9, 16);
    aabbs.push(box(sx, sy, 34, 53, 16 + uni(0, 2), 30, h1));
    aabbs.push(box(sx, sy, 55.5, 78, 16 + uni(0, 2), 32, h2));
    // along the N-S road
    const h3 = uni(12, 22);
    aabbs.push(boxT(sx, sy, 16 + uni(0, 2), 30, 34, 60, h3));
    aabbs.push(boxT(sx, sy, 16 + uni(0, 1), 29, 62.5, 78, uni(8, 14)));
  }
}
function box(sx, sy, u0, u1, v0, v1, h) {
  const xs = [sx * u0, sx * u1].sort((a, b) => a - b);
  const ys = [sy * v0, sy * v1].sort((a, b) => a - b);
  return [xs[0], xs[1], ys[0], ys[1], WALK_Z, WALK_Z + h, uni(55, 110), 2];
}
function boxT(sx, sy, u0, u1, v0, v1, h) {
  return box(sx, sy, u0, u1, v0, v1, h);
}
// lidar poles
const poleBoxes = lidars.map((l) => [
  l.position[0] - 0.15, l.position[0] + 0.15, l.position[1] - 0.15, l.position[1] + 0.15,
  WALK_Z, l.position[2] - 0.25, 90, 4,
]);
// trees along the walkways
const trees = [];
for (const [arm, f] of Object.entries(ARMS)) {
  for (const side of [1, -1]) {
    for (let u = 24; u < 76; u += uni(9, 14)) {
      if (rand() < 0.3) continue;
      const [x, y] = f(u, side * 12.2);
      trees.push({ x, y, r: uni(1.8, 2.6), zc: WALK_Z + uni(4.3, 5.4), trunkH: uni(2.6, 3.4) });
    }
  }
  void arm;
}
for (const t of trees) {
  aabbs.push([t.x - 0.16, t.x + 0.16, t.y - 0.16, t.y + 0.16, WALK_Z, WALK_Z + t.trunkH, 45, 3]);
}

// ------------------------------------------------------------------------------------------
// objects
const SHAPES = {
  car: (l, w, h) => [
    [-l / 2, l / 2, -w / 2, w / 2, 0.28, 0.95],
    [-0.36 * l, 0.18 * l, -0.43 * w, 0.43 * w, 0.95, h],
  ],
  van: (l, w, h) => [
    [-l / 2, l / 2, -w / 2, w / 2, 0.3, 1.05],
    [-l / 2, 0.3 * l, -0.47 * w, 0.47 * w, 1.05, h],
  ],
  bus: (l, w, h) => [[-l / 2, l / 2, -w / 2, w / 2, 0.32, h]],
  truck: (l, w, h) => [
    [0.3 * l, l / 2, -w / 2, w / 2, 0.45, 0.88 * h],
    [-l / 2, 0.27 * l, -w / 2, w / 2, 0.5, h],
  ],
  cyclist: (l, w, h) => [
    [-l / 2, l / 2, -0.09, 0.09, 0.12, 0.95],
    [-0.28, 0.18, -w / 2, w / 2, 0.95, h - 0.22],
    [-0.12, 0.1, -0.11, 0.11, h - 0.22, h],
  ],
  pedestrian: (l, w, h) => [
    [-0.12, 0.12, -0.2, 0.2, 0, 0.88],
    [-0.14, 0.14, -w / 2, w / 2, 0.88, 1.5],
    [-0.1, 0.1, -0.1, 0.1, 1.5, h],
  ],
};
const DIMS = {
  car: () => [uni(4.2, 4.9), uni(1.78, 1.92), uni(1.42, 1.6)],
  van: () => [uni(5.0, 5.6), uni(1.95, 2.05), uni(1.9, 2.3)],
  bus: () => [12.0, 2.55, 3.1],
  truck: () => [9.2, 2.5, 3.4],
  cyclist: () => [1.8, 0.62, 1.75],
  pedestrian: () => [0.6, 0.62, uni(1.65, 1.85)],
};
const GROUP = {
  car: 'vehicle', van: 'vehicle', truck: 'vehicle', bus: 'vehicle', trailer: 'vehicle',
  motorcycle: 'vehicle', cyclist: 'vru', escooter: 'vru', pedestrian: 'vru', other: 'other',
};
const VEHICLE_CLASSES = ['car', 'van', 'truck', 'bus', 'trailer', 'motorcycle'];

const objects = [];
function addObject(cls, extra) {
  const [l, w, h] = DIMS[cls]();
  const o = { id: objects.length, cls, l, w, h, parts: SHAPES[cls](l, w, h), base: uni(35, 115), ...extra };
  objects.push(o);
  return o;
}

// parked cars in the parking lanes (v in [7, 9] on both sides of each arm)
setSeed(11);
for (const [armName, f] of Object.entries(ARMS)) {
  for (const side of [1, -1]) {
    let u = 22.5 + uni(0, 4);
    let n = 0;
    while (u < 74 && n < 3) {
      const cls = rand() < 0.18 ? 'van' : 'car';
      const o = addObject(cls, { parked: true });
      const c = u + o.l / 2;
      if (c + o.l / 2 > 76) { objects.pop(); break; }
      const [x, y] = f(c, side * 8);
      // heading along traffic direction of that side (right-hand traffic)
      const [hx, hy] = f(side > 0 ? -1 : 1, 0);
      const [ox, oy] = f(0, 0);
      o.x = x + uni(-0.05, 0.05);
      o.y = y + uni(-0.12, 0.12);
      o.yaw = Math.atan2(hy - oy, hx - ox) + uni(-0.03, 0.03);
      o.arm = armName;
      u = c + o.l / 2 + uni(1.2, 9);
      if (rand() < 0.25) u += uni(8, 14);
      n++;
    }
  }
}

// moving road users following paths
function pathBuilder(x, y, headingDeg) {
  const pts = [[x, y]];
  let h = (headingDeg * Math.PI) / 180;
  let cx = x;
  let cy = y;
  const step = 0.25;
  const api = {
    straight(len) {
      const n = Math.max(1, Math.round(len / step));
      for (let i = 1; i <= n; i++) pts.push([cx + Math.cos(h) * (len * i) / n, cy + Math.sin(h) * (len * i) / n]);
      cx += Math.cos(h) * len;
      cy += Math.sin(h) * len;
      return api;
    },
    /** go straight until the coordinate along the heading axis reaches `value` */
    to(value) {
      const alongX = Math.abs(Math.cos(h)) > Math.abs(Math.sin(h));
      const d = alongX ? (value - cx) / Math.cos(h) : (value - cy) / Math.sin(h);
      return api.straight(Math.max(0, d));
    },
    turn(r, deg) {
      const a = (deg * Math.PI) / 180;
      const s = Math.sign(a);
      const ccx = cx - s * Math.sin(h) * r;
      const ccy = cy + s * Math.cos(h) * r;
      const n = Math.max(2, Math.round((Math.abs(a) * r) / step));
      const a0 = Math.atan2(cy - ccy, cx - ccx);
      for (let i = 1; i <= n; i++) {
        const t = a0 + (a * i) / n;
        pts.push([ccx + Math.cos(t) * r, ccy + Math.sin(t) * r]);
      }
      cx = pts[pts.length - 1][0];
      cy = pts[pts.length - 1][1];
      h += a;
      return api;
    },
    build() {
      const s = [0];
      for (let i = 1; i < pts.length; i++) s.push(s[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
      return { pts, s, length: s[s.length - 1] };
    },
  };
  return api;
}
function samplePath(p, s) {
  if (s <= 0) s = 0;
  let lo = 0;
  let hi = p.s.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (p.s[mid] <= s) lo = mid; else hi = mid;
  }
  const seg = p.s[hi] - p.s[lo] || 1;
  const t = clamp((s - p.s[lo]) / seg, 0, 1);
  const a = p.pts[lo];
  const b = p.pts[hi];
  return { x: a[0] + (b[0] - a[0]) * t, y: a[1] + (b[1] - a[1]) * t, yaw: Math.atan2(b[1] - a[1], b[0] - a[0]) };
}

// traffic light: E-W green until frame 95, N-S green from frame 105
const EW_RED_FROM = 95;
const NS_GREEN_AT = 105;
const movers = [
  // cls, path, start frame, cruise speed, stop {s-coordinate of stop line along axis, until frame}
  { cls: 'car', path: pathBuilder(-60, -5.25, 0).straight(150), t0: 0, v: 11 },
  { cls: 'car', path: pathBuilder(-32, -1.75, 0).straight(125), t0: 0, v: 12.5 },
  { cls: 'car', path: pathBuilder(72, 1.75, 180).to(-1.75 + 10).turn(10, 90).straight(80), t0: 4, v: 10 },
  { cls: 'bus', path: pathBuilder(5.25, -68, 90).straight(160), t0: 0, v: 8.5, stopY: -19.5, until: NS_GREEN_AT + 3 },
  { cls: 'car', path: pathBuilder(1.75, -76, 90).straight(160), t0: 8, v: 11, stopY: -19.5, until: NS_GREEN_AT },
  { cls: 'car', path: pathBuilder(-5.25, 82, -90).to(9.25).turn(4, -90).straight(80), t0: 98, v: 10, stopY: 19.5, until: NS_GREEN_AT },
  { cls: 'truck', path: pathBuilder(84, 5.25, 180).straight(170), t0: 18, v: 8.8 },
  { cls: 'car', path: pathBuilder(-82, -1.75, 0).straight(170), t0: 55, v: 12, stopX: -19.5, until: 9999, redFrom: EW_RED_FROM },
  { cls: 'car', path: pathBuilder(-82, -5.25, 0).straight(170), t0: 88, v: 11, stopX: -19.5, until: 9999, redFrom: EW_RED_FROM },
  { cls: 'van', path: pathBuilder(5.25, -82, 90).to(-9.25).turn(4, -90).straight(80), t0: 108, v: 10.5, stopY: -19.5, until: NS_GREEN_AT },
  { cls: 'car', path: pathBuilder(-1.75, 84, -90).straight(170), t0: 128, v: 12, stopY: 19.5, until: NS_GREEN_AT },
  { cls: 'cyclist', path: pathBuilder(-70, -9.75, 0).straight(150), t0: 0, v: 5.2 },
  { cls: 'cyclist', path: pathBuilder(72, 9.75, 180).straight(150), t0: 35, v: 5.6 },
  { cls: 'pedestrian', path: pathBuilder(-16, -12, 90).straight(24), t0: 20, v: 1.35 },
  { cls: 'pedestrian', path: pathBuilder(-42, 11.4, 0).straight(40), t0: 0, v: 1.25 },
  { cls: 'pedestrian', path: pathBuilder(12, 16, 180).straight(24), t0: 6, v: 1.5 },
  { cls: 'pedestrian', path: pathBuilder(16.5, 12.5, -90).straight(25), t0: 120, v: 1.4 },
];

setSeed(21);
for (const m of movers) {
  const o = addObject(m.cls, { parked: false });
  const p = m.path.build();
  o.states = new Array(N_FRAMES).fill(null);
  let s = 0;
  let v = m.v;
  const aMax = m.cls === 'bus' || m.cls === 'truck' ? 1.4 : 2.4;
  const bMax = 3.2;
  for (let f = 0; f < N_FRAMES; f++) {
    if (f < m.t0) continue;
    if (s > p.length) break;
    let vt = m.v;
    const hasStop = m.stopX !== undefined || m.stopY !== undefined;
    const red = hasStop && f >= (m.redFrom ?? 0) && f < m.until;
    if (red) {
      // distance from the vehicle front to the stop line
      const st = samplePath(p, s);
      const fx = st.x + (Math.cos(st.yaw) * o.l) / 2;
      const fy = st.y + (Math.sin(st.yaw) * o.l) / 2;
      const before = m.stopX !== undefined ? (m.stopX - st.x) * Math.cos(st.yaw) > 0 : (m.stopY - st.y) * Math.sin(st.yaw) > 0;
      const front = m.stopX !== undefined ? Math.abs(m.stopX - fx) : Math.abs(m.stopY - fy);
      if (before) vt = Math.min(vt, Math.sqrt(Math.max(0, 2 * 2.2 * (front - 1.0))));
    }
    v += clamp(vt - v, -bMax * DT, aMax * DT);
    if (v < 0.05 && vt === 0) v = 0;
    const st = samplePath(p, s);
    const gz = groundAt(st.x, st.y);
    if (Math.abs(st.x) < ARM_END && Math.abs(st.y) < ARM_END) {
      o.states[f] = { x: st.x, y: st.y, yaw: st.yaw, speed: v, gz };
    }
    s += v * DT;
  }
}
for (const o of objects) {
  if (o.parked) {
    o.gz = groundAt(o.x, o.y);
    o.states = new Array(N_FRAMES).fill({ x: o.x, y: o.y, yaw: o.yaw, speed: 0, gz: o.gz });
  }
}

// ------------------------------------------------------------------------------------------
// ray casting
function rayAabb(ox, oy, oz, dx, dy, dz, b, tMax) {
  let t0 = 0;
  let t1 = tMax;
  const o = [ox, oy, oz];
  const d = [dx, dy, dz];
  for (let k = 0; k < 3; k++) {
    const lo = b[2 * k];
    const hi = b[2 * k + 1];
    if (Math.abs(d[k]) < 1e-12) {
      if (o[k] < lo || o[k] > hi) return Infinity;
    } else {
      let ta = (lo - o[k]) / d[k];
      let tb = (hi - o[k]) / d[k];
      if (ta > tb) [ta, tb] = [tb, ta];
      if (ta > t0) t0 = ta;
      if (tb < t1) t1 = tb;
      if (t0 > t1) return Infinity;
    }
  }
  return t0 > 1e-6 ? t0 : Infinity;
}

/** returns {t, part} in object-local frame */
function rayObject(ox, oy, oz, dx, dy, dz, obj, st, tMax) {
  const c = Math.cos(-st.yaw);
  const s = Math.sin(-st.yaw);
  const px = ox - st.x;
  const py = oy - st.y;
  const lx = px * c - py * s;
  const ly = px * s + py * c;
  const lz = oz - st.gz;
  const ldx = dx * c - dy * s;
  const ldy = dx * s + dy * c;
  let best = tMax;
  let bestPart = -1;
  for (let i = 0; i < obj.parts.length; i++) {
    const t = rayAabb(lx, ly, lz, ldx, ldy, dz, obj.parts[i], best);
    if (t < best) {
      best = t;
      bestPart = i;
    }
  }
  if (bestPart < 0) return null;
  return { t: best, hx: lx + ldx * best, hy: ly + ldy * best, hz: lz + dz * best, part: bestPart };
}

function objectIntensity(obj, hit) {
  const { hx, hz, part } = hit;
  if (obj.cls === 'car' || obj.cls === 'van') {
    if (Math.abs(hx) > obj.l / 2 - 0.06 && hz > 0.38 && hz < 0.62 && Math.abs(hit.hy) < 0.3) return 235;
    if (part === 1 && hz > 1.05) return clamp(14 + gauss() * 6, 2, 60);
  }
  if (obj.cls === 'bus' && hz > 1.2 && hz < 2.6) return clamp(20 + gauss() * 8, 2, 60);
  if (obj.cls === 'pedestrian' || obj.cls === 'cyclist') return clamp(70 + gauss() * 18, 5, 200);
  return clamp(obj.base + gauss() * 9, 5, 250);
}

const MAX_RANGE = 160;
const elev = [];
for (let c = 0; c < N_CH; c++) elev.push(((-32 + (40 * c) / (N_CH - 1)) * Math.PI) / 180);
const N_RAYS = N_CH * N_AZ;

// kinds: 0 none, 1 ground, 2 wall, 3 tree, 4 pole, 5 object
const staticCast = lidars.map((L, li) => {
  const r = new Float32Array(N_RAYS).fill(Infinity);
  const kind = new Uint8Array(N_RAYS);
  const obj = new Int16Array(N_RAYS).fill(-1);
  const inten = new Uint8Array(N_RAYS);
  const dirs = new Float32Array(N_RAYS * 3);
  const [ox, oy, oz] = L.position;
  setSeed(100 + li);
  const parked = objects.filter((o) => o.parked);
  for (let c = 0; c < N_CH; c++) {
    const ce = Math.cos(elev[c]);
    const se = Math.sin(elev[c]);
    for (let a = 0; a < N_AZ; a++) {
      const az = (2 * Math.PI * a) / N_AZ + L.yawOff;
      const dx = ce * Math.cos(az);
      const dy = ce * Math.sin(az);
      const dz = se;
      const i = c * N_AZ + a;
      dirs[3 * i] = dx;
      dirs[3 * i + 1] = dy;
      dirs[3 * i + 2] = dz;
      let best = MAX_RANGE;
      let k = 0;
      let o = -1;
      let I = 0;
      // ground
      if (dz < -1e-4) {
        let t = (WALK_Z - oz) / dz;
        let hx = ox + dx * t;
        let hy = oy + dy * t;
        if (isRoad(hx, hy)) {
          t = (GROUND_Z - oz) / dz;
          hx = ox + dx * t;
          hy = oy + dy * t;
        }
        if (t < best) {
          best = t;
          k = 1;
          const road = isRoad(hx, hy);
          I = markingIntensity(hx, hy) ? uni(175, 235) : road ? uni(12, 34) : uni(40, 62);
        }
      }
      for (const b of aabbs) {
        const t = rayAabb(ox, oy, oz, dx, dy, dz, b, best);
        if (t < best) {
          best = t;
          k = b[7];
          I = clamp(b[6] + gauss() * 10, 5, 250);
        }
      }
      for (let p = 0; p < poleBoxes.length; p++) {
        if (p === li) continue;
        const t = rayAabb(ox, oy, oz, dx, dy, dz, poleBoxes[p], best);
        if (t < best) {
          best = t;
          k = 4;
          I = 120;
        }
      }
      // tree canopies (leafy: stochastic hits inside the sphere)
      for (const tr of trees) {
        const px = ox - tr.x;
        const py = oy - tr.y;
        const pz = oz - tr.zc;
        const bq = px * dx + py * dy + pz * dz;
        const cq = px * px + py * py + pz * pz - tr.r * tr.r;
        const disc = bq * bq - cq;
        if (disc <= 0) continue;
        const sq = Math.sqrt(disc);
        const tin = -bq - sq;
        const tout = -bq + sq;
        if (tin <= 0 || tin >= best) continue;
        if (rand() < 0.35) continue;
        const t = tin + (tout - tin) * Math.pow(rand(), 2) * 0.6;
        if (t < best) {
          best = t;
          k = 3;
          I = uni(25, 70);
        }
      }
      for (const ob of parked) {
        const d = Math.hypot(ob.x - ox, ob.y - oy);
        if (d - 4 > best) continue;
        const hit = rayObject(ox, oy, oz, dx, dy, dz, ob, ob.states[0], best);
        if (hit && hit.t < best) {
          best = hit.t;
          k = 5;
          o = ob.id;
          I = objectIntensity(ob, hit);
        }
      }
      if (k) {
        r[i] = best;
        kind[i] = k;
        obj[i] = o;
        inten[i] = I;
      }
    }
  }
  return { r, kind, obj, inten, dirs };
});

// ------------------------------------------------------------------------------------------
// per-frame truth, detections, points
const seqDir = path.join(OUT, SEQ);
fs.rmSync(seqDir, { recursive: true, force: true });
fs.mkdirSync(path.join(seqDir, 'points'), { recursive: true });

const inBounds = (st) => st && Math.abs(st.x) < ARM_END && Math.abs(st.y) < ARM_END;

// detector behaviour per object
setSeed(77);
const detId = new Map();
let nextId = 1;
const order = objects.map((o) => o.id).sort(() => rand() - 0.5);
for (const id of order) detId.set(id, nextId++);
const ID_SWITCH_OBJ = objects.find((o) => !o.parked && o.cls === 'car' && o.states.filter(Boolean).length > 120)?.id;
const ID_SWITCH_FRAME = 118;
const switchedId = nextId++;
const confusedVan = objects.find((o) => o.parked && o.cls === 'van')?.id;
const bias = new Map(objects.map((o) => [o.id, { dx: gauss() * 0.12, dy: gauss() * 0.12, dl: gauss() * 0.05, dyaw: gauss() * 0.02 }]));
const fpDefs = [
  { id: nextId++, cls: 'car', box: [-43.5, -11.6, WALK_Z + 0.7, 3.9, 1.7, 1.4, 0.05], from: 40, to: 71, score: [0.32, 0.48] },
  { id: nextId++, cls: 'pedestrian', box: [23.2, 12.4, WALK_Z + 0.85, 0.6, 0.6, 1.7, 1.2], from: 120, to: 138, score: [0.3, 0.45] },
  { id: nextId++, cls: 'car', box: [52.0, 22.0, WALK_Z + 0.75, 4.4, 1.8, 1.5, -1.5], from: 150, to: 167, score: [0.3, 0.42] },
];

const detFrames = [];
const gtFrames = [];
const frameMeta = [];
const maxRays = N_RAYS;
const cur = { r: new Float32Array(maxRays), kind: new Uint8Array(maxRays), obj: new Int16Array(maxRays), inten: new Uint8Array(maxRays) };
const recBuf = Buffer.alloc(N_RAYS * lidars.length * 8);

let totalPts = 0;
const t0 = Date.now();
for (let f = 0; f < N_FRAMES; f++) {
  setSeed(1000 + f);
  const hits = new Int32Array(objects.length);
  const perLidar = [];
  for (let li = 0; li < lidars.length; li++) {
    const S = staticCast[li];
    cur.r.set(S.r);
    cur.kind.set(S.kind);
    cur.obj.set(S.obj);
    cur.inten.set(S.inten);
    const [ox, oy, oz] = lidars[li].position;
    for (const o of objects) {
      if (o.parked) continue;
      const st = o.states[f];
      if (!inBounds(st)) continue;
      const R = Math.hypot(o.l, o.w) / 2 + 0.2;
      const d = Math.hypot(st.x - ox, st.y - oy);
      if (d < R + 0.2) continue;
      const half = Math.asin(Math.min(1, R / d));
      const azc = Math.atan2(st.y - oy, st.x - ox) - lidars[li].yawOff;
      const a0 = Math.floor(((azc - half) / (2 * Math.PI)) * N_AZ);
      const a1 = Math.ceil(((azc + half) / (2 * Math.PI)) * N_AZ);
      for (let aa = a0; aa <= a1; aa++) {
        const a = ((aa % N_AZ) + N_AZ) % N_AZ;
        for (let c = 0; c < N_CH; c++) {
          const i = c * N_AZ + a;
          const hit = rayObject(ox, oy, oz, S.dirs[3 * i], S.dirs[3 * i + 1], S.dirs[3 * i + 2], o, st, Math.min(cur.r[i], MAX_RANGE));
          if (hit) {
            cur.r[i] = hit.t;
            cur.kind[i] = 6;
            cur.obj[i] = o.id;
            cur.inten[i] = objectIntensity(o, hit);
          }
        }
      }
    }
    // snapshot for emission after detection decisions
    const snap = { r: cur.r.slice(), kind: cur.kind.slice(), obj: cur.obj.slice(), inten: cur.inten.slice() };
    for (let i = 0; i < N_RAYS; i++) if (snap.obj[i] >= 0 && Number.isFinite(snap.r[i])) hits[snap.obj[i]]++;
    perLidar.push(snap);
  }

  // ground truth
  const gt = [];
  for (const o of objects) {
    const st = o.states[f];
    if (!inBounds(st)) continue;
    gt.push({
      id: 1000 + o.id, cls: o.cls, ignore: hits[o.id] < 5, trackable: true,
      box: [st.x, st.y, st.gz + o.h / 2, o.l, o.w, o.h, st.yaw].map(r4),
      score: 1.0, moving: st.speed > 1.0, speed: r2(st.speed), n_points: hits[o.id],
    });
  }
  gtFrames.push(gt);

  // detections
  const det = [];
  const detected = new Uint8Array(objects.length);
  for (const o of objects) {
    const st = o.states[f];
    if (!inBounds(st)) continue;
    const n = hits[o.id];
    const pDet = n >= 25 ? 0.99 : n >= 10 ? 0.85 : n >= 4 ? 0.45 : 0.04;
    if (rand() > pDet) continue;
    detected[o.id] = 1;
    const b = bias.get(o.id);
    const noise = n >= 25 ? 1 : 2.2;
    const x = st.x + b.dx + gauss() * 0.07 * noise;
    const y = st.y + b.dy + gauss() * 0.07 * noise;
    const l = o.l * (1 + b.dl + gauss() * 0.02);
    const w = o.w * (1 + gauss() * 0.02);
    const h = o.h * (1 + gauss() * 0.03);
    const yaw = st.yaw + b.dyaw + gauss() * 0.02;
    const speed = Math.max(0, st.speed + gauss() * 0.18);
    let id = detId.get(o.id);
    if (o.id === ID_SWITCH_OBJ && f >= ID_SWITCH_FRAME) id = switchedId;
    const cls = o.id === confusedVan ? 'car' : o.cls;
    det.push({
      id, cls,
      box: [x, y, st.gz + h / 2 + gauss() * 0.03, l, w, h, yaw].map(r4),
      score: r3(clamp(0.5 + 0.45 * (1 - Math.exp(-n / 50)) + gauss() * 0.03, 0.2, 0.99)),
      moving: speed > 1.0, speed: r2(speed), n_points: n,
    });
  }
  for (const fp of fpDefs) {
    if (f < fp.from || f > fp.to) continue;
    det.push({ id: fp.id, cls: fp.cls, box: fp.box.map((v, k) => r4(v + (k < 2 ? gauss() * 0.05 : 0))), score: r3(uni(...fp.score)), moving: false, speed: r2(Math.abs(gauss() * 0.2)), n_points: Math.round(uni(8, 30)) });
  }
  detFrames.push(det);

  // points
  let n = 0;
  for (let li = 0; li < lidars.length; li++) {
    const S = staticCast[li];
    const P = perLidar[li];
    const [ox, oy, oz] = lidars[li].position;
    for (let i = 0; i < N_RAYS; i++) {
      const r = P.r[i];
      if (!Number.isFinite(r) || r >= MAX_RANGE) continue;
      if (rand() < 0.035) continue; // dropouts
      const rr = r + gauss() * 0.012;
      const x = ox + S.dirs[3 * i] * rr;
      const y = oy + S.dirs[3 * i + 1] * rr;
      const z = oz + S.dirs[3 * i + 2] * rr;
      if (Math.abs(x) > BOUND || Math.abs(y) > BOUND || z < -2 || z > 30) continue;
      const k = P.kind[i];
      const o = P.obj[i];
      let flags = li & 3;
      if (k === 6) flags |= 4;
      if (k === 1) flags |= 8;
      if (o >= 0 && detected[o]) flags |= 16;
      const off = n * 8;
      recBuf.writeInt16LE(Math.round(x / SCALE), off);
      recBuf.writeInt16LE(Math.round(y / SCALE), off + 2);
      recBuf.writeInt16LE(Math.round(z / SCALE), off + 4);
      recBuf.writeUInt8(P.inten[i], off + 6);
      recBuf.writeUInt8(flags, off + 7);
      n++;
    }
  }
  const file = `points/${String(f).padStart(6, '0')}.bin`;
  fs.writeFileSync(path.join(seqDir, file), recBuf.subarray(0, n * 8));
  frameMeta.push({ file, n_points: n });
  totalPts += n;
  if (f % 20 === 0) process.stdout.write(`  frame ${f}/${N_FRAMES}: ${n} points\n`);
}

function r2(v) { return Math.round(v * 100) / 100; }
function r3(v) { return Math.round(v * 1000) / 1000; }
function r4(v) { return Math.round(v * 10000) / 10000; }

// ------------------------------------------------------------------------------------------
// tracks / counts / unique
function summarise(frames, source) {
  const tracks = {};
  const counts = [];
  for (let f = 0; f < frames.length; f++) {
    const c = { vehicle: 0, vru: 0, other: 0, moving: 0, parked: 0, by_class: {} };
    for (const d of frames[f]) {
      const g = GROUP[d.cls] ?? 'other';
      c[g]++;
      if (g === 'vehicle') {
        if (d.moving) c.moving++; else c.parked++;
        c.by_class[d.cls] = (c.by_class[d.cls] ?? 0) + 1;
      }
      const t = (tracks[d.id] ??= { cls: d.cls, group: g, first: f, last: f, moving: false, max_speed: 0, n_frames: 0 });
      t.last = f;
      t.n_frames++;
      t.max_speed = Math.max(t.max_speed, d.speed);
      if (d.moving) t.moving = true;
    }
    counts.push(c);
  }
  const unique = { vehicle: 0, vru: 0, other: 0, by_class: {} };
  for (const t of Object.values(tracks)) {
    t.max_speed = r2(t.max_speed);
    unique[t.group]++;
    if (t.group === 'vehicle') unique.by_class[t.cls] = (unique.by_class[t.cls] ?? 0) + 1;
  }
  return { source, frames, tracks, counts, unique };
}
const detections = summarise(detFrames, 'detector');
const gtData = summarise(gtFrames, 'gt');

// ------------------------------------------------------------------------------------------
// metrics (BEV centre-distance matching as a stand-in for IoU)
function evaluate() {
  const all = [];
  let nGt = 0;
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let idsw = 0;
  const lastMatch = new Map();
  const perClass = {};
  const byRange = [[0, 20], [20, 40], [40, 60], [60, 80]].map(([a, b]) => ({ a, b, n: 0, hit: 0 }));
  for (let f = 0; f < N_FRAMES; f++) {
    const G = gtFrames[f].filter((g) => GROUP[g.cls] === 'vehicle');
    const D = detFrames[f].filter((d) => GROUP[d.cls] === 'vehicle').sort((a, b) => b.score - a.score);
    nGt += G.length;
    const used = new Set();
    for (const g of G) {
      (perClass[g.cls] ??= { n_gt: 0, n_pred: 0 }).n_gt++;
    }
    for (const d of D) {
      (perClass[d.cls] ??= { n_gt: 0, n_pred: 0 }).n_pred++;
      let best = -1;
      let bd = 2.0;
      G.forEach((g, gi) => {
        if (used.has(gi)) return;
        const dd = Math.hypot(g.box[0] - d.box[0], g.box[1] - d.box[1]);
        if (dd < bd) { bd = dd; best = gi; }
      });
      if (best >= 0) {
        used.add(best);
        tp++;
        all.push({ s: d.score, tp: 1 });
        const gid = G[best].id;
        if (lastMatch.has(gid) && lastMatch.get(gid) !== d.id) idsw++;
        lastMatch.set(gid, d.id);
      } else {
        fp++;
        all.push({ s: d.score, tp: 0 });
      }
    }
    G.forEach((g, gi) => {
      const r = Math.hypot(g.box[0], g.box[1]);
      const bin = byRange.find((b) => r >= b.a && r < b.b);
      if (bin) { bin.n++; if (used.has(gi)) bin.hit++; }
      if (!used.has(gi)) fn++;
    });
  }
  all.sort((a, b) => b.s - a.s);
  const rec = [0];
  const prec = [1];
  let ctp = 0;
  let cfp = 0;
  for (const e of all) {
    if (e.tp) ctp++; else cfp++;
    rec.push(ctp / nGt);
    prec.push(ctp / (ctp + cfp));
  }
  // monotone precision envelope + AP
  for (let i = prec.length - 2; i >= 0; i--) prec[i] = Math.max(prec[i], prec[i + 1]);
  let ap = 0;
  for (let i = 1; i < rec.length; i++) ap += (rec[i] - rec[i - 1]) * prec[i];
  // downsample PR curve
  const pr = { recall: [], precision: [] };
  const step = Math.max(1, Math.floor(rec.length / 60));
  for (let i = 0; i < rec.length; i += step) { pr.recall.push(r3(rec[i])); pr.precision.push(r3(prec[i])); }
  pr.recall.push(r3(rec[rec.length - 1]));
  pr.precision.push(r3(prec[prec.length - 1]));
  const precision = tp / (tp + fp);
  const recall = tp / (tp + fn);
  const pc = {};
  for (const [k, v] of Object.entries(perClass)) {
    const base = k === 'car' ? ap : ap * uni(0.7, 0.95);
    pc[k] = { ap: { '0.3': r3(base), '0.5': r3(base * 0.82) }, n_gt: v.n_gt, n_pred: v.n_pred };
  }
  const pred = detections.counts.map((c) => c.vehicle);
  const gtc = gtData.counts.map((c) => c.vehicle);
  const err = pred.map((p, i) => p - gtc[i]);
  const mae = err.reduce((a, e) => a + Math.abs(e), 0) / err.length;
  const rmse = Math.sqrt(err.reduce((a, e) => a + e * e, 0) / err.length);
  const bias = err.reduce((a, e) => a + e, 0) / err.length;
  const within1 = err.filter((e) => Math.abs(e) <= 1).length / err.length;
  return {
    detection: {
      iou_thresholds: [0.3, 0.5],
      vehicle: { ap: { '0.3': r3(ap), '0.5': r3(ap * 0.83) }, precision: r3(precision), recall: r3(recall), f1: r3((2 * precision * recall) / (precision + recall)) },
      per_class: pc,
      by_range: byRange.map((b) => ({ range: `${b.a}-${b.b}`, recall: r3(b.n ? b.hit / b.n : 0) })),
      pr_curve: pr,
    },
    counting: { mae: r3(mae), rmse: r3(rmse), bias: r3(bias), within_1: r3(within1), pred, gt: gtc },
    tracking: {
      mota: r3(1 - (fn + fp + idsw) / nGt), motp: r3(0.78 + uni(-0.03, 0.03)), id_switches: idsw,
      unique_pred: detections.unique.vehicle, unique_gt: gtData.unique.vehicle,
    },
  };
}
const metrics = evaluate();

// ------------------------------------------------------------------------------------------
// HD map
function mapJson() {
  const lines = [];
  const areas = [];
  const L = (kind, subtype, pts, z = GROUND_Z) => lines.push({ kind, subtype, points: pts.map(([x, y]) => [r3(x), r3(y), z]) });
  const A = (kind, poly) => areas.push({ kind, polygon: poly.map(([x, y]) => [r3(x), r3(y)]) });
  for (const [, f] of Object.entries(ARMS)) {
    const T = (pts) => pts.map(([u, v]) => f(u, v));
    // lane markings
    L('line_thin', 'solid', T([[19.5, 0], [ARM_END, 0]]));
    L('line_thin', 'dashed', T([[19.5, 3.5], [ARM_END, 3.5]]));
    L('line_thin', 'dashed', T([[19.5, -3.5], [ARM_END, -3.5]]));
    L('line_thin', 'solid', T([[22, 7], [ARM_END, 7]]));
    L('line_thin', 'solid', T([[22, -7], [ARM_END, -7]]));
    L('stop_line', 'solid', T([[19.5, 0], [19.5, 7]]));
    L('zebra_marking', 'solid', T([[14, -ROAD_HW], [14, ROAD_HW]]));
    L('zebra_marking', 'solid', T([[18, -ROAD_HW], [18, ROAD_HW]]));
    // virtual lane continuations through the intersection
    L('virtual', '', T([[ROAD_HW, 3.5], [19.5, 3.5]]));
    L('virtual', '', T([[ROAD_HW, -3.5], [19.5, -3.5]]));
    // walkway outer edge
    L('curbstone', 'low', T([[WALK_OUT, WALK_OUT], [ARM_END, WALK_OUT]]), WALK_Z);
    L('curbstone', 'low', T([[WALK_OUT, -WALK_OUT], [ARM_END, -WALK_OUT]]), WALK_Z);
    // lanes
    A('road', T([[ROAD_HW, 0], [ARM_END, 0], [ARM_END, 3.5], [ROAD_HW, 3.5]]));
    A('road', T([[ROAD_HW, 3.5], [ARM_END, 3.5], [ARM_END, 7], [ROAD_HW, 7]]));
    A('road', T([[ROAD_HW, 0], [ARM_END, 0], [ARM_END, -3.5], [ROAD_HW, -3.5]]));
    A('road', T([[ROAD_HW, -3.5], [ARM_END, -3.5], [ARM_END, -7], [ROAD_HW, -7]]));
    A('parking', T([[22, 7], [ARM_END, 7], [ARM_END, 9], [22, 9]]));
    A('parking', T([[22, -7], [ARM_END, -7], [ARM_END, -9], [22, -9]]));
    A('crosswalk', T([[14, -ROAD_HW], [18, -ROAD_HW], [18, ROAD_HW], [14, ROAD_HW]]));
    A('walkway', T([[WALK_OUT, ROAD_HW], [ARM_END, ROAD_HW], [ARM_END, WALK_OUT], [WALK_OUT, WALK_OUT]]));
    A('walkway', T([[WALK_OUT, -ROAD_HW], [ARM_END, -ROAD_HW], [ARM_END, -WALK_OUT], [WALK_OUT, -WALK_OUT]]));
  }
  // bicycle lanes on E-W arms (on the walkway)
  for (const f of [ARMS.E, ARMS.W]) {
    for (const s of [1, -1]) {
      const T = (pts) => pts.map(([u, v]) => f(u, v));
      A('bicycle_lane', T([[WALK_OUT, s * 9.05], [ARM_END, s * 9.05], [ARM_END, s * 10.5], [WALK_OUT, s * 10.5]]));
    }
  }
  A('road', [[-ROAD_HW, -ROAD_HW], [ROAD_HW, -ROAD_HW], [ROAD_HW, ROAD_HW], [-ROAD_HW, ROAD_HW]]);
  // corner walkways and road fillets
  for (const sx of [1, -1]) {
    for (const sy of [1, -1]) {
      const arc = [];
      for (let i = 0; i <= 12; i++) {
        const a = -Math.PI / 2 - (Math.PI / 2) * (i / 12);
        arc.push([sx * (WALK_OUT + Math.cos(a) * CORNER_R), sy * (WALK_OUT + Math.sin(a) * CORNER_R)]);
      }
      A('walkway', [...arc, [sx * WALK_OUT, sy * WALK_OUT]]);
      A('road', [[sx * ROAD_HW, sy * ROAD_HW], ...arc]);
    }
  }
  // curbs with corner fillets
  for (const sx of [1, -1]) {
    for (const sy of [1, -1]) {
      const pts = [[sx * ARM_END, sy * ROAD_HW], [sx * WALK_OUT, sy * ROAD_HW]];
      const cx = sx * WALK_OUT;
      const cy = sy * WALK_OUT;
      const a0 = Math.atan2(sy * ROAD_HW - cy, sx * WALK_OUT - cx);
      const a1 = Math.atan2(sy * WALK_OUT - cy, sx * ROAD_HW - cx);
      let da = a1 - a0;
      if (da > Math.PI) da -= 2 * Math.PI;
      if (da < -Math.PI) da += 2 * Math.PI;
      for (let i = 1; i < 12; i++) pts.push([cx + Math.cos(a0 + (da * i) / 12) * CORNER_R, cy + Math.sin(a0 + (da * i) / 12) * CORNER_R]);
      pts.push([sx * ROAD_HW, sy * WALK_OUT], [sx * ROAD_HW, sy * ARM_END]);
      L('curbstone', 'high', pts, WALK_Z);
    }
  }
  for (const b of aabbs.filter((b) => b[7] === 2)) {
    L('road_border', 'building', [[b[0], b[2]], [b[1], b[2]], [b[1], b[3]], [b[0], b[3]], [b[0], b[2]]], WALK_Z);
  }
  return { crossing: 'crossing1', lines, areas };
}

// ------------------------------------------------------------------------------------------
// write
const json = (p, o) => fs.writeFileSync(path.join(OUT, p), JSON.stringify(o));
json(`${SEQ}/detections.json`, detections);
json(`${SEQ}/gt.json`, gtData);
json(`${SEQ}/metrics.json`, metrics);
json('map_crossing1.json', mapJson());
json(`${SEQ}/meta.json`, {
  id: SEQ,
  crossing: 'crossing1',
  map: 'map_crossing1.json',
  fps: FPS,
  n_frames: N_FRAMES,
  timestamps: Array.from({ length: N_FRAMES }, (_, i) => r3(T0 + i * DT)),
  bounds: { xmin: -BOUND, xmax: BOUND, ymin: -BOUND, ymax: BOUND, zmin: -2, zmax: 30 },
  lidars: lidars.map(({ name, index, position }) => ({ name, index, position })),
  point_format: {
    stride_bytes: 8, scale: SCALE, origin: [0, 0, 0],
    fields: ['x:int16', 'y:int16', 'z:int16', 'intensity:uint8', 'flags:uint8'],
  },
  frames: frameMeta,
});
json('index.json', {
  generated_at: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
  sequences: [{
    id: SEQ, crossing: 'crossing1', n_frames: N_FRAMES, duration_s: N_FRAMES / FPS,
    unique_vehicles: detections.unique.vehicle, unique_vehicles_gt: gtData.unique.vehicle,
  }],
});

console.log(`wrote ${SEQ}: ${N_FRAMES} frames, ${(totalPts / N_FRAMES / 1000).toFixed(0)}k points/frame avg, ` +
  `${objects.length} objects (${detections.unique.vehicle} det / ${gtData.unique.vehicle} gt unique vehicles), ` +
  `AP ${metrics.detection.vehicle.ap['0.3']}, MAE ${metrics.counting.mae}, ${((Date.now() - t0) / 1000).toFixed(1)} s`);
