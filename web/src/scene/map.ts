import {
  BufferAttribute, BufferGeometry, Color, Group, Mesh, MeshBasicMaterial, ShapeUtils, Vector2,
} from 'three';
import type { MapFile, MapLine } from '../data/types';
import { SegmentBuffer, makeLineMaterial } from './lines';
import type { Stage } from './stage';

interface LineStyle {
  color: string;
  width: number;
  opacity: number;
  dashed?: boolean;
  dash?: number;
  gap?: number;
  lift?: number;
}

const LINE_STYLES: Record<string, LineStyle> = {
  curb: { color: '#cdd6e4', width: 1.5, opacity: 0.8, lift: 0.08 },
  curbLow: { color: '#8b97ab', width: 1.0, opacity: 0.4, lift: 0.06 },
  border: { color: '#b8c3d4', width: 1.3, opacity: 0.6, lift: 0.08 },
  building: { color: '#6f7d93', width: 1.0, opacity: 0.35, lift: 0.06 },
  lane: { color: '#ffffff', width: 1.1, opacity: 0.3, lift: 0.06 },
  laneDashed: { color: '#ffffff', width: 1.1, opacity: 0.3, dashed: true, dash: 3, gap: 6, lift: 0.06 },
  stop: { color: '#ffffff', width: 2.2, opacity: 0.55, lift: 0.07 },
  zebra: { color: '#ffffff', width: 1.2, opacity: 0.26, lift: 0.06 },
  virtual: { color: '#6f86a6', width: 1.0, opacity: 0.16, dashed: true, dash: 0.6, gap: 1.0, lift: 0.05 },
  other: { color: '#9aa7ba', width: 1.0, opacity: 0.25, lift: 0.06 },
};

const AREA_STYLES: Record<string, { color: string; opacity: number }> = {
  road: { color: '#6f8fb8', opacity: 0.045 },
  parking: { color: '#4a8fe0', opacity: 0.05 },
  crosswalk: { color: '#e6f0ff', opacity: 0.05 },
  walkway: { color: '#8e7dff', opacity: 0.035 },
  bicycle_lane: { color: '#ff5c9a', opacity: 0.045 },
  other: { color: '#7d8aa0', opacity: 0.03 },
};

function lineStyleKey(l: MapLine): string {
  const k = l.kind;
  const st = l.subtype ?? '';
  if (k === 'curbstone') return st === 'low' ? 'curbLow' : 'curb';
  if (k === 'road_border') return st === 'building' ? 'building' : 'border';
  if (k === 'line_thin' || k === 'line_thick') return st.includes('dashed') ? 'laneDashed' : 'lane';
  if (k === 'stop_line') return 'stop';
  if (k === 'zebra_marking') return 'zebra';
  if (k === 'virtual') return 'virtual';
  if (k === 'fence' || k === 'guard_rail' || k === 'wall') return 'border';
  return 'other';
}

function areaStyleKey(kind: string): string {
  if (kind === 'road' || kind === 'highway' || kind === 'play_street') return 'road';
  if (kind in AREA_STYLES) return kind;
  if (kind.includes('walk') || kind === 'stairs') return 'walkway';
  if (kind.includes('bicycle')) return 'bicycle_lane';
  if (kind.includes('parking')) return 'parking';
  if (kind.includes('cross')) return 'crosswalk';
  return 'other';
}

/** Estimate the ground height from the map's line vertices (median z). */
export function estimateGroundZ(map: MapFile | null, fallback = 0.45): number {
  if (!map) return fallback;
  const zs: number[] = [];
  for (const l of map.lines) for (const p of l.points) if (p.length > 2 && Number.isFinite(p[2])) zs.push(p[2]);
  if (zs.length < 5) return fallback;
  zs.sort((a, b) => a - b);
  const med = zs[Math.floor(zs.length / 2)];
  // a flat 2D map (all z = 0) says nothing about the ground height
  if (Math.abs(med) < 1e-6 && Math.abs(zs[zs.length - 1] - zs[0]) < 1e-6) return fallback;
  return med;
}

export class MapLayer {
  readonly root = new Group();

  constructor(private readonly stage: Stage) {
    this.root.renderOrder = 0;
  }

  build(map: MapFile | null, groundZ: number): void {
    for (const child of [...this.root.children]) {
      this.root.remove(child);
      if (child instanceof Mesh) child.geometry.dispose();
    }
    if (!map) return;
    // ignore the map's own z if it does not match the ground (e.g. a 2D map)
    const zs: number[] = [];
    for (const l of map.lines ?? []) for (const p of l.points ?? []) if (p.length > 2 && Number.isFinite(p[2])) zs.push(p[2]);
    zs.sort((a, b) => a - b);
    const useZ = zs.length > 0 && Math.abs(zs[Math.floor(zs.length / 2)] - groundZ) < 0.3;
    const zOf = (p: number[]) => (useZ && p.length > 2 && Number.isFinite(p[2]) ? p[2] : groundZ);

    // lines, merged per style
    const groups = new Map<string, MapLine[]>();
    for (const l of map.lines ?? []) {
      if (!l.points || l.points.length < 2) continue;
      const key = lineStyleKey(l);
      let arr = groups.get(key);
      if (!arr) groups.set(key, (arr = []));
      arr.push(l);
    }
    for (const [key, lines] of groups) {
      const st = LINE_STYLES[key];
      let nSeg = 0;
      for (const l of lines) nSeg += l.points.length - 1;
      const mat = this.stage.registerLineMaterial(makeLineMaterial({
        color: st.color, linewidth: st.width, opacity: st.opacity, dashed: st.dashed,
        dashSize: st.dash, gapSize: st.gap,
      }));
      const seg = new SegmentBuffer(nSeg, mat);
      const lift = st.lift ?? 0.06;
      for (const l of lines) {
        let d = 0;
        for (let i = 0; i < l.points.length - 1; i++) {
          const a = l.points[i];
          const b = l.points[i + 1];
          const az = zOf(a) + lift;
          const bz = zOf(b) + lift;
          seg.push(a[0], a[1], az, b[0], b[1], bz, 1, 1, 1, 1, 1, 1, d);
          d += Math.hypot(b[0] - a[0], b[1] - a[1]);
        }
      }
      seg.commit();
      seg.object.renderOrder = 2;
      this.root.add(seg.object);
    }

    // areas, triangulated and merged per kind
    const areaGroups = new Map<string, number[][][]>();
    for (const a of map.areas ?? []) {
      if (!a.polygon || a.polygon.length < 3) continue;
      const key = areaStyleKey(a.kind);
      let arr = areaGroups.get(key);
      if (!arr) areaGroups.set(key, (arr = []));
      arr.push(a.polygon);
    }
    const order = ['road', 'parking', 'walkway', 'bicycle_lane', 'crosswalk', 'other'];
    for (const [key, polys] of areaGroups) {
      const st = AREA_STYLES[key];
      const pos: number[] = [];
      const z = groundZ - 0.04 + order.indexOf(key) * 0.004;
      for (const poly of polys) {
        const contour = poly.map((p) => new Vector2(p[0], p[1]));
        // drop a duplicated closing vertex
        if (contour.length > 3 && contour[0].distanceTo(contour[contour.length - 1]) < 1e-6) contour.pop();
        let tris: number[][];
        try {
          tris = ShapeUtils.triangulateShape(contour, []);
        } catch {
          continue;
        }
        for (const t of tris) for (const i of t) pos.push(contour[i].x, contour[i].y, z);
      }
      const geo = new BufferGeometry();
      geo.setAttribute('position', new BufferAttribute(new Float32Array(pos), 3));
      const mesh = new Mesh(geo, new MeshBasicMaterial({
        color: new Color(st.color), transparent: true, opacity: st.opacity, depthWrite: false, side: 2,
      }));
      mesh.renderOrder = 0;
      this.root.add(mesh);
    }
  }
}
