import { Group, type Color } from 'three';
import type { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import type { Labels } from '../data/loader';
import type { Detection } from '../data/types';
import { SegmentBuffer, makeLineMaterial } from './lines';
import type { Stage } from './stage';

const MAX_SEGS = 48;

/** Fading polylines of each active track's recent centre positions. */
export class TrailLayer {
  readonly root = new Group();
  private readonly pool: SegmentBuffer[] = [];
  private readonly material: LineMaterial;
  private readonly selMaterial: LineMaterial;
  private used = 0;
  /** trail length in frames */
  window = 30;

  constructor(stage: Stage) {
    this.material = stage.registerLineMaterial(makeLineMaterial({ vertexColors: true, linewidth: 2.2, additive: true }));
    this.selMaterial = stage.registerLineMaterial(makeLineMaterial({ vertexColors: true, linewidth: 3.4, additive: true }));
  }

  update(
    labels: Labels,
    frame: number,
    dets: readonly Detection[],
    colorFn: (d: Detection) => Color,
    selectedId: number | null,
    zOffset: number,
  ): void {
    this.used = 0;
    const f0 = Math.max(0, frame - this.window);
    for (const d of dets) {
      const p = labels.paths.get(d.id);
      if (!p) continue;
      // skip stationary tracks
      let first = -1;
      for (let f = f0; f <= frame; f++) if (!Number.isNaN(p.x[f])) { first = f; break; }
      if (first < 0 || first === frame) continue;
      const disp = Math.hypot(p.x[frame] - p.x[first], p.y[frame] - p.y[first]);
      if (disp < 1.2) continue;

      const seg = this.acquire();
      seg.object.material = d.id === selectedId ? this.selMaterial : this.material;
      seg.reset();
      const col = colorFn(d);
      const boost = d.id === selectedId ? 1.6 : 1.0;
      const span = frame - first;
      let px = p.x[first];
      let py = p.y[first];
      let pz = p.z[first] + zOffset;
      let pa = 0;
      for (let f = first + 1; f <= frame; f++) {
        const x = p.x[f];
        if (Number.isNaN(x)) continue;
        const y = p.y[f];
        const z = p.z[f] + zOffset;
        const a = Math.pow((f - first) / span, 1.7) * boost;
        seg.push(px, py, pz, x, y, z, col.r * pa, col.g * pa, col.b * pa, col.r * a, col.g * a, col.b * a);
        px = x; py = y; pz = z; pa = a;
        if (seg.count >= MAX_SEGS) break;
      }
      seg.commit();
    }
    for (let i = this.used; i < this.pool.length; i++) this.pool[i].object.visible = false;
  }

  clear(): void {
    this.used = 0;
    for (const s of this.pool) s.object.visible = false;
  }

  private acquire(): SegmentBuffer {
    let s = this.pool[this.used];
    if (!s) {
      s = new SegmentBuffer(MAX_SEGS, this.material);
      s.object.renderOrder = 2;
      this.pool.push(s);
      this.root.add(s.object);
    }
    this.used++;
    return s;
  }
}
