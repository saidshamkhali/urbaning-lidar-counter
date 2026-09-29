import {
  BoxGeometry, Color, DoubleSide, Group, Mesh, MeshBasicMaterial, PlaneGeometry, type Ray,
} from 'three';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import type { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import type { Detection } from '../data/types';
import { SegmentBuffer, makeLineMaterial } from './lines';
import type { Stage } from './stage';

export interface BoxStyle {
  visible: boolean;
  color: Color;
  /** HDR multiplier on the edge colour (values > ~1 bloom) */
  intensity: number;
  linewidth: number;
  fillOpacity: number;
  bodyOpacity: number;
  label: string | null;
  labelColor: string;
}

const unitPlane = new PlaneGeometry(1, 1);
const unitBox = new BoxGeometry(1, 1, 1);

class BoxVisual {
  readonly group = new Group();
  readonly seg: SegmentBuffer;
  readonly mat: LineMaterial;
  readonly fill: Mesh;
  readonly body: Mesh;
  readonly fillMat: MeshBasicMaterial;
  readonly bodyMat: MeshBasicMaterial;
  readonly label: CSS2DObject;
  readonly labelEl: HTMLDivElement;
  private labelText = '';
  private labelColor = '';
  id = -1;
  box: number[] = [0, 0, 0, 1, 1, 1, 0];
  active = false;

  constructor(stage: Stage, dashed: boolean) {
    this.mat = stage.registerLineMaterial(
      makeLineMaterial({ vertexColors: true, dashed, dashSize: 0.45, gapSize: 0.32, linewidth: 1.6 }),
    );
    this.seg = new SegmentBuffer(16, this.mat);
    this.seg.object.renderOrder = 5;
    this.fillMat = new MeshBasicMaterial({ transparent: true, depthWrite: false, opacity: 0.15, side: DoubleSide });
    this.bodyMat = new MeshBasicMaterial({ transparent: true, depthWrite: false, opacity: 0.05, side: DoubleSide });
    this.fill = new Mesh(unitPlane, this.fillMat);
    this.body = new Mesh(unitBox, this.bodyMat);
    this.fill.renderOrder = 3;
    this.body.renderOrder = 4;
    this.labelEl = document.createElement('div');
    this.labelEl.className = 'chip';
    this.label = new CSS2DObject(this.labelEl);
    this.label.center.set(0.5, 1.25);
    this.group.add(this.seg.object, this.fill, this.body, this.label);
    this.group.visible = false;
  }

  set(d: Detection, s: BoxStyle): void {
    const [x, y, z, l, w, h, yaw] = d.box;
    this.id = d.id;
    this.box = d.box;
    this.group.position.set(x, y, z);
    this.group.rotation.set(0, 0, yaw);

    const hl = l / 2;
    const hw = w / 2;
    const hh = h / 2;
    const seg = this.seg;
    seg.reset();
    const rear = 0.38;
    const front = 1.0;
    const mid = 0.7;
    // bottom + top rectangles; longitudinal edges fade from rear to front
    for (const zz of [-hh, hh]) {
      seg.push(-hl, -hw, zz, hl, -hw, zz, rear, rear, rear, front, front, front);
      seg.push(-hl, hw, zz, hl, hw, zz, rear, rear, rear, front, front, front);
      seg.push(hl, -hw, zz, hl, hw, zz, front, front, front);
      seg.push(-hl, -hw, zz, -hl, hw, zz, rear, rear, rear);
    }
    // vertical edges
    seg.push(hl, -hw, -hh, hl, -hw, hh, front, front, front);
    seg.push(hl, hw, -hh, hl, hw, hh, front, front, front);
    seg.push(-hl, -hw, -hh, -hl, -hw, hh, rear, rear, rear);
    seg.push(-hl, hw, -hh, -hl, hw, hh, rear, rear, rear);
    // heading chevron on the roof
    const sz = Math.min(w * 0.36, l * 0.3, 1.1);
    const tip = hl - Math.min(0.5, l * 0.1);
    seg.push(tip - sz, -sz, hh, tip, 0, hh, mid, mid, mid, front, front, front);
    seg.push(tip - sz, sz, hh, tip, 0, hh, mid, mid, mid, front, front, front);
    // heading tick on the ground in front of the box
    const tl = Math.min(1.2, Math.max(0.4, l * 0.18));
    seg.push(hl, 0, -hh, hl + tl, 0, -hh, front, front, front, 0.2, 0.2, 0.2);
    seg.commit();

    this.mat.color.copy(s.color).multiplyScalar(s.intensity);
    this.mat.linewidth = s.linewidth;
    this.fillMat.color.copy(s.color);
    this.fillMat.opacity = s.fillOpacity;
    this.fill.visible = s.fillOpacity > 0.001;
    this.fill.scale.set(l, w, 1);
    this.fill.position.set(0, 0, -hh + 0.03);
    this.bodyMat.color.copy(s.color);
    this.bodyMat.opacity = s.bodyOpacity;
    this.body.visible = s.bodyOpacity > 0.001;
    this.body.scale.set(l, w, h);

    if (s.label) {
      if (s.label !== this.labelText) {
        this.labelEl.innerHTML = s.label;
        this.labelText = s.label;
      }
      if (s.labelColor !== this.labelColor) {
        this.labelEl.style.setProperty('--c', s.labelColor);
        this.labelColor = s.labelColor;
      }
      this.label.position.set(0, 0, hh);
      this.label.visible = true;
    } else {
      this.label.visible = false;
    }
    this.group.visible = true;
    this.active = true;
  }

  hide(): void {
    this.group.visible = false;
    this.label.visible = false;
    this.active = false;
  }
}

/** Pool of box visuals, reused frame to frame. */
export class BoxLayer {
  readonly root = new Group();
  private readonly pool: BoxVisual[] = [];
  private used = 0;
  private readonly style: BoxStyle = {
    visible: true, color: new Color(), intensity: 1, linewidth: 1.6, fillOpacity: 0.12, bodyOpacity: 0.04,
    label: null, labelColor: '#fff',
  };

  constructor(private readonly stage: Stage, private readonly dashed: boolean) {}

  update(dets: readonly Detection[], styleFn: (d: Detection, index: number, out: BoxStyle) => void): void {
    this.used = 0;
    for (let i = 0; i < dets.length; i++) {
      const s = this.style;
      s.visible = true;
      s.label = null;
      styleFn(dets[i], i, s);
      if (!s.visible) continue;
      this.acquire().set(dets[i], s);
    }
    for (let i = this.used; i < this.pool.length; i++) if (this.pool[i].active) this.pool[i].hide();
  }

  clear(): void {
    this.used = 0;
    for (const v of this.pool) if (v.active) v.hide();
  }

  /** Closest box hit by the ray, in world space. */
  pick(ray: Ray): { id: number; dist: number } | null {
    let best: { id: number; dist: number } | null = null;
    for (let i = 0; i < this.used; i++) {
      const v = this.pool[i];
      const [x, y, z, l, w, h, yaw] = v.box;
      const c = Math.cos(-yaw);
      const s = Math.sin(-yaw);
      const ox = ray.origin.x - x;
      const oy = ray.origin.y - y;
      const o = [ox * c - oy * s, ox * s + oy * c, ray.origin.z - z];
      const d = [ray.direction.x * c - ray.direction.y * s, ray.direction.x * s + ray.direction.y * c, ray.direction.z];
      // generous picking margin for small objects
      const m = 0.25;
      const ext = [l / 2 + m, w / 2 + m, h / 2 + m];
      let t0 = -Infinity;
      let t1 = Infinity;
      let miss = false;
      for (let k = 0; k < 3; k++) {
        if (Math.abs(d[k]) < 1e-9) {
          if (Math.abs(o[k]) > ext[k]) { miss = true; break; }
          continue;
        }
        let ta = (-ext[k] - o[k]) / d[k];
        let tb = (ext[k] - o[k]) / d[k];
        if (ta > tb) { const tmp = ta; ta = tb; tb = tmp; }
        t0 = Math.max(t0, ta);
        t1 = Math.min(t1, tb);
        if (t0 > t1) { miss = true; break; }
      }
      if (miss || t1 < 0) continue;
      const dist = Math.max(0, t0);
      if (!best || dist < best.dist) best = { id: v.id, dist };
    }
    return best;
  }

  private acquire(): BoxVisual {
    let v = this.pool[this.used];
    if (!v) {
      v = new BoxVisual(this.stage, this.dashed);
      this.pool.push(v);
      this.root.add(v.group);
    }
    this.used++;
    return v;
  }
}
