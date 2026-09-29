import { InstancedInterleavedBuffer, InterleavedBufferAttribute } from 'three';
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';

/**
 * A fixed-capacity fat-line segment buffer that can be rewritten in place every frame
 * without allocating (unlike LineSegmentsGeometry.setPositions).
 */
export class SegmentBuffer {
  readonly object: LineSegments2;
  readonly geometry: LineSegmentsGeometry;
  readonly pos: Float32Array;
  readonly col: Float32Array;
  readonly dist: Float32Array;
  private readonly posBuf: InstancedInterleavedBuffer;
  private readonly colBuf: InstancedInterleavedBuffer;
  private readonly distBuf: InstancedInterleavedBuffer;
  count = 0;

  constructor(readonly capacity: number, material: LineMaterial) {
    const geo = new LineSegmentsGeometry();
    this.pos = new Float32Array(capacity * 6);
    this.col = new Float32Array(capacity * 6).fill(1);
    this.dist = new Float32Array(capacity * 2);
    this.posBuf = new InstancedInterleavedBuffer(this.pos, 6, 1);
    this.colBuf = new InstancedInterleavedBuffer(this.col, 6, 1);
    this.distBuf = new InstancedInterleavedBuffer(this.dist, 2, 1);
    geo.setAttribute('instanceStart', new InterleavedBufferAttribute(this.posBuf, 3, 0));
    geo.setAttribute('instanceEnd', new InterleavedBufferAttribute(this.posBuf, 3, 3));
    geo.setAttribute('instanceColorStart', new InterleavedBufferAttribute(this.colBuf, 3, 0));
    geo.setAttribute('instanceColorEnd', new InterleavedBufferAttribute(this.colBuf, 3, 3));
    geo.setAttribute('instanceDistanceStart', new InterleavedBufferAttribute(this.distBuf, 1, 0));
    geo.setAttribute('instanceDistanceEnd', new InterleavedBufferAttribute(this.distBuf, 1, 1));
    geo.instanceCount = 0;
    this.geometry = geo;
    this.object = new LineSegments2(geo, material);
    this.object.frustumCulled = false;
  }

  reset(): void {
    this.count = 0;
  }

  /** Append a segment; returns false when full. `d0` is the dash distance at the start. */
  push(
    ax: number, ay: number, az: number, bx: number, by: number, bz: number,
    r = 1, g = 1, b = 1, r2 = r, g2 = g, b2 = b, d0 = 0,
  ): boolean {
    if (this.count >= this.capacity) return false;
    const i = this.count * 6;
    const p = this.pos;
    p[i] = ax; p[i + 1] = ay; p[i + 2] = az; p[i + 3] = bx; p[i + 4] = by; p[i + 5] = bz;
    const c = this.col;
    c[i] = r; c[i + 1] = g; c[i + 2] = b; c[i + 3] = r2; c[i + 4] = g2; c[i + 5] = b2;
    const j = this.count * 2;
    this.dist[j] = d0;
    this.dist[j + 1] = d0 + Math.hypot(bx - ax, by - ay, bz - az);
    this.count++;
    return true;
  }

  commit(): void {
    this.geometry.instanceCount = this.count;
    this.posBuf.needsUpdate = true;
    this.colBuf.needsUpdate = true;
    this.distBuf.needsUpdate = true;
    this.object.visible = this.count > 0;
  }
}

export function makeLineMaterial(opts: {
  color?: number | string;
  linewidth?: number;
  opacity?: number;
  dashed?: boolean;
  dashSize?: number;
  gapSize?: number;
  vertexColors?: boolean;
  additive?: boolean;
  depthTest?: boolean;
}): LineMaterial {
  const m = new LineMaterial({
    color: opts.color ?? 0xffffff,
    linewidth: opts.linewidth ?? 1.5,
    vertexColors: opts.vertexColors ?? false,
    dashed: opts.dashed ?? false,
    dashSize: opts.dashSize ?? 0.5,
    gapSize: opts.gapSize ?? 0.3,
    transparent: true,
    opacity: opts.opacity ?? 1,
    depthWrite: false,
    depthTest: opts.depthTest ?? true,
    worldUnits: false,
  });
  if (opts.additive) m.blending = 2; // AdditiveBlending
  return m;
}
