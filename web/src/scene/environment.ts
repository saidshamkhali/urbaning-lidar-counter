import {
  Color, DoubleSide, Group, Mesh, MeshBasicMaterial, PlaneGeometry, RingGeometry, ShaderMaterial,
  SphereGeometry, Vector2, Vector3, type Camera,
} from 'three';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import type { LidarInfo } from '../data/types';
import { SegmentBuffer, makeLineMaterial } from './lines';
import { HEX, LIDAR_HEX } from './palette';
import type { Stage } from './stage';

/** Faint 10 m ground grid with 50 m major lines, fading out radially. */
export class Grid {
  readonly mesh: Mesh;
  private readonly material: ShaderMaterial;

  constructor() {
    this.material = new ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: {
        uCenter: { value: new Vector2() },
        uFade: { value: 150 },
        uMinor: { value: new Color('#1d2c3f') },
        uMajor: { value: new Color('#2f4763') },
        uOpacity: { value: 1 },
      },
      vertexShader: /* glsl */ `
        varying vec2 vW;
        void main() {
          vec4 w = modelMatrix * vec4(position, 1.0);
          vW = w.xy;
          gl_Position = projectionMatrix * viewMatrix * w;
        }`,
      fragmentShader: /* glsl */ `
        uniform vec2 uCenter;
        uniform float uFade;
        uniform vec3 uMinor;
        uniform vec3 uMajor;
        uniform float uOpacity;
        varying vec2 vW;
        float gridLine(vec2 p, float spacing) {
          vec2 c = p / spacing;
          vec2 d = abs(fract(c - 0.5) - 0.5) / fwidth(c);
          return 1.0 - min(min(d.x, d.y), 1.0);
        }
        void main() {
          float minor = gridLine(vW, 10.0);
          float major = gridLine(vW, 50.0);
          float r = length(vW - uCenter);
          float fade = 1.0 - smoothstep(uFade * 0.3, uFade, r);
          vec3 col = mix(uMinor, uMajor, major);
          float a = max(minor * 0.55, major * 0.8) * fade * uOpacity;
          if (a < 0.003) discard;
          gl_FragColor = vec4(col, a);
        }`,
    });
    this.mesh = new Mesh(new PlaneGeometry(800, 800), this.material);
    this.mesh.renderOrder = -1;
    this.mesh.frustumCulled = false;
  }

  place(cx: number, cy: number, z: number, fade: number): void {
    this.mesh.position.set(cx, cy, z);
    (this.material.uniforms.uCenter.value as Vector2).set(cx, cy);
    this.material.uniforms.uFade.value = fade;
  }
}

interface SensorVisual {
  info: LidarInfo;
  group: Group;
  pulse: Mesh;
  pulseMat: MeshBasicMaterial;
  label: CSS2DObject;
  phase: number;
}

const ringGeo = new RingGeometry(0.95, 1.12, 64);
const pulseGeo = new RingGeometry(0.97, 1.0, 64);
const headGeo = new SphereGeometry(0.2, 16, 12);

/** LiDAR poles: ground ring, a slow pulse and a glowing sensor head. */
export class SensorLayer {
  readonly root = new Group();
  private sensors: SensorVisual[] = [];
  private readonly mastMat;
  private mast: SegmentBuffer | null = null;
  private hovered = -1;

  constructor(stage: Stage) {
    this.mastMat = stage.registerLineMaterial(makeLineMaterial({ vertexColors: true, linewidth: 1.2, additive: true }));
  }

  build(lidars: LidarInfo[], groundZ: number): void {
    for (const c of [...this.root.children]) this.root.remove(c);
    this.sensors = [];
    this.mast = new SegmentBuffer(Math.max(1, lidars.length), this.mastMat);
    const accent = new Color(HEX.accent);
    lidars.forEach((info, i) => {
      const col = new Color(LIDAR_HEX[info.index % 4] ?? HEX.accent).lerp(accent, 0.5);
      const g = new Group();
      const [x, y, z] = info.position;
      g.position.set(x, y, groundZ + 0.06);
      const ring = new Mesh(ringGeo, new MeshBasicMaterial({ color: col.clone().multiplyScalar(1.3), transparent: true, opacity: 0.8, depthWrite: false, side: DoubleSide }));
      const pulseMat = new MeshBasicMaterial({ color: col, transparent: true, opacity: 0.5, depthWrite: false, side: DoubleSide });
      const pulse = new Mesh(pulseGeo, pulseMat);
      const head = new Mesh(headGeo, new MeshBasicMaterial({ color: col.clone().multiplyScalar(2.4) }));
      head.position.set(0, 0, z - groundZ - 0.06);
      const el = document.createElement('div');
      el.className = 'sensor-chip';
      el.innerHTML = `<b>${escapeHtml(info.name)}</b><span>LiDAR ${info.index} · h ${(z - groundZ).toFixed(1)} m</span>`;
      const label = new CSS2DObject(el);
      label.position.set(0, 0, z - groundZ + 0.6);
      label.center.set(0.5, 1.1);
      label.visible = false;
      g.add(ring, pulse, head, label);
      ring.renderOrder = 6;
      pulse.renderOrder = 6;
      this.root.add(g);
      this.mast!.push(x, y, groundZ + 0.06, x, y, z, col.r * 0.1, col.g * 0.1, col.b * 0.1, col.r * 0.9, col.g * 0.9, col.b * 0.9);
      this.sensors.push({ info, group: g, pulse, pulseMat, label, phase: i * 0.61 });
    });
    this.mast.commit();
    this.root.add(this.mast.object);
  }

  update(time: number): void {
    for (const s of this.sensors) {
      const t = (time / 2.6 + s.phase) % 1;
      const k = 1 + t * 5.5;
      s.pulse.scale.set(k, k, 1);
      s.pulseMat.opacity = 0.55 * (1 - t) * (1 - t);
    }
  }

  /** Show the name of the sensor closest to the pointer (screen px), if within reach. */
  hover(camera: Camera, px: number, py: number, width: number, height: number): boolean {
    let best = -1;
    let bestD = 18;
    const v = new Vector3();
    this.sensors.forEach((s, i) => {
      for (const zz of [0, s.info.position[2] - s.group.position.z]) {
        v.set(s.group.position.x, s.group.position.y, s.group.position.z + zz).project(camera);
        if (v.z > 1) continue;
        const sx = ((v.x + 1) / 2) * width;
        const sy = ((1 - v.y) / 2) * height;
        const d = Math.hypot(sx - px, sy - py);
        if (d < bestD) { bestD = d; best = i; }
      }
    });
    if (best !== this.hovered) {
      this.sensors.forEach((s, i) => (s.label.visible = i === best));
      this.hovered = best;
    }
    return best >= 0;
  }
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}
