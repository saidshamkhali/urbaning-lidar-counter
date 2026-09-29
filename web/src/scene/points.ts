import {
  BufferGeometry, Color, InterleavedBuffer, InterleavedBufferAttribute, NormalBlending, Points,
  ShaderMaterial, Sphere, Vector2, Vector3,
} from 'three';
import { LIDAR_HEX } from './palette';

export type ColorMode = 'height' | 'intensity' | 'lidar' | 'focus';
export const COLOR_MODES: ColorMode[] = ['height', 'intensity', 'lidar', 'focus'];
const MODE_INDEX: Record<ColorMode, number> = { height: 0, intensity: 1, lidar: 2, focus: 3 };

const vertexShader = /* glsl */ `
  // position = raw int16 xyz, aIF = raw int16 holding intensity (low byte) and flags (high byte)
  attribute float aIF;

  uniform float uScale;
  uniform vec3 uOrigin;
  uniform float uSize;
  uniform float uSizeMul;
  uniform float uIsOrtho;
  uniform float uPxPerMetre;
  uniform float uPerspK;
  uniform float uMinPx;
  uniform float uMaxPx;
  uniform int uMode;
  uniform float uGroundZ;
  uniform vec3 uLidar[4];
  uniform vec3 uRamp[6];
  uniform vec3 uIRamp[5];
  uniform vec3 uSlate;
  uniform vec2 uScanCenter;
  uniform float uScanRadius;
  uniform vec3 uScanColor;

  varying vec3 vColor;
  varying float vAlpha;

  vec3 heightRamp(float h) {
    // h: metres above ground
    // tuned for pole-mounted LiDARs at an intersection: cars reach cyan, trucks/walls yellow-amber
    if (h < 0.4) return mix(uRamp[0], uRamp[1], smoothstep(0.15, 0.4, h));
    if (h < 1.3) return mix(uRamp[1], uRamp[2], (h - 0.4) / 0.9);
    if (h < 2.6) return mix(uRamp[2], uRamp[3], (h - 1.3) / 1.3);
    if (h < 4.2) return mix(uRamp[3], uRamp[4], (h - 2.6) / 1.6);
    return mix(uRamp[4], uRamp[5], clamp((h - 4.2) / 6.0, 0.0, 1.0));
  }

  vec3 intensityRamp(float t) {
    t = pow(clamp(t, 0.0, 1.0), 0.55) * 4.0;
    if (t < 1.0) return mix(uIRamp[0], uIRamp[1], t);
    if (t < 2.0) return mix(uIRamp[1], uIRamp[2], t - 1.0);
    if (t < 3.0) return mix(uIRamp[2], uIRamp[3], t - 2.0);
    return mix(uIRamp[3], uIRamp[4], t - 3.0);
  }

  void main() {
    vec3 p = position * uScale + uOrigin;
    float v = aIF < 0.0 ? aIF + 65536.0 : aIF;
    float flags = floor(v / 256.0 + 0.001);
    float inten = (v - flags * 256.0) / 255.0;
    float lidar = mod(flags, 4.0);
    float fg = mod(floor(flags / 4.0), 2.0);
    float ground = mod(floor(flags / 8.0), 2.0);
    float inBox = mod(floor(flags / 16.0), 2.0);
    float h = p.z - uGroundZ;
    float isGround = max(ground, 1.0 - step(0.18, h));

    vec3 col;
    float alpha = 1.0;
    if (uMode == 0) {
      col = heightRamp(h);
      // let road markings shimmer through the ground colour
      col = mix(col, uSlate * (0.75 + 1.9 * inten), isGround);
    } else if (uMode == 1) {
      col = intensityRamp(inten);
    } else if (uMode == 2) {
      int li = int(lidar + 0.5);
      vec3 lc = li == 0 ? uLidar[0] : li == 1 ? uLidar[1] : li == 2 ? uLidar[2] : uLidar[3];
      col = lc * mix(1.0, 0.45, isGround);
    } else {
      float focus = max(fg, inBox);
      col = focus > 0.5 ? heightRamp(max(h, 0.6)) * 1.12 : uSlate * 0.55 * (0.8 + inten);
      alpha = focus > 0.5 ? 1.0 : 0.55;
    }
    col *= 1.0 + 0.32 * inBox * (1.0 - isGround);

    // intro / sequence-switch scan wave
    float dist = length(p.xy - uScanCenter);
    float band = smoothstep(uScanRadius - 9.0, uScanRadius, dist) * step(dist, uScanRadius);
    col = mix(col, uScanColor, band * 0.85);
    float visible = step(dist, uScanRadius);

    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    float px = uIsOrtho > 0.5 ? uSize * uPxPerMetre : uSize * uPerspK / max(0.1, -mv.z);
    px = clamp(px * uSizeMul, uMinPx * uSizeMul, uMaxPx);
    px *= 1.0 + band * 0.8;
    gl_PointSize = px * visible;
    vColor = col;
    vAlpha = alpha * visible;
  }
`;

const fragmentShader = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float d = dot(c, c);
    if (d > 1.0 || vAlpha <= 0.0) discard;
    float a = 1.0 - smoothstep(0.35, 1.0, d);
    gl_FragColor = vec4(vColor, a * vAlpha);
  }
`;

const c = (hex: string) => new Color(hex);

/** One preallocated point buffer; raw int16/uint8 records are uploaded unchanged. */
export class PointLayer {
  readonly points: Points;
  readonly material: ShaderMaterial;
  private geometry: BufferGeometry;
  private buffer: InterleavedBuffer;
  private capacity = 0;
  count = 0;

  constructor(capacity: number) {
    this.material = new ShaderMaterial({
      vertexShader,
      fragmentShader,
      transparent: true,
      depthWrite: true,
      blending: NormalBlending,
      uniforms: {
        uScale: { value: 0.01 },
        uOrigin: { value: new Vector3() },
        uSize: { value: 0.09 },
        uSizeMul: { value: 1 },
        uIsOrtho: { value: 1 },
        uPxPerMetre: { value: 5 },
        uPerspK: { value: 500 },
        uMinPx: { value: 1.6 },
        uMaxPx: { value: 9 },
        uMode: { value: 0 },
        uGroundZ: { value: 0.45 },
        uLidar: { value: LIDAR_HEX.map(c) },
        uRamp: { value: ['#27354a', '#17a3a8', '#3fe0ff', '#f3e36b', '#ffa435', '#ff6a3d'].map(c) },
        uIRamp: { value: ['#0c1a3d', '#1f5fd0', '#2fd8ff', '#d9ffb8', '#ffd24a'].map(c) },
        uSlate: { value: c('#56698a') },
        uScanCenter: { value: new Vector2() },
        uScanRadius: { value: 1e6 },
        uScanColor: { value: c('#bff6ff').multiplyScalar(2.2) },
      },
    });
    this.buffer = new InterleavedBuffer(new Int16Array(4), 4);
    this.geometry = new BufferGeometry();
    this.points = new Points(this.geometry, this.material);
    this.points.frustumCulled = false;
    this.points.renderOrder = 1;
    this.ensureCapacity(capacity);
  }

  ensureCapacity(n: number): void {
    if (n <= this.capacity) return;
    this.capacity = Math.ceil(n * 1.05);
    const geo = new BufferGeometry();
    const buf = new InterleavedBuffer(new Int16Array(this.capacity * 4), 4);
    buf.setUsage(35048); // DynamicDrawUsage
    geo.setAttribute('position', new InterleavedBufferAttribute(buf, 3, 0));
    geo.setAttribute('aIF', new InterleavedBufferAttribute(buf, 1, 3));
    geo.boundingSphere = new Sphere(new Vector3(), 1e5);
    geo.setDrawRange(0, 0);
    this.geometry.dispose();
    this.geometry = geo;
    this.buffer = buf;
    this.points.geometry = geo;
    this.count = 0;
  }

  setFormat(scale: number, origin: [number, number, number], groundZ: number): void {
    this.material.uniforms.uScale.value = scale;
    (this.material.uniforms.uOrigin.value as Vector3).set(origin[0], origin[1], origin[2]);
    this.material.uniforms.uGroundZ.value = groundZ;
  }

  /** Copy one frame's raw records into the preallocated buffer (no decoding). */
  upload(data: ArrayBuffer): void {
    let n = Math.floor(data.byteLength / 8);
    if (n > this.capacity) this.ensureCapacity(n);
    n = Math.min(n, this.capacity);
    const src = new Int16Array(data, 0, n * 4);
    const arr = this.buffer.array as Int16Array;
    arr.set(src, 0);
    this.buffer.clearUpdateRanges();
    this.buffer.addUpdateRange(0, n * 4);
    this.buffer.needsUpdate = true;
    this.geometry.setDrawRange(0, n);
    this.count = n;
  }

  setMode(mode: ColorMode): void {
    this.material.uniforms.uMode.value = MODE_INDEX[mode] ?? 0;
  }

  setSizeMul(m: number): void {
    this.material.uniforms.uSizeMul.value = m;
  }

  setCameraParams(isOrtho: boolean, pxPerMetre: number, perspK: number, pixelRatio: number): void {
    const u = this.material.uniforms;
    u.uIsOrtho.value = isOrtho ? 1 : 0;
    u.uPxPerMetre.value = pxPerMetre;
    u.uPerspK.value = perspK;
    u.uMinPx.value = 1.35 * pixelRatio;
    u.uMaxPx.value = 7 * pixelRatio;
  }

  setScan(cx: number, cy: number, radius: number): void {
    (this.material.uniforms.uScanCenter.value as Vector2).set(cx, cy);
    this.material.uniforms.uScanRadius.value = radius;
  }
}
