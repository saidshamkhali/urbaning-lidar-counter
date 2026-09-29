import { MOUSE, TOUCH, Vector3 } from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { Stage } from './stage';

export type ViewName = 'bev' | 'orbit' | 'follow';

interface Sph {
  az: number;
  pol: number;
  dist: number;
  target: Vector3;
}

interface Flight {
  from: Sph;
  to: Sph;
  t: number;
  dur: number;
  ease: (t: number) => number;
  done: () => void;
}

const easeInOutCubic = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const easeOutQuint = (t: number) => 1 - Math.pow(1 - t, 5);
const TOP_POL = 0.012;

function sphOf(pos: Vector3, target: Vector3): Sph {
  const o = pos.clone().sub(target);
  const dist = o.length();
  return {
    az: Math.atan2(o.y, o.x),
    pol: Math.acos(Math.min(1, Math.max(-1, o.z / Math.max(dist, 1e-9)))),
    dist,
    target: target.clone(),
  };
}

function posOf(s: Sph, out: Vector3): Vector3 {
  return out.set(
    s.target.x + s.dist * Math.sin(s.pol) * Math.cos(s.az),
    s.target.y + s.dist * Math.sin(s.pol) * Math.sin(s.az),
    s.target.z + s.dist * Math.cos(s.pol),
  );
}

/** Owns the BEV (orthographic) and 3D (perspective) controls, view switching and flights. */
export class CameraRig {
  readonly bevControls: OrbitControls;
  readonly orbitControls: OrbitControls;
  view: ViewName = 'bev';
  focus = new Vector3();
  /** visible ground height (m) of the default BEV framing */
  bevSpan = 130;
  private flight: Flight | null = null;
  private lastInteraction = -1e9;
  private clock = 0;
  private followTarget: Vector3 | null = null;
  private followHeading = 0;
  private readonly tmp = new Vector3();

  constructor(private readonly stage: Stage) {
    const el = stage.renderer.domElement;
    const bev = new OrbitControls(stage.ortho, el);
    bev.enableDamping = true;
    bev.dampingFactor = 0.12;
    bev.minPolarAngle = 0;
    bev.maxPolarAngle = 0;
    bev.minZoom = 0.25;
    bev.maxZoom = 14;
    bev.zoomToCursor = true;
    bev.mouseButtons = { LEFT: MOUSE.PAN, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.ROTATE };
    bev.touches = { ONE: TOUCH.PAN, TWO: TOUCH.DOLLY_ROTATE };
    this.bevControls = bev;

    const orbit = new OrbitControls(stage.persp, el);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.09;
    orbit.maxPolarAngle = (86 * Math.PI) / 180;
    orbit.minDistance = 4;
    orbit.maxDistance = 600;
    orbit.zoomToCursor = false;
    this.orbitControls = orbit;

    for (const c of [bev, orbit]) {
      c.addEventListener('start', () => (this.lastInteraction = Infinity));
      c.addEventListener('end', () => (this.lastInteraction = this.clock));
    }
    this.applyEnabled();
  }

  get flying(): boolean {
    return this.flight !== null;
  }

  /** Set the scene focus and reset cameras to their default framing. */
  setFocus(focus: Vector3, span: number): void {
    this.focus.copy(focus);
    this.bevSpan = span;
  }

  private bevZoomFor(span: number): number {
    return this.stage.orthoHeight / span;
  }

  private perspDistForSpan(span: number): number {
    return span / (2 * Math.tan((this.stage.persp.fov * Math.PI) / 360));
  }

  private orbitDefault(target: Vector3, dist = 105): Sph {
    return { az: (-122 * Math.PI) / 180, pol: (58 * Math.PI) / 180, dist, target: target.clone() };
  }

  /** Put the BEV camera at a target/span immediately. */
  private placeBev(target: Vector3, span: number, az = -Math.PI / 2): void {
    const cam = this.stage.ortho;
    cam.zoom = this.bevZoomFor(span);
    cam.updateProjectionMatrix();
    this.bevControls.target.copy(target);
    posOf({ az, pol: TOP_POL, dist: 400, target }, cam.position);
    cam.lookAt(target);
    this.bevControls.update();
  }

  private currentPerspSph(): Sph {
    return sphOf(this.stage.persp.position, this.orbitControls.target);
  }

  /** Equivalent perspective pose of the current BEV view. */
  private bevAsPersp(): Sph {
    const cam = this.stage.ortho;
    const span = this.stage.orthoHeight / cam.zoom;
    const s = sphOf(cam.position, this.bevControls.target);
    return { az: s.az, pol: TOP_POL, dist: this.perspDistForSpan(span), target: this.bevControls.target.clone() };
  }

  private fly(from: Sph, to: Sph, dur: number, done: () => void, ease = easeInOutCubic): void {
    // shortest azimuth path
    let da = to.az - from.az;
    while (da > Math.PI) da -= 2 * Math.PI;
    while (da < -Math.PI) da += 2 * Math.PI;
    to = { ...to, az: from.az + da };
    this.flight = { from, to, t: 0, dur, ease, done };
    this.stage.setCamera(this.stage.persp);
    this.applyPersp(from);
    this.applyEnabled();
  }

  private applyPersp(s: Sph): void {
    const cam = this.stage.persp;
    posOf(s, cam.position);
    this.orbitControls.target.copy(s.target);
    cam.lookAt(s.target);
  }

  /** Fly-in on load: high sweeping overview down to the requested view. */
  intro(view: ViewName, dur = 3.6, followPos?: Vector3, followHeading = 0): void {
    const start: Sph = { az: (-160 * Math.PI) / 180, pol: (64 * Math.PI) / 180, dist: 420, target: this.focus.clone() };
    this.view = view;
    this.placeBev(this.focus, this.bevSpan);
    if (view === 'bev') {
      const end: Sph = { az: -Math.PI / 2, pol: TOP_POL, dist: this.perspDistForSpan(this.bevSpan), target: this.focus.clone() };
      this.fly(start, end, dur, () => this.finishToBev(end));
    } else if (view === 'follow' && followPos) {
      const end = this.chasePose(followPos, followHeading);
      this.fly(start, end, dur, () => this.finishPersp());
    } else {
      this.fly(start, this.orbitDefault(this.focus), dur, () => this.finishPersp());
    }
  }

  private finishToBev(end: Sph): void {
    const span = (2 * end.dist * Math.tan((this.stage.persp.fov * Math.PI) / 360));
    this.placeBev(end.target, span, end.az);
    this.stage.setCamera(this.stage.ortho);
    this.flight = null;
    this.applyEnabled();
  }

  private finishPersp(): void {
    this.flight = null;
    this.orbitControls.update();
    this.applyEnabled();
  }

  /** length (m) of the followed object, used to scale the chase distance */
  followLength = 4.5;

  private chasePose(pos: Vector3, heading: number): Sph {
    const dist = 17 + 2.2 * Math.min(18, this.followLength);
    return { az: heading + Math.PI - 0.35, pol: (62 * Math.PI) / 180, dist, target: pos.clone() };
  }

  setView(view: ViewName, opts: { instant?: boolean; followPos?: Vector3; followHeading?: number } = {}): void {
    const prev = this.view;
    const dur = opts.instant ? 0 : 1.1;
    if (this.flight) {
      // finish an ongoing flight immediately
      const f = this.flight;
      this.flight = null;
      this.applyPersp(f.to);
    }
    this.view = view;
    if (view === 'bev') {
      if (prev === 'bev' && !opts.instant) {
        // re-frame the default BEV
        const from = this.bevAsPersp();
        const to: Sph = { az: -Math.PI / 2, pol: TOP_POL, dist: this.perspDistForSpan(this.bevSpan), target: this.focus.clone() };
        this.fly(from, to, 0.8, () => this.finishToBev(to));
        return;
      }
      const from = prev === 'bev' ? this.bevAsPersp() : this.currentPerspSph();
      const target = prev === 'bev' ? this.focus.clone() : new Vector3(from.target.x, from.target.y, this.focus.z);
      const span = prev === 'bev' ? this.bevSpan : Math.min(260, Math.max(30, from.dist * 1.1));
      const to: Sph = { az: -Math.PI / 2, pol: TOP_POL, dist: this.perspDistForSpan(span), target };
      if (dur === 0) this.finishToBev(to);
      else this.fly(from, to, dur, () => this.finishToBev(to));
      return;
    }
    let from: Sph;
    if (prev === 'bev') from = this.bevAsPersp();
    else from = this.currentPerspSph();
    let to: Sph;
    if (view === 'follow' && opts.followPos) {
      to = this.chasePose(opts.followPos, opts.followHeading ?? 0);
      this.followTarget = opts.followPos.clone();
      this.followHeading = opts.followHeading ?? 0;
    } else if (prev === 'bev') {
      to = this.orbitDefault(from.target, Math.min(220, Math.max(40, from.dist * 0.75)));
    } else if (prev === 'follow' && view === 'orbit') {
      this.stage.setCamera(this.stage.persp);
      this.applyEnabled();
      return;
    } else {
      to = this.orbitDefault(this.focus);
    }
    if (dur === 0) {
      this.stage.setCamera(this.stage.persp);
      this.applyPersp(to);
      this.finishPersp();
    } else {
      this.fly(from, to, dur, () => this.finishPersp(), easeInOutCubic);
    }
  }

  /** Feed the followed object's (smoothly interpolated) position each frame. */
  setFollow(pos: Vector3 | null, heading: number): void {
    if (!pos) {
      this.followTarget = null;
      return;
    }
    if (!this.followTarget) this.followTarget = pos.clone();
    else this.followTarget.copy(pos);
    this.followHeading = heading;
  }

  update(dt: number): void {
    this.clock += dt;
    const f = this.flight;
    if (f) {
      f.t = Math.min(1, f.t + dt / Math.max(f.dur, 1e-6));
      const k = f.ease(f.t);
      // when following, keep the flight's end point glued to the moving object
      if (this.view === 'follow' && this.followTarget) f.to.target.copy(this.followTarget);
      const s: Sph = {
        az: f.from.az + (f.to.az - f.from.az) * k,
        pol: f.from.pol + (f.to.pol - f.from.pol) * k,
        dist: f.from.dist * Math.pow(f.to.dist / f.from.dist, k),
        target: this.tmp.copy(f.from.target).lerp(f.to.target, k).clone(),
      };
      this.applyPersp(s);
      if (f.t >= 1) f.done();
      return;
    }
    if (this.view === 'bev') {
      this.bevControls.update();
      return;
    }
    if (this.view === 'follow' && this.followTarget) {
      const c = this.orbitControls;
      const cam = this.stage.persp;
      const k = 1 - Math.exp(-dt * 7);
      const delta = this.tmp.copy(this.followTarget).sub(c.target).multiplyScalar(k);
      c.target.add(delta);
      cam.position.add(delta);
      // gently swing behind the vehicle when the user is not interacting
      if (this.clock - this.lastInteraction > 2.5) {
        const off = cam.position.clone().sub(c.target);
        const az = Math.atan2(off.y, off.x);
        let d = this.followHeading + Math.PI - 0.35 - az;
        while (d > Math.PI) d -= 2 * Math.PI;
        while (d < -Math.PI) d += 2 * Math.PI;
        const rot = d * (1 - Math.exp(-dt * 0.9));
        const r = Math.hypot(off.x, off.y);
        const na = az + rot;
        cam.position.set(c.target.x + r * Math.cos(na), c.target.y + r * Math.sin(na), c.target.z + off.z);
      }
    }
    this.orbitControls.update();
  }

  private applyEnabled(): void {
    const flying = this.flight !== null;
    this.bevControls.enabled = !flying && this.view === 'bev';
    this.orbitControls.enabled = !flying && this.view !== 'bev';
  }

  /** Shared easing for other animations. */
  static easeOut = easeOutQuint;
}
