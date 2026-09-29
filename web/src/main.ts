import './ui/styles.css';
import { Color, Raycaster, Vector2, Vector3 } from 'three';
import { PointCache, loadIndex, loadSequence, type Sequence } from './data/loader';
import { MatchCache, type FrameMatch } from './data/match';
import type { Detection, IndexFile } from './data/types';
import { VEHICLE_CLASSES, groupOf } from './data/types';
import { BoxLayer, type BoxStyle } from './scene/boxes';
import { CameraRig, type CameraPose, type ViewName } from './scene/camera';
import { Grid, SensorLayer } from './scene/environment';
import { MapLayer, estimateGroundZ } from './scene/map';
import { HEX, classColor, classHex } from './scene/palette';
import { COLOR_MODES, PointLayer, type ColorMode } from './scene/points';
import { Stage } from './scene/stage';
import { TrailLayer } from './scene/trails';
import { Hud, LAYER_DEFS, gtRefLabel, type LayerKey, type Layers } from './ui/hud';
import { Timeline } from './ui/timeline';

// ------------------------------------------------------------------------------------------
// URL parameters

const params = new URLSearchParams(location.search);
const boolParam = (k: string, def: boolean) => {
  const v = params.get(k);
  if (v === null) return def;
  return !['0', 'false', 'no', 'off'].includes(v.toLowerCase());
};
const VIEWS: ViewName[] = ['bev', 'orbit', 'follow'];
const LAYER_KEYS = LAYER_DEFS.map((d) => d.key);

const DEFAULT_LAYERS: Layers = {
  det: true, gt: false, compare: false, map: true, trails: true, labels: false, sensors: true, grid: true,
};

/**
 * `layers=det,gt,map,trails` sets the overlay layers exactly (det, gt, compare, map, trails, labels);
 * the ambient layers `sensors` and `grid` stay on unless listed with a minus (`-grid`).
 * Tokens prefixed with `+`/`-` only modify the defaults (e.g. `layers=+gt,-trails`).
 */
function parseLayers(v: string | null): Layers {
  if (v === null) return { ...DEFAULT_LAYERS };
  const tokens = v.split(',').map((s) => s.trim().toLowerCase()).filter(Boolean);
  const relative = tokens.length > 0 && tokens.every((t) => t.startsWith('+') || t.startsWith('-'));
  const l = { ...DEFAULT_LAYERS };
  const on = new Set<string>();
  if (!relative) for (const k of LAYER_KEYS) if (k !== 'sensors' && k !== 'grid') l[k] = false;
  for (const t of tokens) {
    const key = t.replace(/^[+-]/, '') === 'detections' ? 'det' : t.replace(/^[+-]/, '');
    if (!(LAYER_KEYS as string[]).includes(key)) continue;
    l[key as LayerKey] = !t.startsWith('-');
    on.add(key);
  }
  if (tokens.includes('all')) for (const k of LAYER_KEYS) if (!on.has(k)) l[k] = k !== 'labels' && k !== 'compare';
  return l;
}

const TMP_COLOR = new Color();
const nextAnimationFrame = () => new Promise<void>((r) => requestAnimationFrame(() => r()));
const escapeHtml = (s: string) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);

interface ViewerState {
  sequence: string | null;
  frame: number;
  nFrames: number;
  pointsFrame: number;
  nPoints: number;
  view: ViewName;
  layers: Layers;
  colorMode: ColorMode;
  pointSize: number;
  playing: boolean;
  speed: number;
  selected: number | null;
  ui: boolean;
  cachedFrames: number;
  introRunning: boolean;
  title: string | null;
  camera: CameraPose;
}

// ------------------------------------------------------------------------------------------

class App {
  readonly stage: Stage;
  readonly rig: CameraRig;
  readonly points: PointLayer;
  readonly detBoxes: BoxLayer;
  readonly gtBoxes: BoxLayer;
  readonly trails: TrailLayer;
  readonly mapLayer: MapLayer;
  readonly grid: Grid;
  readonly sensors: SensorLayer;
  readonly hud: Hud;
  readonly timeline: Timeline;

  index: IndexFile | null = null;
  seq: Sequence | null = null;
  cache: PointCache | null = null;
  matches: MatchCache | null = null;
  frame = 0;
  shownPointsFrame = -1;
  pendingPointsFrame = -1;
  playing = false;
  speed = 1;
  private acc = 0;
  buffering = false;
  layers: Layers;
  colorMode: ColorMode = 'height';
  pointSize = 1;
  selectedId: number | null = null;
  hoveredId: number | null = null;
  uiVisible: boolean;
  /** big title overlay (video titles), set via ?title= or __viewer.setCaption() */
  title: string | null = null;
  showCaption: boolean;
  groundZ = 0.45;
  focus = new Vector3();
  private scan = { active: false, t: 0, dur: 2.4, maxR: 150 };
  private introRunning = false;
  private introResolve: (() => void) | null = null;
  readonly introDone: Promise<void>;
  private holdPlaybackUntilScan = false;
  private autoplay = false;
  private lastTime = performance.now();
  private readonly raycaster = new Raycaster();
  private readonly mouse = new Vector2();
  private mouseValid = false;
  private mouseMoved = false;
  private downPos: { x: number; y: number } | null = null;
  private fpsFrames = 0;
  private fpsTime = 0;
  private fps = 0;
  private statusTimer = 0;
  private readonly followPos = new Vector3();
  private seqToken = 0;

  constructor() {
    const container = document.getElementById('viewport')!;
    this.stage = new Stage(container);
    this.rig = new CameraRig(this.stage);
    this.points = new PointLayer(200_000);
    this.detBoxes = new BoxLayer(this.stage, false);
    this.gtBoxes = new BoxLayer(this.stage, true);
    this.trails = new TrailLayer(this.stage);
    this.mapLayer = new MapLayer(this.stage);
    this.grid = new Grid();
    this.sensors = new SensorLayer(this.stage);
    const scene = this.stage.scene;
    scene.add(this.grid.mesh, this.mapLayer.root, this.points.points, this.trails.root, this.gtBoxes.root, this.detBoxes.root, this.sensors.root);

    this.layers = parseLayers(params.get('layers'));
    const cm = params.get('color') as ColorMode | null;
    if (cm && COLOR_MODES.includes(cm)) this.colorMode = cm;
    const size = Number(params.get('size'));
    if (size > 0) this.pointSize = Math.min(4, Math.max(0.25, size));
    const speed = Number(params.get('speed'));
    if ([0.5, 1, 2, 4].includes(speed)) this.speed = speed;
    this.uiVisible = boolParam('ui', true);
    this.showCaption = boolParam('caption', !this.uiVisible);
    const title = params.get('title');
    if (title) this.title = escapeHtml(title);
    const sel = params.get('select');
    if (sel !== null && sel !== '' && Number.isFinite(Number(sel))) this.selectedId = Number(sel);

    this.introDone = new Promise((r) => (this.introResolve = r));

    this.hud = new Hud({
      selectSequence: (id) => void this.setSequence(id),
      setView: (v) => this.setView(v),
      togglePlay: () => this.setPlaying(!this.playing),
      setSpeed: (s) => this.setSpeed(s),
      toggleLayer: (k) => this.toggleLayer(k),
      setColorMode: (m) => this.setColorMode(m),
      setPointSize: (v) => this.setPointSize(v),
      deselect: () => this.select(null),
      followSelected: () => this.setView(this.rig.view === 'follow' ? 'orbit' : 'follow'),
    });
    this.timeline = new Timeline(
      document.getElementById('chart-svg') as unknown as SVGSVGElement,
      document.getElementById('chart-tip')!,
      (f) => {
        this.setPlaying(false);
        this.applyFrame(f);
      },
    );
    this.hud.setUiVisible(this.uiVisible);
    this.hud.setTitle(this.title);
    this.hud.setSpeed(this.speed);
    this.hud.setColorMode(this.colorMode);
    this.hud.setPointSize(this.pointSize);
    this.points.setMode(this.colorMode);
    this.points.setSizeMul(this.pointSize);
    this.applyLayerVisibility();

    const dock = document.querySelector<HTMLElement>('.dock')!;
    const root = document.documentElement;
    new ResizeObserver(() => root.style.setProperty('--dock-h', `${dock.offsetHeight + 10}px`)).observe(dock);

    this.bindPointer();
    this.bindKeys();
  }

  // ---------------------------------------------------------------------------------------- loading

  async init(): Promise<void> {
    this.hud.splash('Loading sequence index…');
    try {
      this.index = await loadIndex();
    } catch {
      this.index = null;
    }
    if (!this.index || !this.index.sequences?.length) {
      this.hud.splash(
        'No data found at <code>data/index.json</code>.<br/>Export real data with the Python pipeline, or run <code>npm run mock</code> to generate a synthetic sequence.',
        true,
      );
      throw new Error('no data');
    }
    const want = params.get('seq');
    const seqId = this.index.sequences.find((s) => s.id === want)?.id ?? this.index.sequences[0].id;
    this.hud.setSequences(this.index.sequences, seqId);

    const frameParam = params.get('frame');
    const startFrame = frameParam !== null ? Math.max(0, Math.floor(Number(frameParam)) || 0) : 0;
    this.autoplay = boolParam('play', frameParam === null && this.uiVisible);
    await this.loadSeq(seqId, startFrame, false);
    this.hud.splash(null);

    const view = (VIEWS as string[]).includes(params.get('view') ?? '') ? (params.get('view') as ViewName) : 'bev';
    const intro = boolParam('intro', true);
    const followInfo = view === 'follow' ? this.pickFollowTarget() : null;
    if (view === 'follow' && followInfo) {
      this.selectedId = followInfo.id;
      this.rig.followLength = followInfo.length;
    }
    if (intro) {
      this.introRunning = true;
      this.holdPlaybackUntilScan = true;
      this.startScan(3.0, 0.3);
      this.rig.intro(view === 'follow' && !followInfo ? 'orbit' : view, 3.6, followInfo?.pos, followInfo?.heading);
    } else {
      this.rig.setView('bev', { instant: true });
      if (view !== 'bev') this.setView(view, true);
      this.introResolve?.();
    }
    this.hud.setView(this.rig.view, this.selectedId !== null);
    this.setPlaying(this.autoplay && !intro);
    this.refreshOverlays();
    requestAnimationFrame(this.loop);
  }

  private async loadSeq(id: string, startFrame: number, scanWave: boolean): Promise<void> {
    const token = ++this.seqToken;
    const seq = await loadSequence(id);
    if (token !== this.seqToken) return;
    this.cache?.dispose();
    this.seq = seq;
    const meta = seq.meta;
    this.cache = new PointCache(id, meta, 80, 15, 6);
    this.points.ensureCapacity(Math.max(1, seq.maxPoints));
    this.groundZ = estimateGroundZ(seq.map, 0.45);
    const pf = meta.point_format ?? { scale: 0.01, origin: [0, 0, 0] as [number, number, number], stride_bytes: 8, fields: [] };
    this.points.setFormat(pf.scale ?? 0.01, pf.origin ?? [0, 0, 0], this.groundZ);

    // scene focus: centroid of the LiDAR poles, else the bounds centre
    const b = meta.bounds ?? { xmin: -80, xmax: 80, ymin: -80, ymax: 80, zmin: -2, zmax: 12 };
    let span = Math.min(b.ymax - b.ymin, b.xmax - b.xmin, 160);
    if (meta.lidars?.length) {
      let sx = 0;
      let sy = 0;
      for (const l of meta.lidars) { sx += l.position[0]; sy += l.position[1]; }
      this.focus.set(sx / meta.lidars.length, sy / meta.lidars.length, this.groundZ);
      let spread = 0;
      for (const l of meta.lidars) spread = Math.max(spread, Math.abs(l.position[0] - this.focus.x), Math.abs(l.position[1] - this.focus.y));
      if (spread > 5) span = Math.min(span, Math.max(70, spread * 2 * 2.4));
    } else {
      this.focus.set((b.xmin + b.xmax) / 2, (b.ymin + b.ymax) / 2, this.groundZ);
    }
    this.rig.setFocus(this.focus, span);
    const half = Math.max(b.xmax - b.xmin, b.ymax - b.ymin) / 2;
    this.scan.maxR = half * 1.5 + 20;
    this.grid.place(this.focus.x, this.focus.y, this.groundZ - 0.03, half * 1.35);
    this.mapLayer.build(seq.map, this.groundZ);
    this.sensors.build(meta.lidars ?? [], this.groundZ);
    this.matches = seq.gt ? new MatchCache(seq.det.frames, seq.gt.frames) : null;
    if (!seq.gt) {
      this.layers.gt = false;
      this.layers.compare = false;
    }

    const present = new Set<string>();
    for (const fr of [...seq.det.frames, ...(seq.gt?.frames ?? [])]) for (const d of fr) present.add(d.cls);
    const classes = VEHICLE_CLASSES.filter((c) => ['car', 'van', 'truck', 'bus'].includes(c) || present.has(c));
    this.hud.setClasses(classes);
    this.hud.setMetrics(seq.metrics);
    this.timeline.setGtLabel(seq.gt ? gtRefLabel(seq.gt.refKind) : 'ground truth');
    this.timeline.setData(
      seq.det.counts.map((c) => c.vehicle),
      seq.gt ? seq.gt.refCounts.map((c) => c.vehicle) : null,
      meta.fps || 10,
      meta.n_frames,
    );
    this.hud.setSequences(this.index?.sequences ?? [], id);
    if (this.selectedId !== null && !seq.det.tracks.has(this.selectedId)) this.selectedId = null;
    this.shownPointsFrame = -1;
    this.applyFrame(Math.min(startFrame, meta.n_frames - 1), true);
    await this.ensurePoints(this.frame);
    this.hud.setLayers(this.layers, !!seq.gt);
    if (scanWave) this.startScan(1.5, 0);
  }

  async setSequence(id: string): Promise<void> {
    if (this.seq?.id === id) return;
    const wasPlaying = this.playing;
    this.setPlaying(false);
    this.hud.setStatus(`loading <span class="hl">${escapeHtml(id)}</span>…`, '', '');
    await this.loadSeq(id, 0, true);
    const u = new URL(location.href);
    u.searchParams.set('seq', id);
    history.replaceState(null, '', u);
    if (this.rig.view === 'follow') this.setView('bev');
    if (wasPlaying) this.setPlaying(true);
  }

  private startScan(dur: number, delay: number): void {
    this.scan.active = true;
    this.scan.dur = dur;
    this.scan.t = -delay;
    this.points.setScan(this.focus.x, this.focus.y, 0);
  }

  private scanRadius(): number {
    if (!this.scan.active) return Infinity;
    const t = Math.max(0, this.scan.t / this.scan.dur);
    return this.scan.maxR * (1 - Math.pow(1 - Math.min(1, t), 1.5));
  }

  // ---------------------------------------------------------------------------------------- frames

  applyFrame(f: number, instant = false): void {
    const seq = this.seq;
    if (!seq || !this.cache) return;
    const n = seq.meta.n_frames;
    f = Math.max(0, Math.min(n - 1, Math.floor(f)));
    this.frame = f;
    const buf = this.cache.peek(f);
    if (buf) {
      this.points.upload(buf);
      this.shownPointsFrame = f;
      this.pendingPointsFrame = -1;
    } else {
      this.pendingPointsFrame = f;
      const cache = this.cache;
      cache.get(f).then((b) => {
        if (cache === this.cache && this.pendingPointsFrame === f) {
          this.points.upload(b);
          this.shownPointsFrame = f;
          this.pendingPointsFrame = -1;
          this.updateStatus();
        }
      }).catch(() => undefined);
    }
    this.cache.prefetch(f);
    this.refreshOverlays(instant);
    this.timeline.setFrame(f);
    const t = this.frameTime(f);
    this.hud.setFrame(f, n, t);
    this.updateStatus();
  }

  private frameTime(f: number): number {
    const m = this.seq?.meta;
    if (!m) return 0;
    const ts = m.timestamps;
    if (ts && ts.length > f && ts.length > 0) return ts[f] - ts[0];
    return f / (m.fps || 10);
  }

  private async ensurePoints(f: number): Promise<void> {
    if (!this.cache || this.shownPointsFrame === f) return;
    const cache = this.cache;
    try {
      const b = await cache.get(f);
      if (cache === this.cache && this.frame === f) {
        this.points.upload(b);
        this.shownPointsFrame = f;
        this.pendingPointsFrame = -1;
      }
    } catch {
      /* missing frame: keep the last cloud */
    }
  }

  /** Re-style boxes, trails and HUD for the current frame and settings. */
  refreshOverlays(instant = false): void {
    const seq = this.seq;
    if (!seq) return;
    const f = this.frame;
    const dets = seq.det.frames[f] ?? [];
    const gts = seq.gt?.frames[f] ?? [];
    const compare = this.layers.compare && !!seq.gt;
    const match = compare && this.matches ? this.matches.get(f) : null;
    const scanR = this.scanRadius();
    const fx = this.focus.x;
    const fy = this.focus.y;
    const hidden = (d: Detection) => scanR !== Infinity && Math.hypot(d.box[0] - fx, d.box[1] - fy) > scanR - 4;

    if (this.layers.det) {
      this.detBoxes.update(dets, (d, i, s) => this.styleDet(d, i, s, match, hidden(d)));
    } else this.detBoxes.clear();

    if (seq.gt && (compare || this.layers.gt)) {
      this.gtBoxes.update(gts, (g, i, s) => {
        // labels never seen by the infrastructure LiDARs are out of coverage: not drawn
        if (hidden(g) || g.trackable === false) { s.visible = false; return; }
        const occluded = g.ignore === true;
        if (compare && match) {
          const matched = match.gtToDet[i] >= 0;
          if (matched && !this.layers.gt) { s.visible = false; return; }
          if (!matched && match.gtExcused[i]) {
            // occluded in this frame: not a miss
            s.color.set(HEX.occluded);
            s.intensity = 0.4;
            s.linewidth = 0.9;
            s.fillOpacity = 0;
            s.bodyOpacity = 0;
            if (this.layers.labels || this.hoveredId === g.id) {
              s.label = `<i></i><b>GT ${g.id}</b><span>${escapeHtml(g.cls)}</span><em>occluded</em>`;
              s.labelColor = HEX.occluded;
            }
            return;
          }
          const miss = !matched;
          s.color.set(miss ? HEX.fn : HEX.gt);
          s.intensity = miss ? 1.35 : 0.55;
          s.linewidth = miss ? 1.8 : 1.0;
          s.fillOpacity = miss ? 0.09 : 0;
          s.bodyOpacity = miss ? 0.03 : 0;
          if (miss && (this.layers.labels || this.hoveredId === g.id)) {
            s.label = `<i></i><b>GT ${g.id}</b><span>${escapeHtml(g.cls)}</span><em>missed</em>`;
            s.labelColor = HEX.fn;
          }
          return;
        }
        s.color.set(occluded ? HEX.occluded : HEX.gt);
        s.intensity = occluded ? 0.35 : 0.8;
        s.linewidth = occluded ? 0.8 : 1.1;
        s.fillOpacity = 0;
        s.bodyOpacity = 0;
      });
    } else this.gtBoxes.clear();

    if (this.layers.trails) {
      this.trails.update(seq.det, f, dets, (d) => {
        if (match) {
          const i = dets.indexOf(d);
          return TMP_COLOR.set(match.detToGt[i] >= 0 ? HEX.tp : HEX.fp);
        }
        return classColor(d.cls);
      }, this.selectedId, 0.12);
    } else this.trails.clear();

    // HUD
    const dc = seq.det.counts[f] ?? { vehicle: 0, vru: 0 };
    const gt = seq.gt;
    this.hud.updateCounts({
      det: dc,
      gt: gt?.refCounts[f] ?? null,
      gtRef: gt?.refKind ?? 'all',
      gtAllVehicles: gt && gt.refKind !== 'all' ? gt.counts[f]?.vehicle ?? null : null,
      gtAllUnique: gt && gt.refKind !== 'all' ? gt.uniqueVehicles : null,
      uniqueSoFar: seq.det.cumulative('vehicle', f),
      uniqueTotal: seq.det.uniqueVehicles,
      uniqueSoFarGt: gt ? gt.cumulativeRef(f) : null,
      uniqueTotalGt: gt ? gt.uniqueRefVehicles : null,
    }, instant);
    if (match) this.hud.setCompareStats(match.tp, match.fp, match.fn);
    this.updateSelectionCard(match);
    this.updateCaption();
  }

  private styleDet(d: Detection, i: number, s: BoxStyle, match: FrameMatch | null, hidden: boolean): void {
    if (hidden) { s.visible = false; return; }
    const sel = d.id === this.selectedId;
    const hov = d.id === this.hoveredId;
    const moving = d.moving ?? (d.speed ?? 0) > 1;
    if (match) s.color.set(match.detToGt[i] >= 0 ? HEX.tp : HEX.fp);
    else s.color.copy(classColor(d.cls));
    if (moving) {
      s.intensity = 1.55; s.linewidth = 2.0; s.fillOpacity = 0.13; s.bodyOpacity = 0.045;
    } else {
      s.intensity = 0.72; s.linewidth = 1.25; s.fillOpacity = 0.06; s.bodyOpacity = 0.018;
    }
    if (match) s.intensity = Math.max(s.intensity, 1.15);
    if ((d.score ?? 1) < 0.4) s.intensity *= 0.8;
    // tracker-predicted boxes without a supporting detection
    if (d.coasted) { s.intensity *= 0.55; s.fillOpacity *= 0.4; s.bodyOpacity = 0; }
    if (hov) { s.intensity *= 1.45; s.linewidth += 0.7; s.fillOpacity += 0.06; }
    if (sel) { s.intensity = 1.9; s.linewidth = 2.8; s.fillOpacity = 0.2; s.bodyOpacity = 0.06; }
    if (sel || hov || this.layers.labels) {
      const kmh = (d.speed ?? 0) * 3.6;
      s.label = `<i></i><b>#${d.id}</b><span>${escapeHtml(d.cls)}</span><em>${moving ? `${kmh.toFixed(0)} km/h` : 'parked'}</em>`;
      s.labelColor = match ? (match.detToGt[i] >= 0 ? HEX.tp : HEX.fp) : classHex(d.cls);
    }
  }

  private updateSelectionCard(match: FrameMatch | null): void {
    const seq = this.seq;
    if (!seq || this.selectedId === null) {
      this.hud.setSelection(null);
      return;
    }
    const id = this.selectedId;
    const dets = seq.det.frames[this.frame] ?? [];
    const idx = dets.findIndex((d) => d.id === id);
    const d = idx >= 0 ? dets[idx] : undefined;
    const tr = seq.det.tracks.get(id);
    if (!d && !tr) {
      this.hud.setSelection(null);
      return;
    }
    const cls = d?.cls ?? tr!.cls;
    this.hud.setSelection({
      id,
      cls,
      present: !!d,
      moving: d ? d.moving ?? (d.speed ?? 0) > 1 : false,
      speedKmh: (d?.speed ?? 0) * 3.6,
      dims: d ? [d.box[3], d.box[4], d.box[5]] : null,
      headingDeg: d ? ((((d.box[6] * 180) / Math.PI) % 360) + 360) % 360 : null,
      score: d?.score ?? null,
      nPoints: d?.n_points ?? null,
      first: tr?.first ?? this.frame,
      last: tr?.last ?? this.frame,
      nFrames: tr?.n_frames ?? (tr ? tr.last - tr.first + 1 : 1),
      maxSpeedKmh: tr?.max_speed !== undefined ? tr.max_speed * 3.6 : null,
      fps: seq.meta.fps || 10,
      match: match && idx >= 0 ? (match.detToGt[idx] >= 0 ? 'true positive' : 'false positive') : null,
      coasted: d?.coasted ?? false,
      following: this.rig.view === 'follow',
    });
  }

  private updateCaption(): void {
    if (!this.showCaption || this.uiVisible) {
      this.hud.setCaption(null);
      return;
    }
    const seq = this.seq;
    if (!seq) return;
    const c = seq.det.counts[this.frame];
    const u = seq.det.cumulative('vehicle', this.frame);
    this.hud.setCaption(
      `<b>${c?.vehicle ?? 0}</b> vehicles in view · <b>${c?.moving ?? 0}</b> moving · <b>${u}</b> counted · ` +
      `t = ${this.frameTime(this.frame).toFixed(1)} s`,
    );
  }

  private updateStatus(): void {
    const seq = this.seq;
    if (!seq) return;
    const n = seq.meta.n_frames;
    const c = seq.det.counts[this.frame];
    const pts = this.points.count;
    const main =
      `${escapeHtml(seq.id)} | frame <span class="hl">${this.frame}</span>/${n} | t=${this.frameTime(this.frame).toFixed(1)} s | ` +
      `${pts.toLocaleString('en-US')} pts | <span class="hl">${c?.vehicle ?? 0}</span> veh · ${c?.vru ?? 0} vru`;
    const lag = this.shownPointsFrame !== this.frame ? ' · loading points' : '';
    this.hud.setStatus(main, `cache ${this.cache?.size ?? 0}/80${lag}`, `${this.fps.toFixed(0)} fps`);
  }

  // ---------------------------------------------------------------------------------------- controls

  setPlaying(p: boolean): void {
    this.playing = p;
    this.acc = 0;
    if (p) this.holdPlaybackUntilScan = this.holdPlaybackUntilScan && this.scan.active;
    this.hud.setPlaying(p, this.buffering, this.seq?.meta.fps || 10, this.speed);
  }

  setSpeed(s: number): void {
    this.speed = s;
    this.hud.setSpeed(s);
    this.hud.setPlaying(this.playing, this.buffering, this.seq?.meta.fps || 10, this.speed);
  }

  setColorMode(m: ColorMode): void {
    if (!COLOR_MODES.includes(m)) return;
    this.colorMode = m;
    this.points.setMode(m);
    this.hud.setColorMode(m);
  }

  setPointSize(v: number): void {
    this.pointSize = Math.min(4, Math.max(0.25, v));
    this.points.setSizeMul(this.pointSize);
    this.hud.setPointSize(this.pointSize);
  }

  toggleLayer(k: LayerKey): void {
    this.setLayers({ [k]: !this.layers[k] });
  }

  setLayers(obj: Partial<Layers>): void {
    for (const [k, v] of Object.entries(obj)) if ((LAYER_KEYS as string[]).includes(k)) this.layers[k as LayerKey] = !!v;
    if (!this.seq?.gt) {
      this.layers.gt = false;
      this.layers.compare = false;
    }
    if (obj.compare && this.layers.compare) this.layers.det = true;
    this.applyLayerVisibility();
    this.refreshOverlays();
  }

  private applyLayerVisibility(): void {
    this.mapLayer.root.visible = this.layers.map;
    this.sensors.root.visible = this.layers.sensors;
    this.grid.mesh.visible = this.layers.grid;
    this.hud.setLayers(this.layers, !!this.seq?.gt);
  }

  select(id: number | null): void {
    this.selectedId = id;
    if (id === null && this.rig.view === 'follow') this.setView('orbit');
    this.refreshOverlays();
    this.hud.setView(this.rig.view, id !== null);
  }

  /** Choose a good track to follow: the selected one, else the longest-remaining moving vehicle. */
  private pickFollowTarget(): { id: number; pos: Vector3; heading: number; length: number } | null {
    const seq = this.seq;
    if (!seq) return null;
    const dets = seq.det.frames[this.frame] ?? [];
    let best: Detection | undefined;
    if (this.selectedId !== null) best = dets.find((d) => d.id === this.selectedId);
    if (!best) {
      let score = -1;
      for (const d of dets) {
        const tr = seq.det.tracks.get(d.id);
        const moving = d.moving ?? (d.speed ?? 0) > 1;
        const s = (moving ? 1000 : 0) + (groupOf(d.cls) === 'vehicle' ? 500 : 0) + (tr ? tr.last - this.frame : 0);
        if (s > score) { score = s; best = d; }
      }
    }
    if (!best) return null;
    return { id: best.id, pos: new Vector3(best.box[0], best.box[1], best.box[2]), heading: best.box[6], length: best.box[3] };
  }

  setView(v: ViewName, instant = false): Promise<void> {
    if (v === 'follow') {
      const t = this.pickFollowTarget();
      if (!t) v = 'orbit';
      else {
        this.selectedId = t.id;
        this.rig.followLength = t.length;
        this.rig.setView('follow', { instant, followPos: t.pos, followHeading: t.heading });
        this.refreshOverlays();
        this.hud.setView('follow', true);
        return this.waitFlight();
      }
    }
    this.rig.setView(v, { instant });
    this.hud.setView(v, this.selectedId !== null);
    this.refreshOverlays();
    return this.waitFlight();
  }

  private async waitFlight(): Promise<void> {
    while (this.rig.flying) await nextAnimationFrame();
  }

  step(delta: number): void {
    if (!this.seq) return;
    this.setPlaying(false);
    const n = this.seq.meta.n_frames;
    this.applyFrame((this.frame + delta + n) % n);
  }

  // ---------------------------------------------------------------------------------------- input

  private bindPointer(): void {
    const el = this.stage.renderer.domElement;
    el.addEventListener('pointermove', (e) => {
      const r = el.getBoundingClientRect();
      this.mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      this.mouseValid = true;
      this.mouseMoved = true;
    });
    el.addEventListener('pointerleave', () => {
      this.mouseValid = false;
      this.setHovered(null);
    });
    el.addEventListener('pointerdown', (e) => (this.downPos = { x: e.clientX, y: e.clientY }));
    el.addEventListener('pointerup', (e) => {
      if (!this.downPos || e.button !== 0) return;
      const moved = Math.hypot(e.clientX - this.downPos.x, e.clientY - this.downPos.y);
      this.downPos = null;
      if (moved > 5) return;
      const id = this.pickAt();
      this.select(id);
    });
    el.addEventListener('dblclick', () => {
      const id = this.pickAt();
      if (id !== null) {
        this.select(id);
        void this.setView('follow');
      }
    });
  }

  private pickAt(): number | null {
    if (!this.layers.det) return null;
    this.raycaster.setFromCamera(this.mouse, this.stage.camera);
    return this.detBoxes.pick(this.raycaster.ray)?.id ?? null;
  }

  private setHovered(id: number | null): void {
    if (id === this.hoveredId) return;
    this.hoveredId = id;
    this.stage.renderer.domElement.style.cursor = id !== null ? 'pointer' : '';
    this.refreshOverlays();
  }

  private bindKeys(): void {
    window.addEventListener('keydown', (e) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA')) {
        if (e.key !== ' ' || t.tagName !== 'INPUT') return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const k = e.key.toLowerCase();
      let handled = true;
      switch (k) {
        case ' ': this.setPlaying(!this.playing); break;
        case 'arrowright': this.step(e.shiftKey ? 10 : 1); break;
        case 'arrowleft': this.step(e.shiftKey ? -10 : -1); break;
        case '1': void this.setView('bev'); break;
        case '2': void this.setView('orbit'); break;
        case '3': void this.setView('follow'); break;
        case 'f': void this.setView(this.rig.view === 'follow' ? 'orbit' : 'follow'); break;
        case 'd': this.toggleLayer('det'); break;
        case 'g': this.toggleLayer('gt'); break;
        case 'c': this.toggleLayer('compare'); break;
        case 'm': this.toggleLayer('map'); break;
        case 't': this.toggleLayer('trails'); break;
        case 'l': this.toggleLayer('labels'); break;
        case 'v': this.setColorMode(COLOR_MODES[(COLOR_MODES.indexOf(this.colorMode) + 1) % COLOR_MODES.length]); break;
        case 'h': this.setUi(!this.uiVisible); break;
        case '?': this.hud.toggleHelp(); break;
        case 'escape': this.hud.toggleHelp(false); this.select(null); break;
        case 'home': this.step(-this.frame); break;
        default: handled = false;
      }
      if (handled) e.preventDefault();
    });
  }

  setTitle(html: string | null): void {
    this.title = html && html.trim() ? html : null;
    this.hud.setTitle(this.title);
  }

  setCameraPose(p: CameraPose): void {
    if (!p || !Array.isArray(p.target) || p.target.length < 3) throw new Error('setCameraPose: target [x, y, z] required');
    this.rig.setPose(p);
    this.hud.setView('orbit', this.selectedId !== null);
  }

  setUi(v: boolean): void {
    this.uiVisible = v;
    this.hud.setUiVisible(v);
    this.updateCaption();
  }

  // ---------------------------------------------------------------------------------------- loop

  private readonly loop = (now: number): void => {
    const dt = Math.min(0.1, Math.max(0, (now - this.lastTime) / 1000));
    this.lastTime = now;
    const seq = this.seq;

    // playback
    if (seq && this.cache && this.playing && !this.holdPlaybackUntilScan) {
      const fps = seq.meta.fps || 10;
      this.acc += dt * fps * this.speed;
      let buffering = false;
      let steps = 0;
      while (this.acc >= 1 && steps < 4) {
        const next = (this.frame + 1) % seq.meta.n_frames;
        if (!this.cache.has(next)) {
          buffering = true;
          this.acc = Math.min(this.acc, 1);
          void this.cache.get(next).catch(() => undefined);
          break;
        }
        this.acc -= 1;
        steps++;
        this.applyFrame(next);
      }
      if (buffering !== this.buffering) {
        this.buffering = buffering;
        this.hud.setPlaying(this.playing, buffering, fps, this.speed);
      }
    }

    // scan wave
    if (this.scan.active) {
      this.scan.t += dt;
      const r = this.scanRadius();
      this.points.setScan(this.focus.x, this.focus.y, r);
      if (this.scan.t >= this.scan.dur) {
        this.scan.active = false;
        this.points.setScan(0, 0, 1e7);
        if (this.holdPlaybackUntilScan) {
          this.holdPlaybackUntilScan = false;
          if (this.autoplay) this.setPlaying(true);
        }
      }
      this.refreshOverlays();
    }

    // follow camera
    if (seq && this.rig.view === 'follow' && this.selectedId !== null) {
      const d = seq.det.find(this.frame, this.selectedId);
      if (d) {
        this.followPos.set(d.box[0], d.box[1], d.box[2]);
        this.rig.setFollow(this.followPos, d.box[6]);
      }
    }
    const wasFlying = this.rig.flying;
    this.rig.update(dt);
    if (wasFlying && !this.rig.flying && this.introRunning) {
      this.introRunning = false;
      this.introResolve?.();
    }

    const isOrtho = this.stage.camera === this.stage.ortho;
    this.points.setCameraParams(isOrtho, this.stage.orthoPxPerMetre(), this.stage.perspPxFactor(), this.stage.pixelRatio);
    this.sensors.update(now / 1000);

    // hover picking (once per rendered frame)
    if (this.mouseMoved && this.mouseValid && !this.rig.flying) {
      this.mouseMoved = false;
      this.setHovered(this.pickAt());
      if (this.layers.sensors) {
        const el = this.stage.renderer.domElement;
        const r = el.getBoundingClientRect();
        this.sensors.hover(this.stage.camera, ((this.mouse.x + 1) / 2) * r.width, ((1 - this.mouse.y) / 2) * r.height, r.width, r.height);
      }
    }

    this.stage.render();

    // stats
    this.fpsFrames++;
    this.fpsTime += dt;
    if (this.fpsTime >= 0.5) {
      this.fps = this.fpsFrames / this.fpsTime;
      this.fpsFrames = 0;
      this.fpsTime = 0;
    }
    this.statusTimer += dt;
    if (this.statusTimer > 0.25) {
      this.statusTimer = 0;
      this.updateStatus();
      if (this.cache) {
        const cache = this.cache;
        this.timeline.setBuffered((f) => cache.has(f));
      }
    }
    requestAnimationFrame(this.loop);
  };

  /** Render synchronously (used by the recording API). */
  renderNow(): void {
    const isOrtho = this.stage.camera === this.stage.ortho;
    this.points.setCameraParams(isOrtho, this.stage.orthoPxPerMetre(), this.stage.perspPxFactor(), this.stage.pixelRatio);
    this.stage.render();
  }

  // ---------------------------------------------------------------------------------------- recording API

  getState(): ViewerState {
    return {
      sequence: this.seq?.id ?? null,
      frame: this.frame,
      nFrames: this.seq?.meta.n_frames ?? 0,
      pointsFrame: this.shownPointsFrame,
      nPoints: this.points.count,
      view: this.rig.view,
      layers: { ...this.layers },
      colorMode: this.colorMode,
      pointSize: this.pointSize,
      playing: this.playing,
      speed: this.speed,
      selected: this.selectedId,
      ui: this.uiVisible,
      cachedFrames: this.cache?.size ?? 0,
      introRunning: this.introRunning,
      title: this.title,
      camera: this.rig.getPose(),
    };
  }

  async apiSetFrame(i: number): Promise<void> {
    if (!this.seq) return;
    this.setPlaying(false);
    this.applyFrame(i);
    await this.ensurePoints(this.frame);
    this.refreshOverlays(true);
    this.renderNow();
    await nextAnimationFrame();
  }
}

// ------------------------------------------------------------------------------------------
// bootstrap + window.__viewer

declare global {
  interface Window {
    __viewer: {
      ready: Promise<void>;
      introDone: Promise<void>;
      setSequence(id: string): Promise<void>;
      setFrame(i: number): Promise<void>;
      setView(name: ViewName, opts?: { instant?: boolean }): Promise<void>;
      setLayers(obj: Partial<Layers>): void;
      getState(): ViewerState;
      play(): void;
      pause(): void;
      setSpeed(s: number): void;
      setColorMode(m: ColorMode): void;
      setPointSize(v: number): void;
      select(id: number | null): void;
      setUi(visible: boolean): void;
      /** big title overlay for videos (HTML; `<small>` = subtitle, `<span class="kicker">` = overline); null hides it */
      setCaption(html: string | null): void;
      /** instant perspective camera pose (degrees / metres); switches to the orbit view */
      setCameraPose(pose: CameraPose): void;
      getCameraPose(): CameraPose;
    };
  }
}

const app = new App();
const ready = app.init().then(async () => {
  // resolve once the first frame (points included) is on screen
  await nextAnimationFrame();
  await nextAnimationFrame();
});
ready.catch((err) => console.warn('viewer init:', err));

window.__viewer = {
  ready,
  introDone: app.introDone,
  setSequence: async (id) => {
    await ready;
    await app.setSequence(id);
    app.renderNow();
    await nextAnimationFrame();
  },
  setFrame: async (i) => {
    await ready;
    await app.apiSetFrame(i);
  },
  setView: async (name, opts) => {
    await ready;
    await app.setView(name, opts?.instant ?? false);
  },
  setLayers: (obj) => app.setLayers(obj),
  getState: () => app.getState(),
  play: () => app.setPlaying(true),
  pause: () => app.setPlaying(false),
  setSpeed: (s) => app.setSpeed(s),
  setColorMode: (m) => app.setColorMode(m),
  setPointSize: (v) => app.setPointSize(v),
  select: (id) => app.select(id),
  setUi: (v) => app.setUi(v),
  setCaption: (html) => app.setTitle(html),
  setCameraPose: (pose) => {
    app.setCameraPose(pose);
    app.renderNow();
  },
  getCameraPose: () => app.rig.getPose(),
};
