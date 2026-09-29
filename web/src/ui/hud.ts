import type { FrameCounts, IndexSequence, MetricsFile } from '../data/types';
import type { ColorMode } from '../scene/points';
import type { ViewName } from '../scene/camera';
import { CLASS_HEX, HEX } from '../scene/palette';

export type LayerKey = 'det' | 'gt' | 'compare' | 'map' | 'trails' | 'labels' | 'sensors' | 'grid';
export type Layers = Record<LayerKey, boolean>;

export const LAYER_DEFS: { key: LayerKey; label: string; kbd?: string; color: string }[] = [
  { key: 'det', label: 'Detections', kbd: 'D', color: CLASS_HEX.car },
  { key: 'gt', label: 'Ground truth', kbd: 'G', color: HEX.gt },
  { key: 'compare', label: 'Compare', kbd: 'C', color: HEX.tp },
  { key: 'map', label: 'HD map', kbd: 'M', color: '#cdd6e4' },
  { key: 'trails', label: 'Trails', kbd: 'T', color: HEX.accent },
  { key: 'labels', label: 'All labels', kbd: 'L', color: '#ffffff' },
  { key: 'sensors', label: 'Sensors', color: HEX.accent },
  { key: 'grid', label: 'Grid', color: '#5b7aa0' },
];

export interface HudActions {
  selectSequence(id: string): void;
  setView(v: ViewName): void;
  togglePlay(): void;
  setSpeed(s: number): void;
  toggleLayer(k: LayerKey): void;
  setColorMode(m: ColorMode): void;
  setPointSize(v: number): void;
  deselect(): void;
  followSelected(): void;
}

export type GtRef = 'trackable' | 'visible' | 'all';

export function gtRefLabel(kind: GtRef): string {
  return kind === 'trackable' ? 'GT (in coverage)' : kind === 'visible' ? 'GT (visible)' : 'GT';
}

const GT_REF_HELP: Record<GtRef, string> = {
  trackable: 'Ground truth restricted to vehicles inside the LiDAR coverage (seen with ≥5 points in ≥5 frames).',
  visible: 'Ground truth restricted to vehicles with ≥5 LiDAR points in the current frame.',
  all: 'All ground-truth labels.',
};

export interface CountView {
  det: FrameCounts;
  /** reference GT counts (see gtRef) */
  gt: FrameCounts | null;
  gtRef: GtRef;
  /** all GT labels incl. occluded / out of coverage, when the reference is a subset */
  gtAllVehicles: number | null;
  gtAllUnique: number | null;
  uniqueSoFar: number;
  uniqueTotal: number;
  uniqueSoFarGt: number | null;
  uniqueTotalGt: number | null;
}

export interface SelectionView {
  id: number;
  cls: string;
  present: boolean;
  moving: boolean;
  speedKmh: number;
  dims: [number, number, number] | null;
  headingDeg: number | null;
  score: number | null;
  nPoints: number | null;
  first: number;
  last: number;
  nFrames: number;
  maxSpeedKmh: number | null;
  fps: number;
  match: string | null;
  coasted: boolean;
  following: boolean;
}

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const fmt = (v: number | undefined | null, d = 2) => (v === undefined || v === null || Number.isNaN(v) ? '–' : v.toFixed(d));
const pct = (v: number | undefined | null) => (v === undefined || v === null ? '–' : `${(v * 100).toFixed(1)}`);

/** Number that eases towards its target value. */
class AnimatedNumber {
  private value = 0;
  private target = 0;
  private from = 0;
  private t0 = 0;
  private raf = 0;
  constructor(private readonly node: HTMLElement, private readonly dur = 380) {}
  set(v: number, instant = false): void {
    if (v === this.target && !instant) return;
    this.from = this.value;
    this.target = v;
    this.t0 = performance.now();
    if (instant) {
      this.value = v;
      this.node.textContent = String(Math.round(v));
      return;
    }
    if (!this.raf) this.raf = requestAnimationFrame(this.tick);
  }
  private readonly tick = (now: number): void => {
    const t = Math.min(1, (now - this.t0) / this.dur);
    const k = 1 - Math.pow(1 - t, 3);
    this.value = this.from + (this.target - this.from) * k;
    this.node.textContent = String(Math.round(this.value));
    this.raf = t < 1 ? requestAnimationFrame(this.tick) : 0;
  };
}

export class Hud {
  private readonly big = new AnimatedNumber($('c-vehicles'));
  private readonly uniq = new AnimatedNumber($('u-so-far'));
  private classRows = new Map<string, { row: HTMLElement; bar: HTMLElement; gt: HTMLElement; n: HTMLElement; ngt: HTMLElement }>();
  private readonly toggles = new Map<LayerKey, HTMLButtonElement>();

  constructor(actions: HudActions) {
    // collapsible panels
    document.querySelectorAll<HTMLElement>('[data-collapse]').forEach((h) => {
      h.addEventListener('click', () => h.parentElement?.classList.toggle('collapsed'));
    });

    $<HTMLSelectElement>('seq-select').addEventListener('change', (e) => actions.selectSequence((e.target as HTMLSelectElement).value));
    $('view-seg').querySelectorAll<HTMLButtonElement>('button').forEach((b) =>
      b.addEventListener('click', () => actions.setView(b.dataset.view as ViewName)));
    $('play-btn').addEventListener('click', () => actions.togglePlay());
    $('speed-seg').querySelectorAll<HTMLButtonElement>('button').forEach((b) =>
      b.addEventListener('click', () => actions.setSpeed(Number(b.dataset.speed))));
    $('color-seg').querySelectorAll<HTMLButtonElement>('button').forEach((b) =>
      b.addEventListener('click', () => actions.setColorMode(b.dataset.mode as ColorMode)));
    const range = $<HTMLInputElement>('size-range');
    range.addEventListener('input', () => actions.setPointSize(Number(range.value)));
    $('i-close').addEventListener('click', () => actions.deselect());
    $('i-follow').addEventListener('click', () => actions.followSelected());
    $('help-btn').addEventListener('click', () => this.toggleHelp());

    const tg = $('layer-toggles');
    for (const def of LAYER_DEFS) {
      const b = document.createElement('button');
      b.className = 'toggle';
      b.style.setProperty('--c', def.color);
      b.innerHTML = `<i class="sw"></i><span>${def.label}</span>${def.kbd ? `<kbd>${def.kbd}</kbd>` : ''}`;
      b.addEventListener('click', () => actions.toggleLayer(def.key));
      tg.appendChild(b);
      this.toggles.set(def.key, b);
    }
  }

  toggleHelp(force?: boolean): void {
    $('help').classList.toggle('hidden', force === undefined ? undefined : !force);
  }

  setSequences(list: IndexSequence[], current: string): void {
    const sel = $<HTMLSelectElement>('seq-select');
    sel.replaceChildren(...list.map((s) => {
      const o = document.createElement('option');
      o.value = s.id;
      const gtU = s.unique_vehicles_gt_trackable ?? s.unique_vehicles_gt_visible ?? s.unique_vehicles_gt;
      o.textContent = s.unique_vehicles !== undefined ? `${s.id}  ·  ${s.unique_vehicles} veh${gtU !== undefined ? ` / GT ${gtU}` : ''}` : s.id;
      o.title = `${s.id}: ${s.unique_vehicles ?? '?'} unique vehicles detected` +
        (s.unique_vehicles_gt_trackable !== undefined ? `, ${s.unique_vehicles_gt_trackable} in GT within LiDAR coverage (${s.unique_vehicles_gt ?? '?'} labelled incl. occluded)` : '');
      return o;
    }));
    sel.value = current;
  }

  setView(v: ViewName, canFollow: boolean): void {
    $('view-seg').querySelectorAll<HTMLButtonElement>('button').forEach((b) => {
      b.classList.toggle('on', b.dataset.view === v);
      if (b.dataset.view === 'follow') b.title = canFollow ? 'Follow selected track (3 / F)' : 'Follow a moving track (3 / F)';
    });
    $('s-view').textContent = v === 'bev' ? 'BEV · ortho' : v === 'orbit' ? '3D orbit' : 'follow cam';
  }

  setPlaying(playing: boolean, buffering: boolean, fps: number, speed: number): void {
    $('play-btn').classList.toggle('playing', playing);
    const pill = $('live-pill');
    pill.classList.toggle('playing', playing && !buffering);
    pill.classList.toggle('buffering', playing && buffering);
    $('live-text').textContent = playing ? (buffering ? 'BUFFERING' : `PLAY · ${fps * speed} Hz`) : `PAUSED · ${fps} Hz`;
  }

  setSpeed(s: number): void {
    $('speed-seg').querySelectorAll<HTMLButtonElement>('button').forEach((b) => b.classList.toggle('on', Number(b.dataset.speed) === s));
  }

  setColorMode(m: ColorMode): void {
    $('color-seg').querySelectorAll<HTMLButtonElement>('button').forEach((b) => b.classList.toggle('on', b.dataset.mode === m));
  }

  setPointSize(v: number): void {
    $<HTMLInputElement>('size-range').value = String(v);
    $('size-val').textContent = `${v.toFixed(2)}×`;
  }

  setLayers(l: Layers, hasGt: boolean): void {
    for (const [k, b] of this.toggles) {
      b.classList.toggle('on', l[k]);
      b.disabled = (k === 'gt' || k === 'compare') && !hasGt;
    }
    $('compare-legend').classList.toggle('hidden', !(l.compare && hasGt));
  }

  setCompareStats(tp: number, fp: number, fn: number): void {
    $('cmp-tp').textContent = String(tp);
    $('cmp-fp').textContent = String(fp);
    $('cmp-fn').textContent = String(fn);
  }

  /** Build class rows for the classes that appear in this sequence. */
  setClasses(classes: string[]): void {
    const box = $('c-classes');
    box.replaceChildren();
    this.classRows.clear();
    for (const cls of classes) {
      const row = document.createElement('div');
      row.className = 'class-row';
      row.style.setProperty('--c', CLASS_HEX[cls] ?? CLASS_HEX.other);
      row.innerHTML = `<span><i class="dot" style="background:var(--c)"></i>${cls}</span><span class="bar"><i></i><s></s></span><span class="n"><b>0</b><em></em></span>`;
      box.appendChild(row);
      this.classRows.set(cls, {
        row,
        bar: row.querySelector('.bar i')!,
        gt: row.querySelector('.bar s')!,
        n: row.querySelector('.n b')!,
        ngt: row.querySelector('.n em')!,
      });
    }
  }

  updateCounts(c: CountView, instant = false): void {
    const d = c.det;
    const g = c.gt;
    this.big.set(d.vehicle, instant);
    $('c-vehicles-gt').textContent = g ? String(g.vehicle) : '–';
    const label = gtRefLabel(c.gtRef);
    $('c-gt-label').textContent = label;
    $('c-gt-label').title = GT_REF_HELP[c.gtRef];
    const foot = $('c-foot');
    if (g && c.gtAllVehicles !== null) {
      foot.innerHTML = `<span title="${GT_REF_HELP[c.gtRef]}">${label}: labelled vehicles the LiDARs can see.</span> ` +
        `<span>${c.gtAllVehicles} labelled in frame, ${c.gtAllUnique ?? '–'} in sequence (incl. occluded).</span>`;
      foot.classList.remove('hidden');
    } else foot.classList.add('hidden');
    const delta = $('c-delta');
    if (g) {
      const diff = d.vehicle - g.vehicle;
      delta.textContent = diff === 0 ? '±0 vs GT' : `${diff > 0 ? '+' : ''}${diff} vs GT`;
      delta.className = `delta ${diff === 0 ? 'zero' : diff > 0 ? 'pos' : 'neg'}`;
    } else {
      delta.textContent = '';
    }
    const moving = d.moving ?? 0;
    const parked = d.parked ?? Math.max(0, d.vehicle - moving);
    $('c-moving').textContent = String(moving);
    $('c-parked').textContent = String(parked);
    $('c-moving-gt').textContent = g ? `/${g.moving ?? 0}` : '';
    $('c-parked-gt').textContent = g ? `/${g.parked ?? 0}` : '';
    $('c-moving-bar').style.width = `${d.vehicle ? (moving / d.vehicle) * 100 : 0}%`;
    const total = Math.max(1, d.vehicle, g?.vehicle ?? 0);
    for (const [cls, r] of this.classRows) {
      const n = d.by_class?.[cls] ?? 0;
      const ng = g?.by_class?.[cls];
      r.n.textContent = String(n);
      r.ngt.textContent = g ? `/${ng ?? 0}` : '';
      r.bar.style.width = `${(n / total) * 100}%`;
      r.gt.style.left = `${((ng ?? 0) / total) * 100}%`;
      r.gt.style.display = g ? '' : 'none';
      r.row.classList.toggle('zero', n === 0 && !ng);
    }
    $('c-vru').textContent = String(d.vru);
    $('c-vru-gt').textContent = g ? `/${g.vru}` : '';
    this.uniq.set(c.uniqueSoFar, instant);
    $('u-total').textContent = String(c.uniqueTotal);
    $('u-bar').style.width = `${(c.uniqueSoFar / Math.max(1, c.uniqueTotal)) * 100}%`;
    const ugt = $('u-bar-gt');
    if (c.uniqueSoFarGt !== null && c.uniqueTotalGt !== null) {
      $('u-gt').innerHTML = `<span>${label}</span><span><b>${c.uniqueSoFarGt}</b> / ${c.uniqueTotalGt}</span>`;
      $('u-gt').title = GT_REF_HELP[c.gtRef];
      ugt.style.display = '';
      ugt.style.left = `${Math.min(100, (c.uniqueSoFarGt / Math.max(1, c.uniqueTotal)) * 100)}%`;
    } else {
      $('u-gt').textContent = '';
      ugt.style.display = 'none';
    }
  }

  setMetrics(m: MetricsFile | null): void {
    const body = $('eval-body');
    const panel = $('eval-panel');
    if (!m) {
      panel.classList.add('hidden');
      return;
    }
    panel.classList.remove('hidden');
    const v = m.detection?.vehicle;
    const ap = v?.ap ?? {};
    const apKeys = Object.keys(ap).sort();
    const k03 = apKeys.find((k) => Math.abs(Number(k) - 0.3) < 1e-6) ?? apKeys[0];
    const k05 = apKeys.find((k) => Math.abs(Number(k) - 0.5) < 1e-6) ?? apKeys[1];
    const gauge = (label: string, val: number | undefined) =>
      `<div class="gauge"><span>${label}</span><span class="bar"><i style="width:${val === undefined ? 0 : Math.max(0, Math.min(1, val)) * 100}%"></i></span><b>${pct(val)}</b></div>`;
    const tile = (label: string, val: string, cls = '', unit = '') =>
      `<div class="tile ${cls}"><div class="k">${label}</div><div class="v">${val}${unit ? `<small>${unit}</small>` : ''}</div></div>`;
    let html = '';
    if (v) {
      html += `<div class="eval-grid">
        ${tile('AP @ IoU 0.3', pct(k03 ? ap[k03] : undefined), 'hero', '%')}
        ${tile('AP @ IoU 0.5', pct(k05 ? ap[k05] : undefined), 'hero', '%')}
      </div>
      <div class="gauges">${gauge('Precision', v.precision)}${gauge('Recall', v.recall)}${gauge('F1', v.f1)}</div>`;
    }
    const cRef = m.counting_trackable ?? m.counting_visible ?? m.counting;
    const refName = m.counting_trackable ? 'in coverage' : m.counting_visible ? 'visible' : 'all labels';
    const maeTip = [
      m.counting_trackable?.mae !== undefined ? `vs GT in coverage: ${fmt(m.counting_trackable.mae)}` : '',
      m.counting_visible?.mae !== undefined ? `vs GT visible in frame: ${fmt(m.counting_visible.mae)}` : '',
      m.counting?.mae !== undefined ? `vs all labels (incl. occluded): ${fmt(m.counting.mae)}` : '',
    ].filter(Boolean).join('\n');
    const t = m.tracking;
    if (cRef || t) {
      html += `<div class="eval-row3">
        <div class="tile" title="Mean absolute error of the per-frame vehicle count
${maeTip}"><div class="k">Count MAE</div><div class="v">${fmt(cRef?.mae, 2)}</div><div class="tile-sub">vs GT ${refName}</div></div>
        ${tile('MOTA', t?.mota !== undefined ? pct(t.mota) : '–', '', t?.mota !== undefined ? '%' : '')}
        ${tile('ID sw.', t?.id_switches !== undefined ? String(t.id_switches) : '–')}
      </div>`;
    }
    const pr = m.detection?.pr_curve;
    if (pr && pr.recall?.length > 1) {
      const W = 240;
      const H = 70;
      const pts = pr.recall.map((r, i) => [r * W, 4 + (1 - (pr.precision[i] ?? 0)) * (H - 4)]);
      const line = pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join('');
      const area = `${line}L${pts[pts.length - 1][0].toFixed(1)},${H}L${pts[0][0].toFixed(1)},${H}Z`;
      html += `<div class="pr"><div class="pr-head"><span class="k">PR curve · vehicles</span><span class="gt-muted">${m.detection?.iou_thresholds?.length ? '' : ''}</span></div>
        <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
          <defs><linearGradient id="pr-g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#39e1ff" stop-opacity="0.35"/><stop offset="1" stop-color="#39e1ff" stop-opacity="0"/></linearGradient></defs>
          <path d="M0,${H}L${W},${H}" stroke="rgba(140,170,210,0.18)" />
          <path d="M0,0L0,${H}" stroke="rgba(140,170,210,0.18)" />
          <path d="${area}" fill="url(#pr-g)" />
          <path d="${line}" fill="none" stroke="#39e1ff" stroke-width="1.6" vector-effect="non-scaling-stroke" />
          <text x="${W}" y="${H - 3}" text-anchor="end" fill="#4c5a70" font-size="9" font-family="JetBrains Mono">recall →</text>
          <text x="3" y="9" fill="#4c5a70" font-size="9" font-family="JetBrains Mono">precision</text>
        </svg></div>`;
    }
    body.innerHTML = html || '<div class="eval-empty">No metrics available for this sequence.</div>';
  }

  setSelection(s: SelectionView | null): void {
    const card = $('info-card');
    if (!s) {
      card.classList.add('hidden');
      return;
    }
    card.classList.remove('hidden');
    $('i-title').textContent = `Track #${s.id}`;
    const st = $('i-status');
    st.className = `pill ${!s.present ? 'absent' : s.moving ? 'moving' : 'parked'}`;
    st.textContent = !s.present ? 'not in frame' : s.moving ? 'moving' : 'parked';
    $('i-speed').textContent = s.present ? s.speedKmh.toFixed(1) : '–';
    const cls = $('i-cls');
    cls.style.setProperty('--c', CLASS_HEX[s.cls] ?? CLASS_HEX.other);
    cls.innerHTML = `<i></i>${s.cls}`;
    const rows: [string, string][] = [];
    if (s.dims) rows.push(['size l×w×h', `${s.dims[0].toFixed(2)} × ${s.dims[1].toFixed(2)} × ${s.dims[2].toFixed(2)} m`]);
    if (s.headingDeg !== null) rows.push(['heading', `${s.headingDeg.toFixed(0)}°`]);
    rows.push(['tracked', `${s.nFrames} fr · ${(s.nFrames / s.fps).toFixed(1)} s`]);
    rows.push(['frames', `${s.first} – ${s.last}`]);
    if (s.maxSpeedKmh !== null) rows.push(['max speed', `${s.maxSpeedKmh.toFixed(1)} km/h`]);
    if (s.score !== null) rows.push(['score', s.score.toFixed(2)]);
    if (s.nPoints !== null) rows.push(['points', String(s.nPoints)]);
    if (s.match) rows.push(['vs GT', s.match]);
    if (s.coasted) rows.push(['state', 'coasted (predicted)']);
    $('i-kv').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
    $('i-follow').classList.toggle('on', s.following);
  }

  setFrame(f: number, n: number, t: number): void {
    $('t-frame').textContent = String(f);
    $('t-frames').textContent = String(n);
    $('t-time').textContent = `t = ${t.toFixed(1)} s`;
  }

  setStatus(main: string, buffer: string, fps: string): void {
    $('s-main').innerHTML = main;
    $('s-buffer').textContent = buffer;
    $('s-fps').textContent = fps;
  }

  setCaption(html: string | null): void {
    const cap = $('caption');
    if (html === null) {
      cap.classList.add('hidden');
      return;
    }
    cap.classList.remove('hidden');
    $('cap-line').innerHTML = html;
  }

  setTitle(html: string | null): void {
    const el = $('title-card');
    if (html === null) {
      el.classList.remove('show');
      el.innerHTML = '';
      return;
    }
    if (el.innerHTML !== html) {
      el.innerHTML = html;
      // restart the entrance animation
      el.classList.remove('show');
      void el.offsetWidth;
    }
    el.classList.add('show');
  }

  setUiVisible(v: boolean): void {
    document.body.classList.toggle('ui-hidden', !v);
  }

  splash(text: string | null, error = false): void {
    const s = $('splash');
    if (text === null) {
      s.classList.add('gone');
      return;
    }
    s.classList.remove('gone');
    s.classList.toggle('error', error);
    $('splash-text').innerHTML = text;
  }
}
