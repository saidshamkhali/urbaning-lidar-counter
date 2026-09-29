const NS = 'http://www.w3.org/2000/svg';

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}, parent?: Element): SVGElementTagNameMap[K] {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  parent?.appendChild(e);
  return e;
}

/** Per-frame vehicle count chart that doubles as the frame scrubber. */
export class Timeline {
  private det: number[] = [];
  private gt: number[] | null = null;
  private fps = 10;
  private n = 1;
  private frame = 0;
  private w = 600;
  private h = 70;
  private readonly pad = { l: 24, r: 8, t: 6, b: 15 };
  private yMax = 1;
  private readonly gGrid: SVGGElement;
  private readonly area: SVGPathElement;
  private readonly lineDet: SVGPathElement;
  private readonly lineGt: SVGPathElement;
  private readonly buffered: SVGPathElement;
  private readonly played: SVGRectElement;
  private readonly head: SVGGElement;
  private readonly headLine: SVGLineElement;
  private readonly headDot: SVGCircleElement;
  private readonly headDotGt: SVGCircleElement;
  private readonly hoverLine: SVGLineElement;
  private dragging = false;
  private gtLabel = 'gt';

  constructor(
    private readonly svg: SVGSVGElement,
    private readonly tip: HTMLElement,
    private readonly onSeek: (frame: number) => void,
  ) {
    const defs = el('defs', {}, svg);
    const grad = el('linearGradient', { id: 'tl-area', x1: 0, y1: 0, x2: 0, y2: 1 }, defs);
    el('stop', { offset: '0%', 'stop-color': '#39e1ff', 'stop-opacity': 0.38 }, grad);
    el('stop', { offset: '100%', 'stop-color': '#39e1ff', 'stop-opacity': 0.02 }, grad);
    const glow = el('filter', { id: 'tl-glow', x: '-20%', y: '-50%', width: '140%', height: '200%' }, defs);
    el('feGaussianBlur', { stdDeviation: 2.2, result: 'b' }, glow);
    const merge = el('feMerge', {}, glow);
    el('feMergeNode', { in: 'b' }, merge);
    el('feMergeNode', { in: 'SourceGraphic' }, merge);

    this.gGrid = el('g', { class: 'tl-grid' }, svg);
    this.played = el('rect', { fill: 'rgba(57,225,255,0.035)' }, svg);
    this.area = el('path', { fill: 'url(#tl-area)' }, svg);
    this.lineGt = el('path', { fill: 'none', stroke: 'rgba(223,230,240,0.62)', 'stroke-width': 1.3, 'stroke-dasharray': '4 3' }, svg);
    this.lineDet = el('path', { fill: 'none', stroke: '#39e1ff', 'stroke-width': 1.7, filter: 'url(#tl-glow)', 'stroke-linejoin': 'round' }, svg);
    this.buffered = el('path', { stroke: 'rgba(57,225,255,0.45)', 'stroke-width': 2, fill: 'none' }, svg);
    this.hoverLine = el('line', { stroke: 'rgba(255,255,255,0.25)', 'stroke-width': 1, visibility: 'hidden' }, svg);
    this.head = el('g', {}, svg);
    this.headLine = el('line', { stroke: '#ffffff', 'stroke-width': 1.3, opacity: 0.9 }, this.head);
    this.headDotGt = el('circle', { r: 3, fill: '#04060a', stroke: 'rgba(223,230,240,0.85)', 'stroke-width': 1.3 }, this.head);
    this.headDot = el('circle', { r: 4, fill: '#39e1ff', stroke: '#e9fbff', 'stroke-width': 1.2, filter: 'url(#tl-glow)' }, this.head);

    new ResizeObserver(() => this.layout()).observe(svg);

    svg.addEventListener('pointerdown', (e) => {
      this.dragging = true;
      svg.setPointerCapture(e.pointerId);
      this.seekAt(e.clientX);
    });
    svg.addEventListener('pointermove', (e) => {
      this.hoverAt(e.clientX);
      if (this.dragging) this.seekAt(e.clientX);
    });
    const end = (e: PointerEvent) => {
      this.dragging = false;
      if (svg.hasPointerCapture(e.pointerId)) svg.releasePointerCapture(e.pointerId);
    };
    svg.addEventListener('pointerup', end);
    svg.addEventListener('pointercancel', end);
    svg.addEventListener('pointerleave', () => {
      this.tip.classList.add('hidden');
      this.hoverLine.setAttribute('visibility', 'hidden');
    });
  }

  setGtLabel(label: string): void {
    this.gtLabel = label;
    const el = document.getElementById('chart-gt-label');
    if (el) el.textContent = label;
  }

  setData(det: number[], gt: number[] | null, fps: number, n: number): void {
    this.det = det;
    this.gt = gt;
    this.fps = fps;
    this.n = Math.max(1, n);
    const m = Math.max(1, ...det, ...(gt ?? [0]));
    this.yMax = Math.ceil((m * 1.12) / 5) * 5;
    this.layout();
  }

  private x(f: number): number {
    return this.pad.l + (f / Math.max(1, this.n - 1)) * (this.w - this.pad.l - this.pad.r);
  }

  private y(v: number): number {
    return this.pad.t + (1 - v / this.yMax) * (this.h - this.pad.t - this.pad.b);
  }

  private frameAt(clientX: number): number {
    const r = this.svg.getBoundingClientRect();
    const t = (clientX - r.left - this.pad.l) / (this.w - this.pad.l - this.pad.r);
    return Math.max(0, Math.min(this.n - 1, Math.round(t * (this.n - 1))));
  }

  private seekAt(clientX: number): void {
    this.onSeek(this.frameAt(clientX));
  }

  private hoverAt(clientX: number): void {
    const f = this.frameAt(clientX);
    const x = this.x(f);
    this.hoverLine.setAttribute('x1', String(x));
    this.hoverLine.setAttribute('x2', String(x));
    this.hoverLine.setAttribute('y1', String(this.pad.t));
    this.hoverLine.setAttribute('y2', String(this.h - this.pad.b));
    this.hoverLine.setAttribute('visibility', 'visible');
    const d = this.det[f] ?? 0;
    const g = this.gt?.[f];
    this.tip.innerHTML = `f ${f} <em>·</em> ${(f / this.fps).toFixed(1)} s <em>·</em> det <b>${d}</b>${g !== undefined ? ` <em>· ${this.gtLabel} ${g}</em>` : ''}`;
    const parent = this.svg.parentElement;
    const off = parent ? this.svg.getBoundingClientRect().left - parent.getBoundingClientRect().left : 0;
    this.tip.style.left = `${off + x}px`;
    this.tip.classList.remove('hidden');
  }

  private pathFor(vals: number[]): string {
    let d = '';
    for (let i = 0; i < vals.length; i++) d += `${i ? 'L' : 'M'}${this.x(i).toFixed(1)},${this.y(vals[i]).toFixed(1)}`;
    return d;
  }

  layout(): void {
    const r = this.svg.getBoundingClientRect();
    this.w = Math.max(50, r.width);
    this.h = Math.max(30, r.height);
    this.svg.setAttribute('viewBox', `0 0 ${this.w} ${this.h}`);

    // grid + axes
    this.gGrid.replaceChildren();
    const base = this.h - this.pad.b;
    const steps = this.yMax <= 10 ? [0, 5, 10] : [0, Math.round(this.yMax / 2), this.yMax];
    for (const v of steps) {
      const y = this.y(v);
      el('line', { x1: this.pad.l, x2: this.w - this.pad.r, y1: y, y2: y, stroke: 'rgba(140,170,210,0.08)' }, this.gGrid);
      const t = el('text', { x: this.pad.l - 5, y: y + 3, 'text-anchor': 'end', fill: '#4c5a70', 'font-size': 9, 'font-family': 'JetBrains Mono, monospace' }, this.gGrid);
      t.textContent = String(v);
    }
    const dur = (this.n - 1) / this.fps;
    const tickS = dur > 60 ? 10 : dur > 24 ? 5 : 2;
    for (let s = 0; s <= dur + 1e-6; s += tickS) {
      const x = this.x(s * this.fps);
      el('line', { x1: x, x2: x, y1: base, y2: base + 3, stroke: 'rgba(140,170,210,0.25)' }, this.gGrid);
      const t = el('text', { x, y: this.h - 2, 'text-anchor': s === 0 ? 'start' : 'middle', fill: '#4c5a70', 'font-size': 9, 'font-family': 'JetBrains Mono, monospace' }, this.gGrid);
      t.textContent = `${s}s`;
    }

    if (this.det.length) {
      const top = this.pathFor(this.det);
      this.lineDet.setAttribute('d', top);
      this.area.setAttribute('d', `${top}L${this.x(this.det.length - 1)},${base}L${this.x(0)},${base}Z`);
    } else {
      this.lineDet.setAttribute('d', '');
      this.area.setAttribute('d', '');
    }
    this.lineGt.setAttribute('d', this.gt ? this.pathFor(this.gt) : '');
    this.setFrame(this.frame);
  }

  setFrame(f: number): void {
    this.frame = f;
    const x = this.x(f);
    this.headLine.setAttribute('x1', String(x));
    this.headLine.setAttribute('x2', String(x));
    this.headLine.setAttribute('y1', String(this.pad.t - 2));
    this.headLine.setAttribute('y2', String(this.h - this.pad.b));
    this.headDot.setAttribute('cx', String(x));
    this.headDot.setAttribute('cy', String(this.y(this.det[f] ?? 0)));
    const g = this.gt?.[f];
    this.headDotGt.setAttribute('visibility', g === undefined ? 'hidden' : 'visible');
    this.headDotGt.setAttribute('cx', String(x));
    this.headDotGt.setAttribute('cy', String(this.y(g ?? 0)));
    this.played.setAttribute('x', String(this.pad.l));
    this.played.setAttribute('y', String(this.pad.t));
    this.played.setAttribute('width', String(Math.max(0, x - this.pad.l)));
    this.played.setAttribute('height', String(Math.max(0, this.h - this.pad.b - this.pad.t)));
  }

  /** Thin strip along the baseline marking frames whose points are cached. */
  setBuffered(has: (f: number) => boolean): void {
    const y = this.h - this.pad.b + 0.5;
    let d = '';
    let start = -1;
    for (let f = 0; f <= this.n; f++) {
      const ok = f < this.n && has(f);
      if (ok && start < 0) start = f;
      if (!ok && start >= 0) {
        d += `M${this.x(start).toFixed(1)},${y}L${this.x(Math.min(this.n - 1, f - 1) + 0.9).toFixed(1)},${y}`;
        start = -1;
      }
    }
    this.buffered.setAttribute('d', d);
  }
}
