import type {
  Detection, DetectionsFile, FrameCounts, IndexFile, MapFile, Meta, MetricsFile, TrackInfo,
} from './types';
import { groupOf } from './types';

const DATA_ROOT = './data/';

async function fetchJson<T>(path: string, optional = false): Promise<T | null> {
  try {
    const res = await fetch(DATA_ROOT + path, { cache: 'no-cache' });
    if (!res.ok) {
      if (optional) return null;
      throw new Error(`${res.status} ${res.statusText}`);
    }
    return (await res.json()) as T;
  } catch (err) {
    if (optional) return null;
    throw new Error(`Failed to load ${path}: ${(err as Error).message}`);
  }
}

export function loadIndex(): Promise<IndexFile | null> {
  return fetchJson<IndexFile>('index.json');
}

const mapCache = new Map<string, Promise<MapFile | null>>();
export function loadMap(name: string): Promise<MapFile | null> {
  let p = mapCache.get(name);
  if (!p) {
    p = fetchJson<MapFile>(name, true);
    mapCache.set(name, p);
  }
  return p;
}

/** Per-track centre positions indexed by frame (NaN where absent). */
export interface TrackPath {
  id: number;
  x: Float32Array;
  y: Float32Array;
  z: Float32Array;
  yaw: Float32Array;
}

/**
 * Which GT subset is used as the counting reference:
 * `trackable` = vehicles in sensor coverage, `visible` = with LiDAR points in the frame, `all` = every label.
 */
export type RefKind = 'trackable' | 'visible' | 'all';

/** A detections/gt file plus derived lookups. Missing optional fields are reconstructed. */
export class Labels {
  readonly frames: Detection[][];
  readonly tracks: Map<number, TrackInfo>;
  readonly counts: FrameCounts[];
  readonly paths = new Map<number, TrackPath>();
  /** sorted `first` frames of vehicle tracks, for cumulative unique counts */
  private readonly vehicleFirsts: number[];
  private readonly vruFirsts: number[];
  readonly uniqueVehicles: number;
  readonly uniqueVru: number;
  /** reference subset for counting comparisons (GT only; `all` for detections) */
  readonly refKind: RefKind;
  readonly refCounts: FrameCounts[];
  readonly uniqueRefVehicles: number;
  private readonly refVehicleFirsts: number[];

  constructor(file: DetectionsFile, nFrames: number) {
    const frames: Detection[][] = [];
    for (let f = 0; f < nFrames; f++) frames.push(file.frames?.[f] ?? []);
    this.frames = frames;

    // tracks
    this.tracks = new Map();
    if (file.tracks) {
      for (const [k, v] of Object.entries(file.tracks)) this.tracks.set(Number(k), v);
    }
    const derived = new Map<number, TrackInfo>();
    frames.forEach((dets, f) => {
      for (const d of dets) {
        let t = derived.get(d.id);
        if (!t) {
          t = { cls: d.cls, group: groupOf(d.cls), first: f, last: f, moving: false, max_speed: 0, n_frames: 0 };
          derived.set(d.id, t);
        }
        t.last = f;
        t.n_frames = (t.n_frames ?? 0) + 1;
        t.max_speed = Math.max(t.max_speed ?? 0, d.speed ?? 0);
        if (d.moving) t.moving = true;
      }
    });
    for (const [id, t] of derived) if (!this.tracks.has(id)) this.tracks.set(id, t);
    for (const t of this.tracks.values()) if (!t.group) t.group = groupOf(t.cls);

    // counts
    if (file.counts && file.counts.length >= nFrames) {
      this.counts = file.counts;
    } else {
      this.counts = frames.map((dets) => {
        const c: FrameCounts = { vehicle: 0, vru: 0, other: 0, moving: 0, parked: 0, by_class: {} };
        for (const d of dets) {
          const g = groupOf(d.cls);
          if (g === 'vehicle') {
            c.vehicle++;
            if (d.moving) c.moving = (c.moving ?? 0) + 1; else c.parked = (c.parked ?? 0) + 1;
            c.by_class![d.cls] = (c.by_class![d.cls] ?? 0) + 1;
          } else if (g === 'vru') c.vru++;
          else c.other = (c.other ?? 0) + 1;
        }
        return c;
      });
    }

    // paths
    for (let f = 0; f < nFrames; f++) {
      for (const d of frames[f]) {
        let p = this.paths.get(d.id);
        if (!p) {
          p = {
            id: d.id,
            x: new Float32Array(nFrames).fill(NaN),
            y: new Float32Array(nFrames).fill(NaN),
            z: new Float32Array(nFrames).fill(NaN),
            yaw: new Float32Array(nFrames).fill(NaN),
          };
          this.paths.set(d.id, p);
        }
        p.x[f] = d.box[0];
        p.y[f] = d.box[1];
        p.z[f] = d.box[2] - d.box[5] / 2;
        p.yaw[f] = d.box[6];
      }
    }

    const vf: number[] = [];
    const rf: number[] = [];
    for (const t of this.tracks.values()) {
      if (t.group === 'vehicle') vf.push(t.first);
      else if (t.group === 'vru') rf.push(t.first);
    }
    this.vehicleFirsts = vf.sort((a, b) => a - b);
    this.vruFirsts = rf.sort((a, b) => a - b);
    this.uniqueVehicles = file.unique?.vehicle ?? vf.length;
    this.uniqueVru = file.unique?.vru ?? rf.length;

    // counting reference
    const hasTrackable = (file.counts_trackable?.length ?? 0) >= nFrames;
    const hasVisible = (file.counts_visible?.length ?? 0) >= nFrames;
    this.refKind = hasTrackable ? 'trackable' : hasVisible ? 'visible' : 'all';
    this.refCounts = hasTrackable ? file.counts_trackable! : hasVisible ? file.counts_visible! : this.counts;
    if (this.refKind === 'all') {
      this.refVehicleFirsts = this.vehicleFirsts;
      this.uniqueRefVehicles = this.uniqueVehicles;
    } else {
      const firsts = new Map<number, number>();
      frames.forEach((dets, f) => {
        for (const d of dets) {
          if (groupOf(d.cls) !== 'vehicle' || firsts.has(d.id)) continue;
          const ok = this.refKind === 'trackable' ? d.trackable !== false : d.ignore !== true;
          if (ok) firsts.set(d.id, f);
        }
      });
      this.refVehicleFirsts = [...firsts.values()].sort((a, b) => a - b);
      const u = this.refKind === 'trackable' ? file.unique_trackable : file.unique_visible;
      this.uniqueRefVehicles = u?.vehicle ?? this.refVehicleFirsts.length;
    }
  }

  /** unique reference vehicles whose first (reference) frame is <= f */
  cumulativeRef(f: number): number {
    return countLE(this.refVehicleFirsts, f);
  }

  /** whether a GT object belongs to the counting reference */
  isRef(d: Detection): boolean {
    if (this.refKind === 'trackable') return d.trackable !== false;
    if (this.refKind === 'visible') return d.ignore !== true;
    return true;
  }

  /** unique tracks of a group whose first frame is <= f */
  cumulative(group: 'vehicle' | 'vru', f: number): number {
    return countLE(group === 'vehicle' ? this.vehicleFirsts : this.vruFirsts, f);
  }

  find(f: number, id: number): Detection | undefined {
    return this.frames[f]?.find((d) => d.id === id);
  }
}

function countLE(arr: number[], f: number): number {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] <= f) lo = mid + 1; else hi = mid;
  }
  return lo;
}

export interface Sequence {
  id: string;
  meta: Meta;
  det: Labels;
  gt: Labels | null;
  metrics: MetricsFile | null;
  map: MapFile | null;
  maxPoints: number;
}

export async function loadSequence(id: string): Promise<Sequence> {
  const meta = await fetchJson<Meta>(`${id}/meta.json`);
  if (!meta) throw new Error(`meta.json missing for ${id}`);
  const nFrames = meta.n_frames ?? meta.frames.length;
  const [det, gt, metrics, map] = await Promise.all([
    fetchJson<DetectionsFile>(`${id}/detections.json`, true),
    fetchJson<DetectionsFile>(`${id}/gt.json`, true),
    fetchJson<MetricsFile>(`${id}/metrics.json`, true),
    loadMap(meta.map ?? `map_${meta.crossing}.json`),
  ]);
  let maxPoints = 0;
  for (const fr of meta.frames) maxPoints = Math.max(maxPoints, fr.n_points);
  return {
    id,
    meta,
    det: new Labels(det ?? { frames: [] }, nFrames),
    gt: gt ? new Labels(gt, nFrames) : null,
    metrics,
    map,
    maxPoints,
  };
}

// ---------------------------------------------------------------------------------------------
// Point cloud cache: LRU + prefetch window + limited concurrency

interface Pending {
  frame: number;
  urgent: boolean;
  resolve: (b: ArrayBuffer) => void;
  reject: (e: unknown) => void;
  promise: Promise<ArrayBuffer>;
}

export class PointCache {
  private readonly cache = new Map<number, ArrayBuffer>();
  private readonly inflight = new Map<number, Promise<ArrayBuffer>>();
  private queue: Pending[] = [];
  private active = 0;
  private readonly abort = new AbortController();
  private window = new Set<number>();
  onLoaded: ((frame: number) => void) | null = null;

  constructor(
    private readonly seqId: string,
    private readonly meta: Meta,
    private readonly capacity = 80,
    private readonly prefetchAhead = 15,
    private readonly concurrency = 6,
  ) {}

  get size(): number {
    return this.cache.size;
  }

  has(frame: number): boolean {
    return this.cache.has(frame);
  }

  peek(frame: number): ArrayBuffer | undefined {
    const b = this.cache.get(frame);
    if (b) {
      this.cache.delete(frame);
      this.cache.set(frame, b);
    }
    return b;
  }

  get(frame: number): Promise<ArrayBuffer> {
    const b = this.peek(frame);
    if (b) return Promise.resolve(b);
    const inf = this.inflight.get(frame);
    if (inf) return inf;
    const queued = this.queue.find((q) => q.frame === frame);
    if (queued) {
      queued.urgent = true;
      this.queue = [queued, ...this.queue.filter((q) => q !== queued)];
      return queued.promise;
    }
    return this.enqueue(frame, true);
  }

  /** Queue the next frames after `frame` (wrapping) and drop stale queued prefetches. */
  prefetch(frame: number): void {
    const n = this.meta.frames.length;
    const want: number[] = [];
    for (let k = 0; k <= this.prefetchAhead; k++) want.push((frame + k) % n);
    for (let k = 1; k <= 2; k++) if (frame - k >= 0) want.push(frame - k);
    this.window = new Set(want);
    this.queue = this.queue.filter((q) => {
      if (q.urgent || this.window.has(q.frame)) return true;
      q.reject(new DOMException('dropped', 'AbortError'));
      return false;
    });
    for (const f of want) {
      if (this.cache.has(f) || this.inflight.has(f) || this.queue.some((q) => q.frame === f)) continue;
      this.enqueue(f, false).catch(() => undefined);
    }
  }

  dispose(): void {
    this.abort.abort();
    for (const q of this.queue) q.reject(new DOMException('disposed', 'AbortError'));
    this.queue = [];
    this.cache.clear();
  }

  private enqueue(frame: number, urgent: boolean): Promise<ArrayBuffer> {
    let resolve!: (b: ArrayBuffer) => void;
    let reject!: (e: unknown) => void;
    const promise = new Promise<ArrayBuffer>((res, rej) => {
      resolve = res;
      reject = rej;
    });
    const item: Pending = { frame, urgent, resolve, reject, promise };
    if (urgent) this.queue.unshift(item); else this.queue.push(item);
    this.pump();
    return promise;
  }

  private pump(): void {
    while (this.active < this.concurrency && this.queue.length) {
      const item = this.queue.shift()!;
      this.active++;
      const p = this.fetchFrame(item.frame);
      this.inflight.set(item.frame, p);
      p.then(
        (buf) => {
          this.store(item.frame, buf);
          item.resolve(buf);
          this.onLoaded?.(item.frame);
        },
        (err) => item.reject(err),
      ).finally(() => {
        this.inflight.delete(item.frame);
        this.active--;
        this.pump();
      });
    }
  }

  private async fetchFrame(frame: number): Promise<ArrayBuffer> {
    const fr = this.meta.frames[frame];
    if (!fr) throw new Error(`frame ${frame} out of range`);
    const res = await fetch(`${DATA_ROOT}${this.seqId}/${fr.file}`, { signal: this.abort.signal });
    if (!res.ok) throw new Error(`points ${frame}: ${res.status}`);
    return res.arrayBuffer();
  }

  private store(frame: number, buf: ArrayBuffer): void {
    this.cache.set(frame, buf);
    if (this.cache.size <= this.capacity) return;
    for (const key of this.cache.keys()) {
      if (this.cache.size <= this.capacity) break;
      if (this.window.has(key)) continue;
      this.cache.delete(key);
    }
  }
}
