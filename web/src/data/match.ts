import type { Detection } from './types';
import { groupOf } from './types';

export interface FrameMatch {
  /** per detection index: matched GT index or -1 (false positive) */
  detToGt: Int32Array;
  /** per GT index: matched detection index or -1 (miss) */
  gtToDet: Int32Array;
  tp: number;
  fp: number;
  /** unmatched GT that count as misses (visible, in coverage) */
  fn: number;
  /** per GT index: unmatched but not a miss (occluded / out of coverage) */
  gtExcused: Uint8Array;
}

/** GT that must be detected: visible in this frame and in sensor coverage. */
export const isMissable = (g: Detection) => g.ignore !== true && g.trackable !== false;

/**
 * Greedy BEV centre-distance matching within the same class group. Detections may match any GT
 * (also occluded ones, so they are not false positives); only visible GT can become misses.
 */
export function matchFrame(dets: Detection[], gts: Detection[]): FrameMatch {
  const pairs: { d: number; g: number; dist: number }[] = [];
  for (let i = 0; i < dets.length; i++) {
    const a = dets[i];
    const ga = groupOf(a.cls);
    for (let j = 0; j < gts.length; j++) {
      const b = gts[j];
      if (groupOf(b.cls) !== ga) continue;
      const dist = Math.hypot(a.box[0] - b.box[0], a.box[1] - b.box[1]);
      const thr = ga === 'vehicle' ? Math.max(1.5, Math.min(3, b.box[3] * 0.4)) : 1.0;
      if (dist < thr) pairs.push({ d: i, g: j, dist });
    }
  }
  pairs.sort((p, q) => p.dist - q.dist);
  const detToGt = new Int32Array(dets.length).fill(-1);
  const gtToDet = new Int32Array(gts.length).fill(-1);
  let tp = 0;
  for (const p of pairs) {
    if (detToGt[p.d] >= 0 || gtToDet[p.g] >= 0) continue;
    detToGt[p.d] = p.g;
    gtToDet[p.g] = p.d;
    tp++;
  }
  const gtExcused = new Uint8Array(gts.length);
  let fn = 0;
  for (let j = 0; j < gts.length; j++) {
    if (gtToDet[j] >= 0) continue;
    if (isMissable(gts[j])) fn++;
    else gtExcused[j] = 1;
  }
  return { detToGt, gtToDet, tp, fp: dets.length - tp, fn, gtExcused };
}

export class MatchCache {
  private readonly cache = new Map<number, FrameMatch>();
  constructor(private readonly det: Detection[][], private readonly gt: Detection[][]) {}

  get(frame: number): FrameMatch {
    let m = this.cache.get(frame);
    if (!m) {
      m = matchFrame(this.det[frame] ?? [], this.gt[frame] ?? []);
      this.cache.set(frame, m);
    }
    return m;
  }
}
