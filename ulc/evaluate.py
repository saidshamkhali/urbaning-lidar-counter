"""Detection, counting and tracking evaluation (metrics.json in ``docs/DATA_FORMAT.md``).

Usage::

    python -m ulc.evaluate --pred path/detections.json --gt path/gt.json [--out metrics.json]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import bev_iou_matrix, iou_3d_matrix
from .schema import FINE_CLASSES, GROUP_OF, build_tracks, normalize_class, summarize

__all__ = ["PR_IOU", "TRACK_IOU", "mark_visibility", "evaluate_sequence", "evaluate_many", "main"]

#: IoU threshold for the precision/recall/F1 operating point, PR curve and range breakdown.
PR_IOU = 0.3
#: BEV IoU threshold for CLEAR-MOT matching.
TRACK_IOU = 0.3
_MAX_PR_POINTS = 101

RoiFn = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------- helpers


def _num(x: float | None) -> float | None:
    """Plain float, or None for missing / non-finite values."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _ratio(a: float, b: float) -> float | None:
    return float(a) / float(b) if b else None


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None:
        return None
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def _thr_key(t: float) -> str:
    return f"{t:g}"


@dataclass
class _Frame:
    """Objects of one frame as arrays."""

    boxes: np.ndarray  # (N, 7)
    cls: np.ndarray  # (N,) fine class names
    score: np.ndarray  # (N,)
    ids: list  # (N,) track ids (hashable)
    ignore: np.ndarray  # (N,) bool, "don't care" GT objects

    @property
    def vehicle(self) -> np.ndarray:
        return np.array([GROUP_OF[c] == "vehicle" for c in self.cls], dtype=bool)

    def subset(self, mask: np.ndarray) -> _Frame:
        return _Frame(self.boxes[mask], self.cls[mask], self.score[mask],
                      [i for i, m in zip(self.ids, mask) if m], self.ignore[mask])


def _to_frames(result: dict, n_frames: int, roi: RoiFn | None, tag: str, use_ignore: bool) -> list[_Frame]:
    frames = list(result.get("frames", []))
    frames += [[] for _ in range(n_frames - len(frames))]
    out = []
    for f, objects in enumerate(frames):
        boxes = np.array([o["box"] for o in objects], dtype=np.float64).reshape(-1, 7)
        fr = _Frame(
            boxes=boxes,
            cls=np.array([normalize_class(o["cls"]) for o in objects], dtype=object),
            score=np.array([float(o.get("score", 1.0)) for o in objects], dtype=np.float64),
            ids=[o["id"] if o.get("id") is not None else f"{tag}:{f}:{j}" for j, o in enumerate(objects)],
            ignore=np.array([use_ignore and bool(o.get("ignore", False)) for o in objects], dtype=bool),
        )
        if roi is not None and len(boxes):
            fr = fr.subset(np.asarray(roi(boxes[:, :2]), dtype=bool))
        out.append(fr)
    return out


def _greedy_match(
    iou: np.ndarray, score: np.ndarray, thr: float, gt_ignore: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy matching in descending score order; each GT is matched at most once.

    A prediction takes the best unmatched regular GT with IoU >= ``thr`` (TP). Failing that, if
    it overlaps an unmatched *ignored* GT at the threshold it is absorbed by it and dropped from
    the evaluation (neither TP nor FP). Otherwise it is a FP.

    Returns ``(pred_tp (N,), pred_ignored (N,), gt_matched (M,))`` boolean arrays.
    """
    n, m = iou.shape
    tp = np.zeros(n, dtype=bool)
    dropped = np.zeros(n, dtype=bool)
    gt_used = np.zeros(m, dtype=bool)
    if n == 0 or m == 0:
        return tp, dropped, gt_used
    gt_ignore = np.zeros(m, dtype=bool) if gt_ignore is None else gt_ignore
    for i in np.argsort(-score, kind="stable"):
        free = ~gt_used & (iou[i] >= thr)
        for pool, flag in ((free & ~gt_ignore, tp), (free & gt_ignore, dropped)):
            if pool.any():
                j = int(np.argmax(np.where(pool, iou[i], -1.0)))
                gt_used[j] = flag[i] = True
                break
    return tp, dropped, gt_used


def _average_precision(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> tuple[float | None, np.ndarray, np.ndarray]:
    """All-point interpolated AP plus the raw (recall, precision) curve."""
    if n_gt == 0:
        return None, np.zeros(0), np.zeros(0)
    if len(scores) == 0:
        return 0.0, np.zeros(0), np.zeros(0)
    order = np.argsort(-scores, kind="stable")
    ctp = np.cumsum(tp[order])
    cfp = np.cumsum(~tp[order])
    recall = ctp / n_gt
    precision = ctp / (ctp + cfp)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    step = np.flatnonzero(mrec[1:] != mrec[:-1])
    ap = float(np.sum((mrec[step + 1] - mrec[step]) * mpre[step + 1]))
    return ap, recall, precision


def _downsample(recall: np.ndarray, precision: np.ndarray, n: int = _MAX_PR_POINTS) -> dict:
    if len(recall) > n:
        idx = np.unique(np.linspace(0, len(recall) - 1, n).round().astype(int))
        recall, precision = recall[idx], precision[idx]
    return {"recall": [round(float(r), 5) for r in recall], "precision": [round(float(p), 5) for p in precision]}


def _range_label(lo: float, hi: float) -> str:
    return f"{lo:g}+" if hi >= 1e8 else f"{lo:g}-{hi:g}"


def _count_stats(pred: Sequence[int], gt: Sequence[int]) -> dict:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(gt, dtype=np.float64)
    if len(err) == 0:
        return {"mae": None, "rmse": None, "bias": None, "within_1": None}
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "within_1": float(np.mean(np.abs(err) <= 1)),
    }


def _unique_vehicles(frames: list[_Frame]) -> int:
    objects = [[{"id": i, "cls": c} for i, c in zip(fr.ids, fr.cls)] for fr in frames]
    return sum(1 for t in build_tracks(objects).values() if t["group"] == "vehicle")


# --------------------------------------------------------------------------- detection


def _evaluate_detection(
    pred: list[_Frame],
    gt: list[_Frame],
    iou_thresholds: Sequence[float],
    center: Sequence[float],
    range_bins: Sequence[float],
    iou_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> dict:
    thresholds = sorted(set(map(float, iou_thresholds)) | {PR_IOU})
    classes = [c for c in FINE_CLASSES if any((fr.cls == c).any() for fr in pred + gt)]
    center = np.asarray(center, dtype=np.float64)[:2]
    bins = np.asarray(range_bins, dtype=np.float64)
    n_bins = len(bins) - 1
    rb = {k: np.zeros(n_bins, dtype=int) for k in ("n_gt", "gt_matched", "n_pred", "tp")}

    # (scores, tp) of the non-dropped predictions, per group/class and threshold
    ranked: dict[str, dict[float, list[tuple[np.ndarray, np.ndarray]]]] = {
        key: {t: [] for t in thresholds} for key in ["vehicle", *classes]
    }
    n_gt = dict.fromkeys(["vehicle", *classes], 0)

    def bin_of(boxes: np.ndarray) -> np.ndarray:
        return np.digitize(np.linalg.norm(boxes[:, :2] - center, axis=1), bins) - 1

    for p, g in zip(pred, gt):
        iou = iou_fn(p.boxes, g.boxes)
        selections = [("vehicle", p.vehicle, g.vehicle)] + [(c, p.cls == c, g.cls == c) for c in classes]
        for key, pm, gm in selections:
            n_gt[key] += int((gm & ~g.ignore).sum())
            sub, score, ign = iou[np.ix_(pm, gm)], p.score[pm], g.ignore[gm]
            for t in thresholds:
                tp, dropped, gt_matched = _greedy_match(sub, score, t, ign)
                ranked[key][t].append((score[~dropped], tp[~dropped]))
                if key == "vehicle" and t == PR_IOU and n_bins > 0:
                    bp, bg = bin_of(p.boxes[pm]), bin_of(g.boxes[gm])
                    ok_p = (bp >= 0) & (bp < n_bins) & ~dropped
                    ok_g = (bg >= 0) & (bg < n_bins) & ~ign
                    rb["n_pred"] += np.bincount(bp[ok_p], minlength=n_bins)
                    rb["tp"] += np.bincount(bp[ok_p & tp], minlength=n_bins)
                    rb["n_gt"] += np.bincount(bg[ok_g], minlength=n_bins)
                    rb["gt_matched"] += np.bincount(bg[ok_g & gt_matched], minlength=n_bins)

    def ap_of(key: str, t: float) -> tuple[float | None, np.ndarray, np.ndarray, int, int]:
        parts = ranked[key][t]
        scores = np.concatenate([s for s, _ in parts]) if parts else np.zeros(0)
        tp = np.concatenate([x for _, x in parts]).astype(bool) if parts else np.zeros(0, dtype=bool)
        ap, rec, prec = _average_precision(scores, tp, n_gt[key])
        return ap, rec, prec, int(tp.sum()), len(scores)

    vehicle: dict = {"ap": {_thr_key(t): _num(ap_of("vehicle", float(t))[0]) for t in iou_thresholds}}
    _, rec, prec, n_tp, n_pred = ap_of("vehicle", PR_IOU)
    precision, recall = _ratio(n_tp, n_pred), _ratio(n_tp, n_gt["vehicle"])
    vehicle.update(precision=precision, recall=recall, f1=_f1(precision, recall),
                   n_gt=n_gt["vehicle"], n_pred=n_pred, tp=n_tp)

    per_class = {
        c: {
            "ap": {_thr_key(t): _num(ap_of(c, float(t))[0]) for t in iou_thresholds},
            "n_gt": n_gt[c],
            "n_pred": ap_of(c, PR_IOU)[4],
        }
        for c in classes
    }
    by_range = [
        {
            "range": _range_label(bins[k], bins[k + 1]),
            "recall": _ratio(rb["gt_matched"][k], rb["n_gt"][k]),
            "precision": _ratio(rb["tp"][k], rb["n_pred"][k]),
            **{key: int(rb[key][k]) for key in ("n_gt", "gt_matched", "n_pred", "tp")},
        }
        for k in range(n_bins)
    ]
    return {
        "iou_thresholds": [float(t) for t in iou_thresholds],
        "vehicle": vehicle,
        "per_class": per_class,
        "by_range": by_range,
        "pr_curve": _downsample(rec, prec),
    }


# --------------------------------------------------------------------------- counting


def _counting(pred: dict, gt: dict, p_frames: list[_Frame], g_frames: list[_Frame], key: str, stored: bool) -> dict:
    """Per-frame and unique vehicle counting errors against ``gt[key]`` (``counts`` or ``counts_visible``).

    Stored ``counts`` / ``unique`` blocks are used when ``stored`` and present; otherwise the
    numbers are recomputed from the frames (only non-ignored GT for the ``*_visible`` keys).
    """
    unique_key = key.replace("counts", "unique", 1)
    if key != "counts":
        g_frames = [fr.subset(~fr.ignore) for fr in g_frames]

    def per_frame(counts: list[dict] | None, frames: list[_Frame]) -> list[int]:
        if stored and counts is not None and len(counts) == len(frames) and all("vehicle" in c for c in counts):
            return [int(c["vehicle"]) for c in counts]
        return [int(fr.vehicle.sum()) for fr in frames]

    def unique(block: dict | None, frames: list[_Frame]) -> int:
        if stored and block is not None and "vehicle" in block:
            return int(block["vehicle"])
        return _unique_vehicles(frames)

    pred_counts = per_frame(pred.get("counts"), p_frames)
    gt_counts = per_frame(gt.get(key), g_frames)
    return {
        **_count_stats(pred_counts, gt_counts),
        "pred": pred_counts,
        "gt": gt_counts,
        "unique_pred": unique(pred.get("unique"), p_frames),
        "unique_gt": unique(gt.get(unique_key), g_frames),
    }


# --------------------------------------------------------------------------- tracking


def _evaluate_tracking(pred: list[_Frame], gt: list[_Frame], thr: float = TRACK_IOU) -> dict:
    """CLEAR-MOT (vehicle group, BEV IoU) with preference for keeping previous correspondences.

    Ignored GT take part in the assignment (after regular GT) but never count: unmatched they
    are no miss, predictions assigned to them are no FP, and they cause no ID switches.
    ``unique_gt`` counts GT vehicle tracks that are not ignored in at least one frame.
    """
    last_match: dict = {}  # gt id -> pred id of its most recent match
    n_gt = n_fp = n_miss = n_switch = n_match = 0
    iou_sum = 0.0
    visible_ids: set = set()

    for p_all, g_all in zip(pred, gt):
        p, g = p_all.subset(p_all.vehicle), g_all.subset(g_all.vehicle)
        iou = bev_iou_matrix(p.boxes, g.boxes)
        pred_col = {pid: i for i, pid in enumerate(p.ids)}
        matches: list[tuple[int, int]] = []  # (pred idx, gt idx)
        used_p: set[int] = set()
        used_g: set[int] = set()

        # 1) keep correspondences from the previous match if still valid
        for j, gid in enumerate(g.ids):
            i = pred_col.get(last_match.get(gid))
            if i is not None and i not in used_p and iou[i, j] >= thr:
                matches.append((i, j))
                used_p.add(i)
                used_g.add(j)

        # 2) Hungarian assignment of the rest: regular GT first, then ignored GT
        for pool in (~g.ignore, g.ignore):
            rp = [i for i in range(len(p.ids)) if i not in used_p]
            rg = [int(j) for j in np.flatnonzero(pool) if j not in used_g]
            if not rp or not rg:
                continue
            sub = iou[np.ix_(rp, rg)]
            rows, cols = linear_sum_assignment(np.where(sub >= thr, 1.0 - sub, 1e6))
            for r, c in zip(rows, cols):
                if sub[r, c] < thr:
                    continue
                i, j = rp[r], rg[c]
                gid = g.ids[j]
                if not g.ignore[j] and gid in last_match and last_match[gid] != p.ids[i]:
                    n_switch += 1
                matches.append((i, j))
                used_p.add(i)
                used_g.add(j)

        for i, j in matches:
            last_match[g.ids[j]] = p.ids[i]
        counted = [(i, j) for i, j in matches if not g.ignore[j]]
        n_visible = int((~g.ignore).sum())
        visible_ids.update(gid for gid, ign in zip(g.ids, g.ignore) if not ign)
        iou_sum += float(sum(iou[i, j] for i, j in counted))
        n_gt += n_visible
        n_match += len(counted)
        n_fp += len(p.ids) - len(matches)
        n_miss += n_visible - len(counted)

    gt_visible = [fr.subset(np.array([i in visible_ids for i in fr.ids], dtype=bool)) for fr in gt]
    return {
        "mota": 1.0 - (n_miss + n_fp + n_switch) / n_gt if n_gt else None,
        "motp": iou_sum / n_match if n_match else None,
        "id_switches": n_switch,
        "false_positives": n_fp,
        "misses": n_miss,
        "matches": n_match,
        "n_gt": n_gt,
        "unique_pred": _unique_vehicles(pred),
        "unique_gt": _unique_vehicles(gt_visible),
    }


# --------------------------------------------------------------------------- public API


def mark_visibility(gt: dict, n_points_per_frame: list[list[int]], min_points: int = 5) -> dict:
    """Return a copy of ``gt`` with ``n_points`` and ``ignore`` (``n_points < min_points``) per object.

    ``n_points_per_frame[f][j]`` is the number of infrastructure LiDAR points in object ``j`` of
    frame ``f``. Adds ``counts_visible`` / ``unique_visible`` computed from non-ignored objects
    only; ``counts`` / ``unique`` keep counting all GT.
    """
    out = copy.deepcopy(gt)
    frames = out.get("frames", [])
    if len(n_points_per_frame) != len(frames):
        raise ValueError(f"n_points_per_frame has {len(n_points_per_frame)} frames, gt has {len(frames)}")
    for f, (objects, n_points) in enumerate(zip(frames, n_points_per_frame)):
        if len(n_points) != len(objects):
            raise ValueError(f"frame {f}: {len(n_points)} point counts for {len(objects)} objects")
        for obj, n in zip(objects, n_points):
            obj["n_points"] = int(n)
            obj["ignore"] = bool(n < min_points)
    visible = [[o for o in objects if not o["ignore"]] for objects in frames]
    out["counts_visible"], out["unique_visible"] = summarize(visible, build_tracks(visible))
    return out


def evaluate_sequence(
    pred: dict,
    gt: dict,
    iou_thresholds: Sequence[float] = (0.3, 0.5),
    center: Sequence[float] = (0.0, 0.0),
    range_bins: Sequence[float] = (0, 20, 40, 60, 1e9),
    roi: RoiFn | None = None,
    iou_type: str = "bev",
    count_key: str = "counts",
) -> dict:
    """Evaluate one sequence of predictions against ground truth (both detections.json-shaped).

    Detection AP/PR use ``iou_type`` (``"bev"`` or ``"3d"``) IoU; tracking always uses BEV IoU.
    ``roi(xy (N, 2)) -> bool mask`` restricts the evaluation to box centres inside the ROI.
    GT objects with ``"ignore": true`` are "don't care" (never FN; predictions matching them are
    neither TP nor FP). ``metrics["counting"]`` compares against ``gt[count_key]``; if the GT has
    ``counts_visible``, ``metrics["counting_visible"]`` is added as well.
    Returns the metrics.json structure; undefined values (e.g. no GT) are ``None``.
    """
    iou_fn = {"bev": bev_iou_matrix, "3d": iou_3d_matrix}[iou_type]
    n_frames = max(len(pred.get("frames", [])), len(gt.get("frames", [])))
    p_frames = _to_frames(pred, n_frames, roi, "pred", use_ignore=False)
    g_frames = _to_frames(gt, n_frames, roi, "gt", use_ignore=True)

    detection = _evaluate_detection(p_frames, g_frames, iou_thresholds, center, range_bins, iou_fn)
    detection["iou_type"] = iou_type

    stored = roi is None
    metrics = {
        "n_frames": n_frames,
        "detection": detection,
        "counting": _counting(pred, gt, p_frames, g_frames, count_key, stored),
    }
    if "counts_visible" in gt:
        metrics["counting_visible"] = _counting(pred, gt, p_frames, g_frames, "counts_visible", stored)
    metrics["tracking"] = _evaluate_tracking(p_frames, g_frames)
    return metrics


def _mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def evaluate_many(results: dict[str, dict]) -> dict:
    """Aggregate per-sequence metrics: micro-averaged counts, mean APs over sequences."""
    seqs = list(results.values())
    out: dict = {"sequences": list(results), "n_frames": sum(r.get("n_frames", 0) for r in seqs)}
    if not seqs:
        return out

    dets = [r["detection"] for r in seqs]
    thr_keys = list(dets[0]["vehicle"]["ap"])
    vehicle: dict = {"ap": {k: _mean(d["vehicle"]["ap"].get(k) for d in dets) for k in thr_keys}}
    tp = sum(d["vehicle"]["tp"] for d in dets)
    n_pred = sum(d["vehicle"]["n_pred"] for d in dets)
    n_gt = sum(d["vehicle"]["n_gt"] for d in dets)
    precision, recall = _ratio(tp, n_pred), _ratio(tp, n_gt)
    vehicle.update(precision=precision, recall=recall, f1=_f1(precision, recall), n_gt=n_gt, n_pred=n_pred, tp=tp)

    per_class = {}
    for c in FINE_CLASSES:
        entries = [d["per_class"][c] for d in dets if c in d["per_class"]]
        if entries:
            per_class[c] = {
                "ap": {k: _mean(e["ap"].get(k) for e in entries) for k in thr_keys},
                "n_gt": sum(e["n_gt"] for e in entries),
                "n_pred": sum(e["n_pred"] for e in entries),
            }

    by_range = []
    for k, first in enumerate(dets[0]["by_range"]):
        tot = {key: sum(d["by_range"][k][key] for d in dets) for key in ("n_gt", "gt_matched", "n_pred", "tp")}
        by_range.append({"range": first["range"], "recall": _ratio(tot["gt_matched"], tot["n_gt"]),
                         "precision": _ratio(tot["tp"], tot["n_pred"]), **tot})

    out["detection"] = {"iou_thresholds": dets[0]["iou_thresholds"], "iou_type": dets[0].get("iou_type"),
                        "vehicle": vehicle, "per_class": per_class, "by_range": by_range}

    for key in ("counting", "counting_visible"):
        blocks = [r[key] for r in seqs if key in r]
        if len(blocks) == len(seqs):
            out[key] = {
                **_count_stats([c for b in blocks for c in b["pred"]], [c for b in blocks for c in b["gt"]]),
                "unique_pred": sum(b["unique_pred"] for b in blocks),
                "unique_gt": sum(b["unique_gt"] for b in blocks),
            }

    trk = [r["tracking"] for r in seqs]
    tot = {k: sum(t[k] for t in trk) for k in
           ("id_switches", "false_positives", "misses", "matches", "n_gt", "unique_pred", "unique_gt")}
    iou_sum = sum(t["motp"] * t["matches"] for t in trk if t["motp"] is not None)
    out["tracking"] = {
        "mota": 1.0 - (tot["misses"] + tot["false_positives"] + tot["id_switches"]) / tot["n_gt"] if tot["n_gt"] else None,
        "motp": iou_sum / tot["matches"] if tot["matches"] else None,
        **tot,
    }
    return out


def main(argv: Sequence[str] | None = None) -> dict:
    """Command-line entry point; returns the metrics dict."""
    parser = argparse.ArgumentParser(prog="python -m ulc.evaluate", description=__doc__.splitlines()[0])
    parser.add_argument("--pred", required=True, type=Path, help="detections.json")
    parser.add_argument("--gt", required=True, type=Path, help="gt.json")
    parser.add_argument("--out", type=Path, help="write metrics.json here")
    parser.add_argument("--center", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"),
                        help="centre for the range breakdown (default: 0 0)")
    parser.add_argument("--iou-type", choices=("bev", "3d"), default="bev", help="IoU for detection AP")
    args = parser.parse_args(argv)

    pred = json.loads(args.pred.read_text(encoding="utf-8"))
    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    metrics = evaluate_sequence(pred, gt, center=args.center, iou_type=args.iou_type)
    text = json.dumps(metrics, indent=2, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")

    d, t = metrics["detection"]["vehicle"], metrics["tracking"]
    fmt = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
    print("vehicle AP " + ", ".join(f"@{k}={fmt(v)}" for k, v in d["ap"].items())
          + f" | P={fmt(d['precision'])} R={fmt(d['recall'])} F1={fmt(d['f1'])}")
    for key in ("counting", "counting_visible"):
        if (c := metrics.get(key)) is not None:
            print(f"{key} MAE={fmt(c['mae'])} RMSE={fmt(c['rmse'])} bias={fmt(c['bias'])}"
                  f" within_1={fmt(c['within_1'])} | unique pred={c['unique_pred']} gt={c['unique_gt']}")
    print(f"tracking MOTA={fmt(t['mota'])} MOTP={fmt(t['motp'])} IDSW={t['id_switches']}"
          f" FP={t['false_positives']} FN={t['misses']}")
    return metrics


if __name__ == "__main__":
    main()
