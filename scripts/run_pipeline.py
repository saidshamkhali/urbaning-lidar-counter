"""End-to-end pipeline: detect → track → count → evaluate → export for the web viewer.

    python scripts/run_pipeline.py                  # all three sequences
    python scripts/run_pipeline.py --no-export      # results/metrics only
    python scripts/run_pipeline.py --retrain        # re-mine clusters and retrain classifiers

Outputs
  outputs/results/<seq>/{detections,gt,metrics}.json   (our results, safe to share)
  web/public/data/...                                  (viewer data incl. point clouds, git-ignored)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ulc.background import CACHE_DIR, SceneModel  # noqa: E402
from ulc.classify import in_box, load_models, train_leave_one_out  # noqa: E402
from ulc.detect import Detector  # noqa: E402
from ulc.evaluate import evaluate_many, evaluate_sequence, mark_visibility  # noqa: E402
from ulc.io import LIDAR, ROOT, Sequence, load_lanelet_map  # noqa: E402
from ulc.labels import load_gt  # noqa: E402
from ulc.schema import summarize  # noqa: E402
from ulc.tracking import Detection, track_sequence  # noqa: E402

SEQUENCES = ["20241126_0024_crossing1_09", "20241126_0008_crossing1_01", "20241127_0000_crossing1_00"]
RESULTS = ROOT / "outputs" / "results"
WEB_DATA = ROOT / "web" / "public" / "data"
SCALE = 0.01
VIS_MIN_POINTS = 5


def encode_points(pts: np.ndarray, info: dict, in_det: np.ndarray) -> bytes:
    """Pack points into the 8-byte records described in docs/DATA_FORMAT.md."""
    rec = np.zeros(len(pts), dtype=[("x", "<i2"), ("y", "<i2"), ("z", "<i2"), ("i", "u1"), ("f", "u1")])
    q = np.clip(np.round(pts[:, :3] / SCALE), -32767, 32767).astype(np.int16)
    rec["x"], rec["y"], rec["z"] = q[:, 0], q[:, 1], q[:, 2]
    rec["i"] = np.clip(pts[:, 3], 0, 255).astype(np.uint8)
    flags = pts[:, LIDAR].astype(np.uint8) & 3
    flags |= info["foreground"].astype(np.uint8) << 2
    flags |= info["ground"].astype(np.uint8) << 3
    flags |= in_det.astype(np.uint8) << 4
    rec["f"] = flags
    return rec.tobytes()


def gt_point_counts(pts: np.ndarray, info: dict, gt_frame: list[dict]) -> list[int]:
    """Number of non-ground infrastructure LiDAR points inside each GT box."""
    obj = pts[~info["ground"], :3].astype(np.float64)
    counts = []
    for o in gt_frame:
        b = np.asarray(o["box"], dtype=np.float64)
        near = np.abs(obj[:, 0] - b[0]) < 10
        near &= np.abs(obj[:, 1] - b[1]) < 10
        counts.append(int(in_box(obj[near], b, margin=0.2).sum()))
    return counts


def process(seq_name: str, scene: SceneModel, model, export: bool) -> dict:
    seq = Sequence(seq_name)
    det = Detector(scene, seq.lidar_positions, classifier=model)
    gt = load_gt(seq.label_path, seq.timestamps_ms)
    out_dir = WEB_DATA / seq_name
    if export:
        shutil.rmtree(out_dir, ignore_errors=True)
        (out_dir / "points").mkdir(parents=True)

    frames_det: list[list[Detection]] = []
    gt_npts: list[list[int]] = []
    frame_meta = []
    t0 = time.time()
    for i in range(len(seq)):
        pts = seq.load_frame(i)
        clusters, info = det.detect(pts, seq_name)
        frames_det.append([Detection(box=c.box, cls=c.cls, score=c.score, n_points=c.n_points) for c in clusters])
        gt_npts.append(gt_point_counts(pts, info, gt["frames"][i]))
        if export:
            in_det = np.zeros(len(pts), bool)
            for c in clusters:
                in_det[c.idx] = True
            (out_dir / "points" / f"{i:06d}.bin").write_bytes(encode_points(pts, info, in_det))
            frame_meta.append({"file": f"points/{i:06d}.bin", "n_points": int(len(pts))})
    t_det = time.time() - t0

    frames, tracks = track_sequence(frames_det, dt=0.1)
    counts, unique = summarize(frames, tracks)
    pred = {"source": "detector", "frames": frames, "tracks": tracks, "counts": counts, "unique": unique}
    gt = mark_visibility(gt, gt_npts, min_points=VIS_MIN_POINTS)

    metrics = evaluate_sequence(pred, gt, center=tuple(det.center))
    metrics_visible = evaluate_sequence(pred, gt, center=tuple(det.center), count_key="counts_visible")
    metrics["counting_visible"] = metrics_visible.get("counting_visible", metrics_visible["counting"])
    metrics["runtime"] = {"detect_s_per_frame": t_det / len(seq)}

    res_dir = RESULTS / seq_name
    res_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in (("detections", pred), ("gt", gt), ("metrics", metrics)):
        (res_dir / f"{name}.json").write_text(json.dumps(obj, separators=(",", ":"), default=_json_default))

    if export:
        for name in ("detections", "gt", "metrics"):
            shutil.copy(res_dir / f"{name}.json", out_dir / f"{name}.json")
        allp = np.concatenate([seq.load_frame(0)[:, :3], seq.load_frame(len(seq) - 1)[:, :3]])
        meta = {
            "id": seq_name, "crossing": seq.crossing, "map": f"map_{seq.crossing}.json", "fps": 10,
            "n_frames": len(seq), "timestamps": [t / 1000.0 for t in seq.timestamps_ms],
            "bounds": {"xmin": float(np.percentile(allp[:, 0], 1)), "xmax": float(np.percentile(allp[:, 0], 99)),
                       "ymin": float(np.percentile(allp[:, 1], 1)), "ymax": float(np.percentile(allp[:, 1], 99)),
                       "zmin": float(np.percentile(allp[:, 2], 1)), "zmax": float(np.percentile(allp[:, 2], 99))},
            "lidars": [{"name": n, "index": k, "position": seq.lidar_positions[k].round(3).tolist()}
                       for k, n in enumerate(seq.lidars)],
            "point_format": {"stride_bytes": 8, "scale": SCALE, "origin": [0.0, 0.0, 0.0],
                             "fields": ["x:int16", "y:int16", "z:int16", "intensity:uint8", "flags:uint8"]},
            "frames": frame_meta,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta))
    print(f"[{seq_name}] {t_det / len(seq) * 1000:.0f} ms/frame | unique vehicles pred {unique['vehicle']} "
          f"vs gt {gt['unique']['vehicle']} (visible {gt['unique_visible']['vehicle']})")
    return metrics


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*", default=SEQUENCES)
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--rebuild-scene", action="store_true")
    args = ap.parse_args()

    scene = SceneModel.get(SEQUENCES, rebuild=args.rebuild_scene)
    if args.retrain or not (CACHE_DIR / "classifiers.pkl").exists():
        if args.retrain:
            for f in CACHE_DIR.glob("clusters_*.npz"):
                f.unlink()
        train_leave_one_out(SEQUENCES, scene)
    models = load_models()

    all_metrics = {}
    for s in args.seq:
        all_metrics[s] = process(s, scene, models.get(s, models["all"]), export=not args.no_export)
    summary = evaluate_many(all_metrics)
    (RESULTS / "summary.json").write_text(json.dumps({"sequences": all_metrics, "overall": summary}, indent=1,
                                                     default=_json_default))

    if not args.no_export:
        WEB_DATA.mkdir(parents=True, exist_ok=True)
        (WEB_DATA / "map_crossing1.json").write_text(json.dumps(load_lanelet_map(crossing="crossing1")))
        index = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sequences": []}
        for s in args.seq:
            det = json.loads((RESULTS / s / "detections.json").read_text())
            gt = json.loads((RESULTS / s / "gt.json").read_text())
            index["sequences"].append({
                "id": s, "crossing": s.split("_")[2], "n_frames": len(det["frames"]),
                "duration_s": len(det["frames"]) / 10.0,
                "unique_vehicles": det["unique"]["vehicle"], "unique_vehicles_gt": gt["unique"]["vehicle"],
                "unique_vehicles_gt_visible": gt["unique_visible"]["vehicle"],
            })
        (WEB_DATA / "index.json").write_text(json.dumps(index, indent=1))
    print(json.dumps(summary, indent=1, default=_json_default)[:3000])


if __name__ == "__main__":
    main()
