"""Learned cluster classifier (gradient-boosted trees on geometric features).

Training data is mined automatically: every cluster produced by the geometric detector is
labelled with the class of the ground-truth box that contains most of its points, or
``background`` if none does. Models are trained with leave-one-sequence-out so that every
sequence is always processed by a model that has never seen it.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

from .background import CACHE_DIR, SceneModel
from .detect import FEATURE_NAMES, Detector
from .io import Sequence

GT_TO_DETECTOR = {
    "car": "car", "van": "van", "truck": "truck", "trailer": "truck", "bus": "bus",
    "cyclist": "cyclist", "motorcycle": "cyclist", "escooter": "cyclist", "pedestrian": "pedestrian",
}


def gt_boxes_at(labels: dict, t: float) -> tuple[np.ndarray, list[str]]:
    """All GT boxes (N, 7) and lower-cased classes at timestamp `t` (seconds)."""
    boxes, classes = [], []
    for tr in labels["tracks"]:
        ts = np.asarray(tr["timestamps"])
        k = np.flatnonzero(np.abs(ts - t) < 0.05)
        if not len(k):
            continue
        k = int(k[0])
        dim = tr["dimensions"][0] if len(tr["dimensions"]) == 1 else tr["dimensions"][k]
        boxes.append([*tr["positions"][k], *dim, tr["orientations"][k]])
        classes.append(tr["object_type"].lower())
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 7), classes


def in_box(xyz: np.ndarray, box: np.ndarray, margin: float = 0.3) -> np.ndarray:
    d = xyz[:, :2] - box[:2]
    c, s = np.cos(box[6]), np.sin(box[6])
    u, v = d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c
    return ((np.abs(u) <= box[3] / 2 + margin) & (np.abs(v) <= box[4] / 2 + margin)
            & (np.abs(xyz[:, 2] - box[2]) <= box[5] / 2 + margin))


def mine_clusters(seq_name: str, scene: SceneModel, step: int = 1) -> dict[str, np.ndarray]:
    """Features + auto labels for all clusters of a sequence (cached)."""
    path = CACHE_DIR / f"clusters_{seq_name}.npz"
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}
    seq = Sequence(seq_name)
    det = Detector(scene, seq.lidar_positions)
    labels = json.loads(seq.label_path.read_text())
    X, y, frame = [], [], []
    for i in range(0, len(seq), step):
        pts = seq.load_frame(i)
        clusters, _ = det.clusters(pts, seq_name)
        boxes, classes = gt_boxes_at(labels, seq.timestamps_ms[i] / 1000.0)
        for c in clusters:
            xyz = pts[c.idx, :3].astype(np.float64)
            best, frac = "background", 0.0
            for b, cl in zip(boxes, classes):
                if np.hypot(*(b[:2] - c.box[:2])) > 10:
                    continue
                f = in_box(xyz, b).mean()
                if f > frac:
                    best, frac = GT_TO_DETECTOR.get(cl, "background"), f
            X.append(c.features)
            y.append(best if frac >= 0.5 else "background")
            frame.append(i)
    out = {"X": np.asarray(X, np.float32), "y": np.asarray(y), "frame": np.asarray(frame)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                          l2_regularization=1.0, class_weight="balanced", random_state=0)


def train_leave_one_out(sequences: list[str], scene: SceneModel, verbose: bool = True) -> dict[str, object]:
    """Train one model per held-out sequence; returns {held_out_seq: model} and saves them."""
    data = {s: mine_clusters(s, scene) for s in sequences}
    models: dict[str, object] = {}
    report = {}
    for held in sequences:
        Xtr = np.concatenate([data[s]["X"] for s in sequences if s != held])
        ytr = np.concatenate([data[s]["y"] for s in sequences if s != held])
        model = make_model().fit(Xtr, ytr)
        models[held] = model
        pred = model.predict(data[held]["X"])
        rep = classification_report(data[held]["y"], pred, output_dict=True, zero_division=0)
        report[held] = rep
        if verbose:
            print(f"\n== held-out {held}  (train {len(ytr)} clusters, test {len(pred)})")
            print(classification_report(data[held]["y"], pred, digits=3, zero_division=0))
    # a final model on everything, for inference on new sequences
    models["all"] = make_model().fit(np.concatenate([d["X"] for d in data.values()]),
                                     np.concatenate([d["y"] for d in data.values()]))
    with open(CACHE_DIR / "classifiers.pkl", "wb") as fh:
        pickle.dump(models, fh)
    with open(CACHE_DIR / "classifier_report.json", "w") as fh:
        json.dump(report, fh, indent=1)
    return models


def load_models() -> dict[str, object]:
    with open(CACHE_DIR / "classifiers.pkl", "rb") as fh:
        return pickle.load(fh)


def feature_importance(model, X: np.ndarray, y: np.ndarray, n_repeats: int = 3) -> list[tuple[str, float]]:
    from sklearn.inspection import permutation_importance

    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=0, scoring="f1_macro")
    return sorted(zip(FEATURE_NAMES, r.importances_mean.tolist()), key=lambda t: -t[1])


if __name__ == "__main__":
    import sys

    seqs = sys.argv[1:] or ["20241126_0024_crossing1_09", "20241126_0008_crossing1_01", "20241127_0000_crossing1_00"]
    train_leave_one_out(seqs, SceneModel.get(seqs))
