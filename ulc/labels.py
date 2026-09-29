"""Load UrbanIng-V2X ground-truth label files into the gt.json schema (``docs/DATA_FORMAT.md``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import wrap_angle
from .schema import MOVING_SPEED, build_tracks, normalize_class, summarize

__all__ = ["MATCH_TOLERANCE_MS", "SPEED_SMOOTH_HALF_WINDOW", "track_speeds", "load_gt"]

#: Labels are matched to the nearest frame if within this many milliseconds.
MATCH_TOLERANCE_MS = 50.0
#: Speeds are smoothed with a centred moving average over +-this many label steps.
SPEED_SMOOTH_HALF_WINDOW = 2


def track_speeds(times_s: np.ndarray, positions: np.ndarray, half_window: int = SPEED_SMOOTH_HALF_WINDOW) -> np.ndarray:
    """Ground-plane speed (m/s) per label of one track.

    Velocity is a central difference of ``x, y`` over time (one-sided at the ends), smoothed
    with a centred moving average over ``+-half_window`` samples (truncated at the ends).
    """
    times_s = np.asarray(times_s, dtype=np.float64)
    xy = np.asarray(positions, dtype=np.float64)[:, :2]
    n = len(times_s)
    if n < 2:
        return np.zeros(n)
    vel = np.stack([np.gradient(xy[:, k], times_s) for k in range(2)], axis=1)
    if half_window > 0:
        csum = np.vstack([np.zeros((1, 2)), np.cumsum(vel, axis=0)])
        lo = np.clip(np.arange(n) - half_window, 0, n)
        hi = np.clip(np.arange(n) + half_window + 1, 0, n)
        vel = (csum[hi] - csum[lo]) / (hi - lo)[:, None]
    return np.linalg.norm(vel, axis=1)


def _match_frames(label_ms: np.ndarray, frame_ms: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Index of the nearest frame for each label timestamp, or -1 if further than the tolerance."""
    if len(frame_ms) == 0:
        return np.full(len(label_ms), -1)
    sorted_ms = frame_ms[order]
    pos = np.clip(np.searchsorted(sorted_ms, label_ms), 1, max(len(sorted_ms) - 1, 1))
    left = np.clip(pos - 1, 0, len(sorted_ms) - 1)
    right = np.clip(pos, 0, len(sorted_ms) - 1)
    nearest = np.where(np.abs(sorted_ms[left] - label_ms) <= np.abs(sorted_ms[right] - label_ms), left, right)
    ok = np.abs(sorted_ms[nearest] - label_ms) <= MATCH_TOLERANCE_MS + 1e-6
    return np.where(ok, order[nearest], -1)


def load_gt(label_path: str | Path, frame_timestamps_ms: list[int]) -> dict:
    """Load a label file as a gt.json-shaped dict aligned to the given frame timestamps.

    Each label is assigned to the nearest frame within +-50 ms; frames without labels are empty
    lists. Objects carry ``id, cls, box, score (1.0), speed, moving, n_points (0)`` and, when
    available, the ``occlusion`` attribute of that timestamp.
    """
    with open(label_path, encoding="utf-8") as fh:
        data = json.load(fh)

    frame_ms = np.asarray(frame_timestamps_ms, dtype=np.float64)
    order = np.argsort(frame_ms, kind="stable")
    frames: list[list[dict]] = [[] for _ in range(len(frame_ms))]

    for track in sorted(data.get("tracks", []), key=lambda t: t["track_id"]):
        times = np.asarray(track["timestamps"], dtype=np.float64)
        if len(times) == 0:
            continue
        positions = np.asarray(track["positions"], dtype=np.float64).reshape(-1, 3)
        dims = np.asarray(track["dimensions"], dtype=np.float64).reshape(-1, 3)
        if len(dims) == 1:
            dims = np.repeat(dims, len(times), axis=0)
        yaws = wrap_angle(np.asarray(track["orientations"], dtype=np.float64).reshape(-1))
        occlusion = (track.get("attributes") or {}).get("occlusion")
        speeds = np.round(track_speeds(times, positions), 3)  # "moving" agrees with the stored speed
        cls = normalize_class(track["object_type"])

        label_ms = np.round(times * 1000.0)
        frame_idx = _match_frames(label_ms, frame_ms, order)
        # If several labels of this track land on the same frame, keep the closest one.
        best: dict[int, int] = {}
        for i, f in enumerate(frame_idx):
            if f >= 0 and (f not in best or abs(label_ms[i] - frame_ms[f]) < abs(label_ms[best[f]] - frame_ms[f])):
                best[int(f)] = i

        for f, i in best.items():
            obj = {
                "id": int(track["track_id"]),
                "cls": cls,
                "box": [round(float(v), 4) for v in (*positions[i], *dims[i], yaws[i])],
                "score": 1.0,
                "moving": bool(speeds[i] > MOVING_SPEED),
                "speed": float(speeds[i]),
                "n_points": 0,
            }
            if occlusion is not None and i < len(occlusion):
                obj["occlusion"] = occlusion[i]
            frames[f].append(obj)

    tracks = build_tracks(frames)
    counts, unique = summarize(frames, tracks)
    return {"source": "gt", "frames": frames, "tracks": tracks, "counts": counts, "unique": unique}
