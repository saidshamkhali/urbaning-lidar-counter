import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from ulc.labels import load_gt, track_speeds
from ulc.schema import FINE_CLASSES, GROUP_OF, normalize_class

REAL_LABELS = Path("data/labels/20241126_0024_crossing1_09.json")
T0_MS = 1732640002000


def write_labels(tmp_path: Path, tracks: list[dict]) -> Path:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"id": "seq", "timestamp": T0_MS / 1000, "tracks": tracks}))
    return path


def test_normalize_class():
    assert normalize_class("EScooter") == "escooter"
    assert normalize_class("e-scooter") == "escooter"
    assert normalize_class("Car") == "car"
    assert normalize_class("Trailer") == "trailer"
    assert normalize_class("spaceship") == "other"
    assert set(GROUP_OF) == set(FINE_CLASSES)


def test_track_speeds():
    t = np.arange(20) * 0.1
    pos = np.column_stack([5.0 * t, np.zeros(20), np.ones(20)])
    assert np.allclose(track_speeds(t, pos), 5.0)
    assert np.allclose(track_speeds(t, np.zeros((20, 3))), 0.0)
    assert track_speeds(t[:1], pos[:1]).tolist() == [0.0]


def test_load_gt_synthetic(tmp_path):
    t = [(T0_MS + 100 * i) / 1000 for i in range(10)]
    tracks = [
        {"track_id": 7, "object_type": "Car", "timestamps": t,
         "dimensions": [[4.0, 2.0, 1.5]], "positions": [[1.0 * i, 0.0, 0.8] for i in range(10)],
         "orientations": [0.0] * 10, "attributes": {"occlusion": ["not_occluded"] * 10}},
        {"track_id": 3, "object_type": "EScooter", "timestamps": t[2:5],
         "dimensions": [[1.0, 0.5, 1.8]] * 3, "positions": [[5.0, 5.0, 1.0]] * 3,
         "orientations": [4.0] * 3, "attributes": {}},
    ]
    # Frames: shifted by +30 ms (still matched), plus one frame with no labels at all.
    frames_ms = [T0_MS + 100 * i + 30 for i in range(10)] + [T0_MS + 5000]
    gt = load_gt(write_labels(tmp_path, tracks), frames_ms)

    assert gt["source"] == "gt" and len(gt["frames"]) == 11
    assert gt["frames"][10] == []
    assert [o["id"] for o in gt["frames"][3]] == [3, 7]  # sorted by track id
    car = gt["frames"][0][0]
    assert car["cls"] == "car" and car["score"] == 1.0 and car["n_points"] == 0
    assert car["box"] == [0.0, 0.0, 0.8, 4.0, 2.0, 1.5, 0.0]
    assert car["speed"] == pytest.approx(10.0) and car["moving"]
    assert car["occlusion"] == "not_occluded"
    scooter = gt["frames"][2][0]
    assert scooter["cls"] == "escooter" and not scooter["moving"] and "occlusion" not in scooter
    assert -np.pi < scooter["box"][6] <= np.pi
    assert gt["tracks"]["7"] == {"cls": "car", "group": "vehicle", "first": 0, "last": 9,
                                 "moving": True, "max_speed": pytest.approx(10.0), "n_frames": 10}
    assert gt["counts"][3] == {"vehicle": 1, "vru": 1, "other": 0, "moving": 1, "parked": 0,
                               "by_class": {"car": 1, "escooter": 1}}
    assert gt["unique"] == {"vehicle": 1, "vru": 1, "other": 0, "by_class": {"car": 1, "escooter": 1}}


def test_load_gt_tolerance(tmp_path):
    tracks = [{"track_id": 1, "object_type": "Car", "timestamps": [T0_MS / 1000],
               "dimensions": [[4.0, 2.0, 1.5]], "positions": [[0.0, 0.0, 0.8]], "orientations": [0.0]}]
    path = write_labels(tmp_path, tracks)
    assert len(load_gt(path, [T0_MS + 50])["frames"][0]) == 1
    assert len(load_gt(path, [T0_MS - 49])["frames"][0]) == 1
    assert load_gt(path, [T0_MS + 60])["frames"][0] == []
    # Nearest frame wins.
    assert [len(f) for f in load_gt(path, [T0_MS - 40, T0_MS + 10])["frames"]] == [0, 1]


@pytest.mark.skipif(not REAL_LABELS.exists(), reason="real label file not available")
def test_load_gt_real_labels():
    frames_ms = [T0_MS + 100 * i for i in range(200)]
    gt = load_gt(REAL_LABELS, frames_ms)
    frames = gt["frames"]
    n_labelled = sum(1 for f in frames if f)
    hist = Counter(o["cls"] for f in frames for o in f)
    vehicles = np.array([c["vehicle"] for c in gt["counts"]])
    moving = np.array([c["moving"] for c in gt["counts"]])
    print(
        f"\n[load_gt] {REAL_LABELS.name}: {n_labelled}/{len(frames)} frames with labels, "
        f"{len(gt['tracks'])} tracks, object class histogram {dict(hist.most_common())}, "
        f"mean vehicles/frame {vehicles.mean():.2f} (min {vehicles.min()}, max {vehicles.max()}), "
        f"mean moving vehicles/frame {moving.mean():.2f}, unique {gt['unique']}"
    )

    assert len(frames) == 200 and n_labelled == 200
    assert set(hist) <= set(FINE_CLASSES) and hist["car"] > hist["truck"] > 0
    assert 5 < vehicles.mean() < 200  # many parked cars are labelled
    assert 0 < moving.mean() < vehicles.mean()
    assert 100 < len(gt["tracks"]) <= 141
    assert gt["unique"]["vehicle"] == sum(1 for t in gt["tracks"].values() if t["group"] == "vehicle")
    for f in frames:
        ids = [o["id"] for o in f]
        assert len(ids) == len(set(ids))
        for o in f:
            assert len(o["box"]) == 7 and all(np.isfinite(o["box"]))
            assert 0 <= o["speed"] < 40 and o["moving"] == (o["speed"] > 1.0)
    json.dumps(gt, allow_nan=False)
