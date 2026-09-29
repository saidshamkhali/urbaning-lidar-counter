import copy
import json

import numpy as np
import pytest

from ulc.evaluate import evaluate_many, evaluate_sequence, main, mark_visibility
from ulc.schema import build_tracks, summarize

N_FRAMES = 60


def make_sequence(n_frames: int = N_FRAMES, seed: int = 0) -> dict:
    """Synthetic GT: 8 cars on parallel lanes (some moving, some parked), a truck and 2 pedestrians."""
    rng = np.random.default_rng(seed)
    objects = []
    for k in range(8):
        speed = 0.0 if k % 3 == 0 else rng.uniform(3, 12)
        objects.append(dict(id=k + 1, cls="car", x0=rng.uniform(-50, 10), y=-35 + 10 * k, v=speed, l=4.4, w=1.8, h=1.5))
    objects.append(dict(id=20, cls="truck", x0=-30.0, y=45.0, v=6.0, l=10.0, w=2.5, h=3.5))
    objects.append(dict(id=30, cls="pedestrian", x0=0.0, y=55.0, v=1.4, l=0.6, w=0.6, h=1.7))
    objects.append(dict(id=31, cls="pedestrian", x0=5.0, y=60.0, v=0.0, l=0.6, w=0.6, h=1.7))
    frames = []
    for f in range(n_frames):
        frame = []
        for o in objects:
            x = o["x0"] + o["v"] * 0.1 * f
            frame.append({
                "id": o["id"], "cls": o["cls"], "box": [x, o["y"], 1.2, o["l"], o["w"], o["h"], 0.0],
                "score": 1.0, "speed": o["v"], "moving": o["v"] > 1.0, "n_points": 0,
            })
        frames.append(frame)
    return finalize({"source": "gt", "frames": frames})


def finalize(result: dict) -> dict:
    result["tracks"] = build_tracks(result["frames"])
    result["counts"], result["unique"] = summarize(result["frames"], result["tracks"])
    return result


def as_pred(gt: dict, seed: int = 1) -> dict:
    """Copy GT as detector output with distinct random scores."""
    pred = copy.deepcopy(gt)
    pred["source"] = "detector"
    rng = np.random.default_rng(seed)
    for frame in pred["frames"]:
        for obj in frame:
            obj["score"] = float(rng.uniform(0.5, 1.0))
    return pred


def n_vehicles(result: dict) -> int:
    return sum(1 for fr in result["frames"] for o in fr if o["cls"] in ("car", "truck"))


def test_gt_vs_gt_is_perfect():
    gt = make_sequence()
    m = evaluate_sequence(as_pred(gt), gt)
    det = m["detection"]
    assert det["vehicle"]["ap"] == {"0.3": pytest.approx(1.0), "0.5": pytest.approx(1.0)}
    assert det["vehicle"]["precision"] == 1.0 and det["vehicle"]["recall"] == 1.0 and det["vehicle"]["f1"] == 1.0
    assert det["per_class"]["car"]["ap"]["0.5"] == pytest.approx(1.0)
    assert det["per_class"]["pedestrian"]["n_gt"] == 2 * N_FRAMES
    assert m["counting"]["mae"] == 0.0 and m["counting"]["rmse"] == 0.0 and m["counting"]["within_1"] == 1.0
    assert m["counting"]["gt"] == [9] * N_FRAMES
    assert m["counting"]["unique_pred"] == m["counting"]["unique_gt"] == 9
    t = m["tracking"]
    assert t["mota"] == 1.0 and t["motp"] == pytest.approx(1.0) and t["id_switches"] == 0
    assert t["unique_gt"] == 9
    assert all(b["recall"] in (1.0, None) for b in det["by_range"])
    assert len(det["pr_curve"]["recall"]) <= 101
    json.dumps(m, allow_nan=False)  # plain, JSON-serialisable values


def test_shift_degrades_iou_as_expected():
    gt = make_sequence()
    pred = as_pred(gt)
    for frame in pred["frames"]:
        for obj in frame:
            obj["box"][0] += 1.5  # car: IoU = 2.9/5.9 ~ 0.49; truck 8.5/11.5; pedestrians lost
    m = evaluate_sequence(pred, gt)
    ap = m["detection"]["vehicle"]["ap"]
    assert ap["0.3"] == pytest.approx(1.0)
    # Only the truck (1 of 9 vehicles) survives IoU 0.5; its score rank is random -> AP <= 1/9.
    assert 0.0 < ap["0.5"] <= 1 / 9 + 1e-9
    assert m["detection"]["per_class"]["car"]["ap"]["0.5"] == 0.0
    assert m["detection"]["per_class"]["pedestrian"]["ap"]["0.3"] == 0.0
    assert m["counting"]["mae"] == 0.0  # counts are unaffected by localisation
    assert m["tracking"]["mota"] == 1.0
    assert m["tracking"]["motp"] < 0.6


def test_jitter_lowers_motp_but_keeps_matches():
    gt = make_sequence()
    pred = as_pred(gt)
    rng = np.random.default_rng(5)
    for frame in pred["frames"]:
        for obj in frame:
            obj["box"][0] += rng.normal(0, 0.2)
            obj["box"][1] += rng.normal(0, 0.2)
    m = evaluate_sequence(pred, gt)
    assert m["detection"]["vehicle"]["ap"]["0.3"] == pytest.approx(1.0)
    assert 0.6 < m["tracking"]["motp"] < 1.0
    assert m["tracking"]["mota"] == 1.0


def test_dropped_detections_reduce_recall_exactly():
    gt = make_sequence()
    pred = as_pred(gt)
    dropped = 0
    for f, frame in enumerate(pred["frames"]):
        keep = []
        for obj in frame:
            if obj["cls"] == "car" and (f + obj["id"]) % 4 == 0:
                dropped += 1
            else:
                keep.append(obj)
        pred["frames"][f] = keep
    finalize(pred)
    total = n_vehicles(gt)
    m = evaluate_sequence(pred, gt)
    v = m["detection"]["vehicle"]
    assert v["recall"] == pytest.approx((total - dropped) / total)
    assert v["precision"] == 1.0
    assert v["ap"]["0.3"] == pytest.approx((total - dropped) / total)  # precision 1 up to max recall
    c = m["counting"]
    assert c["mae"] == pytest.approx(dropped / N_FRAMES)
    assert c["bias"] == pytest.approx(-dropped / N_FRAMES)
    assert m["tracking"]["misses"] == dropped
    assert m["tracking"]["mota"] == pytest.approx(1 - dropped / total)
    assert m["tracking"]["id_switches"] == 0


def test_low_score_false_positives_hurt_precision_not_ap():
    gt = make_sequence()
    pred = as_pred(gt)
    for f, frame in enumerate(pred["frames"]):
        frame.append({"id": 900 + f % 3, "cls": "car", "box": [200.0, 200.0 + 10 * (f % 3), 1, 4, 2, 1.5, 0],
                      "score": 0.1, "speed": 0.0, "moving": False, "n_points": 5})
    finalize(pred)
    m = evaluate_sequence(pred, gt)
    v = m["detection"]["vehicle"]
    total = n_vehicles(gt)
    assert v["ap"]["0.3"] == pytest.approx(1.0)
    assert v["precision"] == pytest.approx(total / (total + N_FRAMES))
    assert m["counting"]["bias"] == pytest.approx(1.0)
    assert m["counting"]["within_1"] == 1.0
    assert m["tracking"]["false_positives"] == N_FRAMES
    assert m["counting"]["unique_pred"] == 12


def test_id_swap_is_counted():
    gt = make_sequence()
    pred = as_pred(gt)
    # From frame 30 on, the tracker swaps the ids of cars 2 and 3.
    for frame in pred["frames"][30:]:
        for obj in frame:
            if obj["id"] in (2, 3):
                obj["id"] = 5 - obj["id"]
    m = evaluate_sequence(pred, gt)
    assert m["tracking"]["id_switches"] == 2
    assert m["tracking"]["mota"] == pytest.approx(1 - 2 / n_vehicles(gt))
    assert m["detection"]["vehicle"]["ap"]["0.5"] == pytest.approx(1.0)


def test_track_fragmentation_counts_switch():
    gt = make_sequence()
    pred = as_pred(gt)
    for frame in pred["frames"][40:]:
        for obj in frame:
            if obj["id"] == 4:
                obj["id"] = 444  # new id after a break
    m = evaluate_sequence(pred, gt)
    assert m["tracking"]["id_switches"] == 1
    assert m["tracking"]["unique_pred"] == 10


def test_range_bins_and_roi():
    gt = make_sequence()
    pred = as_pred(gt)
    m = evaluate_sequence(pred, gt, range_bins=(0, 30, 1e9))
    bins = m["detection"]["by_range"]
    assert [b["range"] for b in bins] == ["0-30", "30+"]
    assert sum(b["n_gt"] for b in bins) == n_vehicles(gt)

    # Drop every prediction with y > 0, then evaluate only on the y <= 0 half.
    for frame in pred["frames"]:
        frame[:] = [o for o in frame if o["box"][1] <= 0]
    roi = lambda xy: xy[:, 1] <= 0  # noqa: E731
    full = evaluate_sequence(pred, gt)
    inside = evaluate_sequence(pred, gt, roi=roi)
    assert full["detection"]["vehicle"]["recall"] < 1.0
    assert inside["detection"]["vehicle"]["recall"] == 1.0
    assert inside["counting"]["mae"] == 0.0
    assert inside["counting"]["gt"][0] == 4  # cars on lanes y = -35, -25, -15, -5
    assert inside["tracking"]["mota"] == 1.0


def test_no_gt_is_null_safe():
    gt = {"source": "gt", "frames": [[] for _ in range(5)]}
    pred = as_pred(make_sequence(5))
    m = evaluate_sequence(pred, gt)
    assert m["detection"]["vehicle"]["ap"]["0.3"] is None
    assert m["detection"]["vehicle"]["recall"] is None
    assert m["detection"]["vehicle"]["precision"] == 0.0
    assert m["tracking"]["mota"] is None
    assert m["counting"]["gt"] == [0] * 5
    json.dumps(m, allow_nan=False)
    empty = evaluate_sequence({"frames": []}, {"frames": []})
    json.dumps(empty, allow_nan=False)


def test_counts_block_is_used_when_present():
    gt = make_sequence(10)
    pred = as_pred(gt)
    pred["counts"] = [dict(c, vehicle=c["vehicle"] + 2) for c in pred["counts"]]
    assert evaluate_sequence(pred, gt)["counting"]["bias"] == pytest.approx(2.0)
    del pred["counts"]
    assert evaluate_sequence(pred, gt)["counting"]["bias"] == 0.0


def test_evaluate_many_micro_averages():
    gt = make_sequence()
    perfect = evaluate_sequence(as_pred(gt), gt)
    pred = as_pred(gt)
    for frame in pred["frames"]:
        frame[:] = [o for o in frame if o["id"] != 1]
    finalize(pred)
    partial = evaluate_sequence(pred, gt)
    agg = evaluate_many({"a": perfect, "b": partial})
    total = n_vehicles(gt)
    assert agg["sequences"] == ["a", "b"]
    assert agg["n_frames"] == 2 * N_FRAMES
    assert agg["detection"]["vehicle"]["recall"] == pytest.approx((2 * total - N_FRAMES) / (2 * total))
    assert agg["detection"]["vehicle"]["ap"]["0.3"] == pytest.approx((1 + 8 / 9) / 2)
    assert agg["counting"]["mae"] == pytest.approx(0.5)
    assert agg["counting"]["unique_gt"] == 18 and agg["counting"]["unique_pred"] == 17
    assert agg["tracking"]["mota"] == pytest.approx(1 - N_FRAMES / (2 * total))
    json.dumps(agg, allow_nan=False)


def test_cli(tmp_path, capsys):
    gt = make_sequence(20)
    (tmp_path / "gt.json").write_text(json.dumps(gt))
    (tmp_path / "det.json").write_text(json.dumps(as_pred(gt)))
    out = tmp_path / "metrics.json"
    main(["--pred", str(tmp_path / "det.json"), "--gt", str(tmp_path / "gt.json"), "--out", str(out)])
    metrics = json.loads(out.read_text())
    assert metrics["tracking"]["mota"] == 1.0
    assert "MOTA=1.000" in capsys.readouterr().out


def hide(gt: dict, hidden_ids: set, min_points: int = 5) -> dict:
    """Mark the objects with the given ids as not visible (0 points), everything else 100 points."""
    n_points = [[0 if o["id"] in hidden_ids else 100 for o in frame] for frame in gt["frames"]]
    return mark_visibility(gt, n_points, min_points=min_points)


def test_mark_visibility():
    gt = make_sequence(10)
    marked = hide(gt, {2, 30})
    assert "ignore" not in gt["frames"][0][0]  # input untouched
    objs = {o["id"]: o for o in marked["frames"][0]}
    assert objs[2]["ignore"] and objs[2]["n_points"] == 0
    assert not objs[1]["ignore"] and objs[1]["n_points"] == 100
    assert marked["counts"][0]["vehicle"] == 9 and marked["counts_visible"][0]["vehicle"] == 8
    assert marked["counts_visible"][0]["vru"] == 1
    assert marked["unique"]["vehicle"] == 9 and marked["unique_visible"]["vehicle"] == 8
    with pytest.raises(ValueError):
        mark_visibility(gt, [[1]] * 10)


def test_ignored_gt_is_not_a_miss():
    gt = hide(make_sequence(), {2})
    pred = as_pred(gt)
    for frame in pred["frames"]:
        frame[:] = [o for o in frame if o["id"] != 2]
    finalize(pred)
    visible = n_vehicles(gt) - N_FRAMES
    m = evaluate_sequence(pred, gt)
    v = m["detection"]["vehicle"]
    assert v["n_gt"] == visible and v["recall"] == 1.0 and v["ap"]["0.5"] == pytest.approx(1.0)
    assert m["detection"]["per_class"]["car"]["n_gt"] == 7 * N_FRAMES
    t = m["tracking"]
    assert t["mota"] == 1.0 and t["misses"] == 0 and t["n_gt"] == visible and t["unique_gt"] == 8
    # All-GT counting sees one car too few per frame, visibility-aware counting is exact.
    assert m["counting"]["bias"] == pytest.approx(-1.0) and m["counting"]["unique_gt"] == 9
    assert m["counting_visible"]["mae"] == 0.0 and m["counting_visible"]["unique_gt"] == 8
    alt = evaluate_sequence(pred, gt, count_key="counts_visible")
    assert alt["counting"] == m["counting_visible"]
    # Recomputed (no stored blocks, e.g. with a ROI) gives the same answer.
    everywhere = evaluate_sequence(pred, gt, roi=lambda xy: np.ones(len(xy), dtype=bool))
    assert everywhere["counting_visible"] == m["counting_visible"]


def test_predictions_on_ignored_gt_are_dropped():
    gt = hide(make_sequence(), {2})
    pred = as_pred(gt)  # still detects the hidden car
    m = evaluate_sequence(pred, gt)
    v = m["detection"]["vehicle"]
    visible = n_vehicles(gt) - N_FRAMES
    assert v["n_pred"] == visible and v["precision"] == 1.0 and v["ap"]["0.3"] == pytest.approx(1.0)
    assert sum(b["n_pred"] for b in m["detection"]["by_range"]) == visible
    t = m["tracking"]
    assert t["false_positives"] == 0 and t["mota"] == 1.0 and t["matches"] == visible


def test_regular_gt_preferred_over_ignored():
    def obj(i, x, **kw):
        return {"id": i, "cls": "car", "box": [x, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0], "score": kw.get("score", 1.0), **kw}

    gt = {"frames": [[obj(1, 0.0), obj(2, 1.0, ignore=True)]]}
    # One prediction between both GT boxes (IoU 0.78 with each): it must match the regular GT.
    m = evaluate_sequence({"frames": [[obj(10, 0.5)]]}, gt)
    assert m["detection"]["vehicle"]["recall"] == 1.0 and m["detection"]["vehicle"]["precision"] == 1.0
    assert m["tracking"]["matches"] == 1 and m["tracking"]["misses"] == 0
    # A second, lower-scored prediction is absorbed by the ignored GT; a third far away is a FP.
    pred = {"frames": [[obj(10, 0.5, score=0.9), obj(11, 1.0, score=0.8), obj(12, 30.0, score=0.7)]]}
    m = evaluate_sequence(pred, gt)
    v = m["detection"]["vehicle"]
    assert v["tp"] == 1 and v["n_pred"] == 2 and v["precision"] == 0.5
    assert v["ap"]["0.3"] == pytest.approx(1.0)  # the FP ranks below the only TP
    assert m["tracking"]["false_positives"] == 1 and m["tracking"]["mota"] == 0.0
