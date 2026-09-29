"""Tests for ulc.tracking on synthetic 10 Hz / 200-frame scenarios."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ulc.tracking import (
    Detection,
    MultiObjectTracker,
    bev_iou,
    class_penalty,
    track_sequence,
)

DT = 0.1
N = 200


# --------------------------------------------------------------------------------------------
# synthetic data helpers
# --------------------------------------------------------------------------------------------


def wrap(a, period=2 * math.pi):
    return (np.asarray(a) + period / 2) % period - period / 2


def noisy_det(rng, box, cls="car", pos_sigma=0.15, size_noise=0.3, flip_prob=0.2, score=None, n_points=None):
    """A detection of ``box`` with centre/size/yaw noise and L-shape ambiguity flips."""
    b = np.array(box, float)
    b[:2] += rng.normal(0.0, pos_sigma, 2)
    b[3:6] = np.maximum(b[3:6] + rng.uniform(-size_noise, size_noise, 3), 0.3)
    b[6] += rng.normal(0.0, 0.03)
    if rng.random() < flip_prob:
        kind = rng.integers(3)
        if kind == 0:  # wrong orientation of the rectangle (long side along the wrong axis)
            b[6] += math.pi / 2
        elif kind == 1:  # heading reversed
            b[6] += math.pi
        else:  # same rectangle, other parametrisation (l <-> w swapped)
            b[6] += math.pi / 2
            b[3], b[4] = b[4], b[3]
    b[6] = float(wrap(b[6]))
    if score is None:
        score = float(rng.uniform(0.6, 0.95))
    if n_points is None:
        n_points = int(rng.integers(50, 400))
    return Detection(b, cls, score, n_points)


def trajectory(x0, y0, yaw0, v, t_start, segments, n=N, dt=DT, lim=70.0):
    """GT states per frame: dict frame -> (x, y, yaw, speed).  segments: [(duration_s, yaw_rate)]."""
    out = {}
    x, y, yaw = x0, y0, yaw0
    f = int(round(t_start / dt))
    seg_frames = []
    for dur, rate in segments:
        seg_frames += [rate] * int(round(dur / dt))
    k = 0
    while f < n:
        if abs(x) > lim or abs(y) > lim:
            break
        out[f] = (x, y, yaw, v)
        rate = seg_frames[k] if k < len(seg_frames) else 0.0
        # exact integration of a constant-turn-rate step
        if abs(rate) < 1e-9:
            x += v * dt * math.cos(yaw)
            y += v * dt * math.sin(yaw)
        else:
            x += v / rate * (math.sin(yaw + rate * dt) - math.sin(yaw))
            y += v / rate * (-math.cos(yaw + rate * dt) + math.cos(yaw))
        yaw += rate * dt
        f += 1
        k += 1
    return out


def parked_cars(n_cars=15):
    """GT boxes of parked cars: two parallel rows and one perpendicular row."""
    boxes = []
    for i in range(5):
        boxes.append([-30.0 + 6.0 * i, -12.0, 0.75, 4.5, 1.8, 1.5, 0.0])
        boxes.append([-30.0 + 6.2 * i, 12.0, 0.75, 4.6, 1.85, 1.5, math.pi])
        boxes.append([20.0 + 2.7 * i, 25.0, 0.75, 4.4, 1.8, 1.5, math.pi / 2])
    return [np.array(b) for b in boxes[:n_cars]]


CROSSING_CARS = [
    # x0, y0, yaw0, v, t_start, segments
    (-65.0, -2.0, 0.0, 12.0, 0.0, [(4.2, 0.0), (math.pi / 2 / (4.0 / 12.0), 4.0 / 12.0)]),  # E then left turn
    (65.0, 2.0, math.pi, 10.0, 1.0, [(20.0, 0.0)]),  # straight westwards
    (2.0, 65.0, -math.pi / 2, 14.0, 5.0, [(3.5, 0.0), (math.pi / 2 / (4.0 / 14.0), -4.0 / 14.0)]),  # S then right
    (-2.0, -65.0, math.pi / 2, 8.0, 1.0, [(6.0, 0.0), (math.pi / 2 / (4.0 / 8.0), 4.0 / 8.0)]),  # N then left
    (60.0, -6.0, math.pi, 11.0, 9.0, [(20.0, 0.0)]),  # straight westwards, later
]


def crossing_gt():
    return [trajectory(*c) for c in CROSSING_CARS]


def gt_box(state, dims=(4.5, 1.8, 1.5)):
    x, y, yaw, _ = state
    return [x, y, 0.75, dims[0], dims[1], dims[2], yaw]


def match_ids(objs_per_frame, gt_tracks, max_dist=2.0):
    """For each GT track the list of (frame, obj) matched by nearest centre."""
    res = []
    for g in gt_tracks:
        m = []
        for f, st in g.items():
            best, bd = None, max_dist
            for o in objs_per_frame[f]:
                d = math.hypot(o["box"][0] - st[0], o["box"][1] - st[1])
                if d < bd:
                    best, bd = o, d
            if best is not None:
                m.append((f, best))
        res.append(m)
    return res


def run_online(frames, **params):
    tr = MultiObjectTracker(dt=DT, **params)
    return [tr.step(d) for d in frames]


# --------------------------------------------------------------------------------------------
# unit tests of helpers
# --------------------------------------------------------------------------------------------


def test_bev_iou_basic():
    a = np.array([0, 0, 0, 4, 2, 1.5, 0.0])
    assert bev_iou(a, a) == pytest.approx(1.0)
    b = a.copy()
    b[0] = 2.0
    assert bev_iou(a, b) == pytest.approx(4.0 / 12.0)
    c = a.copy()
    c[6] = math.pi / 2  # rotated by 90 degrees: intersection 2x2
    assert bev_iou(a, c) == pytest.approx(4.0 / 12.0)
    d = a.copy()
    d[0] = 10
    assert bev_iou(a, d) == 0.0


def test_class_compatibility():
    assert class_penalty("car", "van") == 0.0
    assert class_penalty("truck", "bus") == 0.0
    assert math.isinf(class_penalty("car", "pedestrian"))
    assert math.isinf(class_penalty("bus", "cyclist"))
    assert math.isfinite(class_penalty("cyclist", "pedestrian"))


# --------------------------------------------------------------------------------------------
# scenario 1: parked cars with noisy, flipping boxes
# --------------------------------------------------------------------------------------------


def parked_frames(rng, boxes, **kw):
    return [[noisy_det(rng, b, **kw) for b in boxes] for _ in range(N)]


def test_parked_cars_offline():
    rng = np.random.default_rng(1)
    gt = parked_cars(15)
    frames = parked_frames(rng, gt)
    objs, tracks = track_sequence(frames, dt=DT)

    assert len(tracks) == 15
    for t in tracks.values():
        assert t["group"] == "vehicle" and t["cls"] == "car"
        assert not t["moving"]
        assert t["n_frames"] == N
    gt_tracks = [{f: (b[0], b[1], b[6], 0.0) for f in range(N)} for b in gt]
    for g, m in zip(gt, match_ids(objs, gt_tracks)):
        assert len(m) == N
        assert len({o["id"] for _, o in m}) == 1
        yaws = np.array([o["box"][6] for _, o in m])
        # stable: the box does not change at all over the sequence
        assert np.ptp(wrap(yaws - yaws[0])) < 1e-6
        # correct axis (heading sign of a parked car is not observable)
        assert abs(float(wrap(yaws[0] - g[6], math.pi))) < 0.05
        box = np.array(m[0][1]["box"])
        assert abs(box[3] - 4.5) < 0.3 and abs(box[4] - g[4]) < 0.3
        assert all(not o["moving"] and o["speed"] == 0.0 for _, o in m)


def test_parked_cars_online():
    rng = np.random.default_rng(2)
    gt = parked_cars(15)
    out = run_online(parked_frames(rng, gt))
    ids = {o["id"] for fr in out for o in fr}
    assert len(ids) == 15
    objs = [o for fr in out[20:] for o in fr]
    assert all(not o["moving"] for o in objs)
    # after convergence the axis is right in (almost) every frame
    gt_tracks = [{f: (b[0], b[1], b[6], 0.0) for f in range(20, N)} for b in gt]
    errs = []
    for g, m in zip(gt, match_ids(out, gt_tracks)):
        errs += [abs(float(wrap(o["box"][6] - g[6], math.pi))) for _, o in m]
    assert np.mean(np.array(errs) < 0.1) > 0.98


# --------------------------------------------------------------------------------------------
# scenario 2: crossing / turning cars with missed detections
# --------------------------------------------------------------------------------------------


def crossing_frames(rng, gts, miss_prob=0.2, gaps_per_car=2):
    frames = [[] for _ in range(N)]
    detected = []
    for g in gts:
        fr = sorted(g)
        missing = set(f for f in fr if rng.random() < miss_prob)
        for _ in range(gaps_per_car):  # 2-frame occlusion gaps
            f0 = int(rng.integers(fr[0] + 5, fr[-1] - 5))
            missing |= {f0, f0 + 1}
        detected.append([f for f in fr if f not in missing])
        for f in detected[-1]:
            frames[f].append(noisy_det(rng, gt_box(g[f]), pos_sigma=0.2, flip_prob=0.1))
    return frames, detected


@pytest.mark.parametrize("seed", [3, 4])
def test_crossing_cars(seed):
    rng = np.random.default_rng(seed)
    gts = crossing_gt()
    frames, detected = crossing_frames(rng, gts)

    # online: one id per car, no switches
    out = run_online(frames)
    for m in match_ids(out, gts):
        assert len({o["id"] for _, o in m}) == 1
    assert len({o["id"] for fr in out for o in fr}) == 5

    objs, tracks = track_sequence(frames, dt=DT)
    assert len(tracks) == 5
    assert all(t["moving"] for t in tracks.values())
    rel_err, yaw_err = [], []
    for g, det, m in zip(gts, detected, match_ids(objs, gts)):
        assert len({o["id"] for _, o in m}) == 1
        # present (incl. coasted) from the first to the last detection
        assert len(m) == det[-1] - det[0] + 1
        for f, o in m:
            rel_err.append(abs(o["speed"] - g[f][3]) / g[f][3])
            yaw_err.append(abs(float(wrap(o["box"][6] - g[f][2]))))
            assert o["moving"]
    rel_err, yaw_err = np.array(rel_err), np.array(yaw_err)
    assert np.median(rel_err) < 0.03
    assert np.quantile(rel_err, 0.98) < 0.10
    assert np.quantile(yaw_err, 0.98) < 0.15  # heading incl. sign, rad


# --------------------------------------------------------------------------------------------
# scenario 3: short-lived false positives
# --------------------------------------------------------------------------------------------


def test_false_positives_removed_offline():
    rng = np.random.default_rng(5)
    parked = parked_cars(5)
    gts = crossing_gt()
    frames, _ = crossing_frames(rng, gts)
    for f in range(N):
        frames[f] += [noisy_det(rng, b) for b in parked]
    all_gt = gts + [{f: (b[0], b[1], b[6], 0.0) for f in range(N)} for b in parked]

    n_fp = 0
    for f0 in range(5, 190):
        for _ in range(rng.poisson(1.0)):
            while True:
                xy = rng.uniform(-60, 60, 2)
                if all(math.hypot(xy[0] - g[f][0], xy[1] - g[f][1]) > 5.0 for g in all_gt for f in
                       range(f0, f0 + 6) if f in g):
                    break
            cls = str(rng.choice(["pedestrian", "cyclist", "car"]))
            dims = [0.6, 0.6, 1.7] if cls == "pedestrian" else [1.8, 0.7, 1.7] if cls == "cyclist" else [3.5, 1.7, 1.4]
            score = float(rng.uniform(0.2, 0.7))
            for f in range(f0, min(N, f0 + int(rng.integers(1, 6)))):
                b = [xy[0], xy[1], 0.8, *dims, float(rng.uniform(-3, 3))]
                frames[f].append(noisy_det(rng, b, cls=cls, pos_sigma=0.1, flip_prob=0.0, score=score))
            n_fp += 1
    assert n_fp > 100

    out = run_online(frames)
    assert len({o["id"] for fr in out for o in fr}) > 10  # some FPs got confirmed online

    objs, tracks = track_sequence(frames, dt=DT)
    assert len(tracks) == 10
    matched = match_ids(objs, all_gt)
    ids = set()
    for m in matched:
        ids |= {o["id"] for _, o in m}
        assert len({o["id"] for _, o in m}) == 1
    assert ids == {int(k) for k in tracks}


# --------------------------------------------------------------------------------------------
# scenario 4: fragmented tracks are merged offline
# --------------------------------------------------------------------------------------------


def test_fragment_merging():
    rng = np.random.default_rng(6)
    straight = trajectory(-60.0, -2.0, 0.0, 12.0, 0.0, [(20.0, 0.0)])
    turning = trajectory(-2.0, -65.0, math.pi / 2, 8.0, 0.0, [(6.0, 0.0), (math.pi, 0.5)])
    parked = {f: (10.0, 20.0, 0.3, 0.0) for f in range(N)}
    gaps = [set(range(50, 60)), set(range(78, 88)), set(range(60, 85))]
    frames = [[] for _ in range(N)]
    for g, gap in zip([straight, turning, parked], gaps):
        for f, st in g.items():
            if f not in gap:
                frames[f].append(noisy_det(rng, gt_box(st), pos_sigma=0.15, flip_prob=0.1))

    # a short-memory online tracker fragments every one of them
    out = run_online(frames, max_age=3, max_age_static=3)
    for m in match_ids(out, [straight, turning, parked]):
        assert len({o["id"] for _, o in m}) >= 2

    for kw in ({"max_age": 3, "max_age_static": 3}, {}):
        objs, tracks = track_sequence(frames, dt=DT, **kw)
        assert len(tracks) == 3
        for g, gap, m in zip([straight, turning, parked], gaps, match_ids(objs, [straight, turning, parked])):
            assert len({o["id"] for _, o in m}) == 1
            by_f = dict(m)
            for f in gap:  # the gap is filled with coasted boxes
                if f in g:
                    assert f in by_f and by_f[f]["coasted"]


# --------------------------------------------------------------------------------------------
# scenario 5: class voting with noisy labels
# --------------------------------------------------------------------------------------------


def test_class_voting():
    rng = np.random.default_rng(7)
    objs_def = [
        # true class, dims, trajectory, label distribution (class, prob, score range)
        ("car", (4.5, 1.8, 1.5), trajectory(-60, -2, 0.0, 10.0, 0.0, [(20, 0.0)]),
         [("car", 0.6, (0.6, 0.95)), ("van", 0.25, (0.3, 0.6)), ("truck", 0.15, (0.3, 0.5))]),
        ("bus", (12.0, 2.5, 3.2), trajectory(60, 6, math.pi, 8.0, 0.0, [(20, 0.0)]),
         [("bus", 0.55, (0.6, 0.9)), ("truck", 0.45, (0.3, 0.6))]),
        ("pedestrian", (0.6, 0.6, 1.7), trajectory(20, -20, 1.0, 1.4, 0.0, [(20, 0.0)]),
         [("pedestrian", 0.7, (0.5, 0.9)), ("cyclist", 0.3, (0.3, 0.6))]),
        ("cyclist", (1.8, 0.7, 1.7), trajectory(-20, 30, -0.5, 5.0, 0.0, [(20, 0.0)]),
         [("cyclist", 0.65, (0.5, 0.9)), ("pedestrian", 0.35, (0.3, 0.6))]),
    ]
    frames = [[] for _ in range(N)]
    for true_cls, dims, g, dist in objs_def:
        probs = np.array([d[1] for d in dist])
        for f, st in g.items():
            k = rng.choice(len(dist), p=probs / probs.sum())
            lab, _, (s0, s1) = dist[k]
            frames[f].append(
                noisy_det(rng, gt_box(st, dims), cls=lab, pos_sigma=0.1, size_noise=0.2, flip_prob=0.05,
                          score=float(rng.uniform(s0, s1)))
            )
    objs, tracks = track_sequence(frames, dt=DT)
    assert len(tracks) == 4
    for (true_cls, _, g, _), m in zip(objs_def, match_ids(objs, [o[2] for o in objs_def])):
        ids = {o["id"] for _, o in m}
        assert len(ids) == 1
        assert tracks[str(ids.pop())]["cls"] == true_cls
        assert all(o["cls"] == true_cls for _, o in m)
    groups = sorted(t["group"] for t in tracks.values())
    assert groups == ["vehicle", "vehicle", "vru", "vru"]

    out = run_online(frames)
    last = {o["id"]: o["cls"] for fr in out for o in fr}  # class at the end of each track
    assert sorted(last.values()) == sorted(o[0] for o in objs_def)


# --------------------------------------------------------------------------------------------
# scenario 6: a bus split into two clusters, with a car driving alongside
# --------------------------------------------------------------------------------------------


def test_split_bus_counted_once():
    rng = np.random.default_rng(8)
    bus = trajectory(-60, -2.0, 0.0, 8.0, 0.0, [(20, 0.0)])
    car = trajectory(-55, -5.5, 0.0, 8.0, 0.0, [(20, 0.0)])
    frames = [[] for _ in range(N)]
    for f, st in bus.items():
        x, y, yaw, _ = st
        if rng.random() < 0.25:  # split into front and rear halves
            for s in (-1, 1):
                cx, cy = x + s * 3.0 * math.cos(yaw), y + s * 3.0 * math.sin(yaw)
                frames[f].append(noisy_det(rng, [cx, cy, 1.6, 5.8, 2.5, 3.2, yaw], cls="truck", flip_prob=0.1))
        else:
            frames[f].append(noisy_det(rng, [x, y, 1.6, 12.0, 2.5, 3.2, yaw], cls="bus", flip_prob=0.1))
    for f, st in car.items():
        frames[f].append(noisy_det(rng, gt_box(st), flip_prob=0.1))

    objs, tracks = track_sequence(frames, dt=DT)
    assert len(tracks) == 2
    assert sorted(t["cls"] for t in tracks.values()) == ["bus", "car"]
    for m in match_ids(objs, [bus, car], max_dist=3.5):
        assert len({o["id"] for _, o in m}) == 1


def test_contract_fields():
    rng = np.random.default_rng(9)
    frames = parked_frames(rng, parked_cars(3))
    online = MultiObjectTracker().step(frames[0])
    assert online == []  # nothing confirmed after one frame
    objs, tracks = track_sequence(frames)
    o = objs[0][0]
    assert set(o) == {"id", "cls", "box", "score", "moving", "speed", "n_points", "coasted"}
    assert len(o["box"]) == 7
    t = next(iter(tracks.values()))
    assert set(t) == {"cls", "group", "first", "last", "moving", "max_speed", "n_frames"}
    with pytest.raises(TypeError):
        MultiObjectTracker(not_a_param=1)


# --------------------------------------------------------------------------------------------
# scenario 7: stop-and-go (waits at a red light, then pulls away)
# --------------------------------------------------------------------------------------------


def stop_and_go():
    """Arrives at 10 m/s heading north, brakes at 2.5 m/s^2, waits 6 s, accelerates at 2 m/s^2."""
    out, y, v = {}, -60.0, 10.0
    phase_wait = 0
    for f in range(N):
        out[f] = (3.0, y, math.pi / 2, v)
        if phase_wait == 0:
            v = max(0.0, v - 2.5 * DT) if y > -35.0 else v
            if v == 0.0:
                phase_wait = 1
                wait_start = f
        elif phase_wait == 1 and f - wait_start >= 60:
            phase_wait = 2
        elif phase_wait == 2:
            v = min(10.0, v + 2.0 * DT)
        y += v * DT
    return out


def test_stop_and_go():
    rng = np.random.default_rng(10)
    g = stop_and_go()
    frames = [[noisy_det(rng, gt_box(st), pos_sigma=0.15, flip_prob=0.2)] if rng.random() > 0.1 else []
              for f, st in sorted(g.items())]
    # the online tracker needs a few frames to release the stationary lock when pulling away
    for objs, max_wrong in ((run_online(frames), 8), (track_sequence(frames, dt=DT)[0], 2)):
        m = match_ids(objs, [g])[0]
        assert len({o["id"] for _, o in m}) == 1
        # moving flag agrees with the true speed away from the 1 m/s threshold
        wrong = [f for f, o in m if (g[f][3] > 2.0 and not o["moving"]) or (g[f][3] < 0.2 and o["moving"])]
        assert len(wrong) <= max_wrong
        # heading (incl. sign) right while waiting too
        waiting = [o for f, o in m if g[f][3] == 0.0]
        assert len(waiting) > 40
        assert all(abs(float(wrap(o["box"][6] - math.pi / 2))) < 0.15 for o in waiting)


# --------------------------------------------------------------------------------------------
# scenario 8: sparse vehicles seen as two clusters / with a ghost cluster (real-data failure modes)
# --------------------------------------------------------------------------------------------


def test_split_sparse_car_and_ghost_cluster_counted_once():
    """A sparse car crossing the junction is often detected as two car-sized boxes ~3.5 m apart
    (each half completed to a full car, often rotated by 90 degrees); a truck's cab is detected
    as a separate, sparse, truck-sized box ahead of it.  Each must still be one vehicle."""
    rng = np.random.default_rng(11)
    car = trajectory(-50.0, -2.0, 0.0, 8.0, 0.0, [(3.0, 0.0), (math.pi / 2 / 0.4, 0.4)])
    truck = trajectory(50.0, 3.0, math.pi, 7.0, 0.0, [(20.0, 0.0)])
    frames = [[] for _ in range(N)]
    for f, (x, y, yaw, _) in car.items():
        if 30 <= f <= 90 and rng.random() < 0.5:
            for s, n in ((-1.2, 40), (2.3, 15)):
                b = [x + s * math.cos(yaw), y + s * math.sin(yaw), 0.75, 4.3, 1.8, 1.5, yaw + math.pi / 2]
                frames[f].append(noisy_det(rng, b, pos_sigma=0.3, flip_prob=0.0, n_points=n))
        else:
            frames[f].append(noisy_det(rng, gt_box((x, y, yaw, 0)), pos_sigma=0.3, n_points=60))
    for f, (x, y, yaw, _) in truck.items():
        frames[f].append(noisy_det(rng, [x, y, 1.6, 9.0, 2.5, 3.2, yaw], cls="truck", n_points=300))
        if rng.random() < 0.4:  # ghost: cab cluster completed to a full truck away from the sensor
            gx, gy = x + 8.0 * math.cos(yaw), y + 8.0 * math.sin(yaw)
            frames[f].append(noisy_det(rng, [gx, gy, 1.6, 7.6, 2.4, 3.0, yaw], cls="truck", n_points=30))

    objs, tracks = track_sequence(frames, dt=DT)
    assert len(tracks) == 2
    for m in match_ids(objs, [car, truck], max_dist=3.0):
        assert len({o["id"] for _, o in m}) == 1
