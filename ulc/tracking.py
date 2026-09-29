"""Multi-object tracking of LiDAR detections at a fixed intersection.

The detector produces, per frame, a list of 3D boxes ``[x, y, z, l, w, h, yaw]`` (see
``docs/DATA_FORMAT.md``).  This module turns them into stable tracks so that vehicles can be
counted uniquely and parked vehicles can be told apart from moving ones.

Two entry points
----------------
``MultiObjectTracker.step``
    Online (causal) tracking, one frame at a time.  Returns the confirmed tracks of the frame as
    contract objects ``{"id", "cls", "box", "score", "moving", "speed", "n_points", "coasted"}``.

``track_sequence``
    Offline tracking of a whole recorded sequence.  Runs the online tracker forward, then uses the
    knowledge of the full sequence to merge fragments, remove short-lived false positives,
    remove duplicate tracks of split objects, backfill the tentative period, fill gaps and
    re-estimate every trajectory with a Rauch-Tung-Striebel smoother.  Returns the per-frame
    objects and the ``tracks`` dict of the contract.

Design
------
*Motion.*  Each track runs a two-model IMM (interacting multiple model) filter on the BEV centre
state ``[x, y, vx, vy]``: a *stationary* model (velocity forced to zero, tiny process noise) and a
*constant-velocity* model (white-noise acceleration).  Parked cars are explained by the stationary
model, which keeps their velocity at ~0 despite the heavy centre jitter of partial L-shape fits,
while moving vehicles are explained by the CV model; the mode probability switches within a few
frames when a car pulls away.  A plain CTRV model was not used: turn rates at an urban
intersection are moderate and CV with a sufficiently large acceleration noise follows them, while
CTRV would add a poorly observable state for the (majority) parked objects.

*Stationary lock.*  Once a track has been stationary for ``lock_frames`` frames, its output box is
frozen to the running median of the measured centres (speed 0, ``moving`` False).  The lock is
released when measurements consistently leave the locked position or the CV model takes over.

*Shape.*  ``z, l, w, h`` are robust running medians (a rigid object does not change size).
Lengths are taken as ``max(l, w)`` and widths as ``min(l, w)`` so that the 90 degree
re-parametrisation of a box does not matter.

*Yaw.*  L-shape fitting on partial views is ambiguous by pi/2 and pi.  A track keeps a continuous
*axis* estimate (mod pi): each measured long-axis direction is resolved modulo pi/2 relative to it
(so 90 degree flips cannot drag it) and a decaying vote decides which of the two perpendicular
directions is the long axis (so a wrong initial orientation is corrected).  The heading sign is
taken from the velocity when the object moves fast enough, otherwise it is kept continuous.

*Association.*  Hungarian assignment (``scipy.optimize.linear_sum_assignment``) on a cost
``d_M / gate_sigma + w_iou * (1 - IoU_bev) + class_penalty`` where ``d_M`` is the Mahalanobis
distance between predicted and measured centre.  Its covariance is the IMM prediction
covariance + measurement noise + an isotropic floor that grows with the class size
(buses/trucks) and the track speed, so the gate is tight across the lane and wide along it for
uncertain tracks.  Confirmed tracks are associated first, tentative tracks get the remaining
detections.  Vehicles never match VRUs (cyclists, pedestrians).

*Split, partial and merged vehicles.*  Several detections inside the predicted box of a long
vehicle are fused into one; a matched vehicle detection that is much shorter/longer than the
track is re-anchored at its end that agrees with the prediction.  Unmatched detections that are
explained by a confirmed track (mostly inside it, or a sparse *satellite* touching it in its
lane -- e.g. a truck cab completed to a full truck, the sparse second half of a car) neither
spawn nor feed tracks.  Two confirmed vehicle tracks that overlap, or touch in lockstep while
moving, for several frames are the same vehicle: the younger is deleted.

*Track management.*  tentative -> confirmed after ``min_hits`` hits within ``confirm_window``
frames; confirmed tracks are output as ``coasted`` (predicted box) for up to ``max_coast`` missed
frames and deleted after ``max_age`` missed frames (``max_age_young`` for tracks with few hits,
``max_age_static`` for stationary-locked tracks: parked cars get occluded for a long time).

*Class.*  Majority vote over the track history weighted by ``score * (1 + log1p(n_points))``.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field, fields
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

try:  # the project-wide class -> group mapping, if available
    from ulc.schema import GROUP_OF as _SCHEMA_GROUP_OF  # type: ignore
except Exception:  # pragma: no cover - schema module is optional
    _SCHEMA_GROUP_OF = None

__all__ = [
    "Detection",
    "TrackerParams",
    "MultiObjectTracker",
    "track_sequence",
    "class_group",
    "class_penalty",
    "bev_iou",
]

_PI = math.pi
_TWO_PI = 2.0 * math.pi

# --------------------------------------------------------------------------------------------
# classes
# --------------------------------------------------------------------------------------------

_LOCAL_GROUP_OF = {
    "car": "vehicle",
    "van": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "trailer": "vehicle",
    "motorcycle": "vehicle",
    "cyclist": "vru",
    "escooter": "vru",
    "pedestrian": "vru",
    "other": "other",
}


def class_group(cls: str) -> str:
    """Counting group (``vehicle``, ``vru`` or ``other``) of a fine class."""
    if _SCHEMA_GROUP_OF is not None:
        g = _SCHEMA_GROUP_OF.get(cls)
        if g:
            return g
    return _LOCAL_GROUP_OF.get(cls, "other")


# Association penalties for class pairs that are commonly confused by the detector.  Pairs of the
# same group that are not listed get ``TrackerParams.class_mismatch_penalty``; pairs of different
# groups are incompatible (infinite cost) unless listed here.
_PAIR_PENALTY = {
    frozenset(("car", "van")): 0.0,
    frozenset(("truck", "bus")): 0.0,
    frozenset(("truck", "trailer")): 0.0,
    frozenset(("van", "truck")): 0.15,
    frozenset(("cyclist", "pedestrian")): 0.15,
    frozenset(("cyclist", "escooter")): 0.0,
    frozenset(("escooter", "pedestrian")): 0.1,
    frozenset(("cyclist", "motorcycle")): 0.1,
}


def class_penalty(a: str, b: str, mismatch: float = 0.35) -> float:
    """Association cost added for matching class ``a`` with class ``b`` (``inf`` = forbidden)."""
    if a == b:
        return 0.0
    pen = _PAIR_PENALTY.get(frozenset((a, b)))
    if pen is not None:
        return pen
    if class_group(a) == class_group(b) and class_group(a) != "other":
        return mismatch
    return math.inf


def _class_weight(score: float, n_points: int) -> float:
    return max(float(score), 1e-3) * (1.0 + math.log1p(max(int(n_points), 0)))


def _vote(dets: Iterable["Detection"]) -> str:
    acc: dict[str, float] = defaultdict(float)
    for d in dets:
        acc[d.cls] += _class_weight(d.score, d.n_points)
    return max(acc.items(), key=lambda kv: kv[1])[0] if acc else "other"


# --------------------------------------------------------------------------------------------
# data types and parameters
# --------------------------------------------------------------------------------------------


@dataclass
class Detection:
    """One detected object of one frame."""

    box: np.ndarray  # (7,) [x, y, z, l, w, h, yaw], z = box centre, yaw CCW from +x
    cls: str
    score: float
    n_points: int = 0

    def __post_init__(self) -> None:
        self.box = np.asarray(self.box, dtype=np.float64).reshape(7)
        self.cls = str(self.cls)
        self.score = float(self.score)
        self.n_points = int(self.n_points)

    @classmethod
    def from_obj(cls, obj: dict) -> "Detection":
        """Build from a contract object (``{"box", "cls", "score", "n_points"}``)."""
        return cls(obj["box"], obj["cls"], obj.get("score", 1.0), obj.get("n_points", 0))


DEFAULT_GATES = {
    "car": 2.5,
    "van": 3.0,
    "truck": 4.5,
    "bus": 5.0,
    "trailer": 4.5,
    "motorcycle": 2.5,
    "cyclist": 1.8,
    "escooter": 1.5,
    "pedestrian": 1.2,
    "other": 2.0,
}


@dataclass
class TrackerParams:
    """Tunable parameters.  Distances in m, speeds in m/s, durations in frames."""

    # --- association ---
    gates: dict = field(default_factory=lambda: dict(DEFAULT_GATES))
    """Minimum gate radius (m) per (track) class.  Gating is Mahalanobis: the innovation covariance
    is the predicted position covariance + measurement noise + an isotropic floor of
    ((gate + gate_speed_factor * speed) / gate_sigma)^2 that accounts for class size, split and
    partial detections (larger for buses/trucks)."""
    gate_speed_factor: float = 0.1
    """Extra floor radius per m/s of track speed (s): fast tracks get wider gates."""
    gate_sigma: float = 3.0
    """Gate threshold on the Mahalanobis distance."""
    gate_max_factor: float = 2.5
    """Hard cap on the Euclidean centre distance: gate_max_factor * class gate + speed term."""
    w_iou: float = 0.5
    """Weight of (1 - BEV IoU) in the association cost."""
    class_mismatch_penalty: float = 0.35
    """Cost for same-group class pairs that are not a known confusion pair (e.g. car <-> bus).
    Set to ``inf`` for strict class matching."""

    # --- track management ---
    min_hits: int = 3
    """A tentative track is confirmed after min_hits hits within confirm_window frames."""
    confirm_window: int = 5
    tentative_max_miss: int = 2
    """A tentative track is deleted after this many consecutive misses."""
    max_coast: int = 5
    """A confirmed track is output (``coasted``: True) for up to this many missed frames."""
    max_age: int = 10
    """A confirmed track is deleted after this many consecutive missed frames."""
    max_age_static: int = 30
    """Same for stationary-locked tracks (parked cars occluded by passing traffic)."""
    max_age_young: int = 5
    """Same for young tracks (fewer than young_hits detections): short-lived false positives die
    quickly instead of grabbing unrelated detections; real objects are re-linked offline."""
    young_hits: int = 10
    new_track_min_score: float = 0.3
    """Unmatched detections below this score do not spawn tracks (they may extend tracks)."""
    spawn_overlap: float = 0.2
    """Unmatched detections whose BEV area is covered by more than this fraction by a confirmed
    track neither spawn a new track nor feed a tentative one (split / shadow clusters)."""
    satellite_points_ratio: float = 0.5
    """An unmatched vehicle detection aligned with a confirmed vehicle track, in its lane and
    touching it along its axis (+ satellite_margin) with fewer than this fraction of the track's
    points is treated as part of that vehicle (satellite cluster)."""
    satellite_margin: float = 0.5
    dup_overlap: float = 0.4
    """Two confirmed vehicle tracks whose boxes overlap by more than this fraction of the smaller
    box for dup_frames frames (leaky count) are duplicates: the younger one is deleted."""
    dup_frames: int = 3
    lockstep_speed: float = 2.0
    """Above this speed, vehicles touching in the same lane (satellite detections, pairs of
    tracks moving in lockstep with a velocity difference below lockstep_dv) are the same
    vehicle."""
    lockstep_dv: float = 6.0

    # --- split / partial / merged detections of vehicles ---
    fuse_min_length: float = 6.0
    """Several detections lying inside the predicted box of a confirmed vehicle track at least
    this long (bus, truck) are fused into one measurement (object split into several clusters)."""
    fuse_margin: float = 0.5
    """Margin (m) around the predicted box for the fusion test."""
    anchor_min_dl: float = 1.0
    """If a matched vehicle detection is shorter/longer than the track by more than this (partial
    view, split, merged with a neighbour), its centre is re-anchored: one end of the track box is
    aligned with the corresponding end of the detection (the one closest to the prediction)."""
    anchor_min_dw: float = 0.6
    """Same across the vehicle (width)."""
    anchor_min_samples: int = 5
    """Anchoring only starts once the track has this many shape samples."""

    # --- motion model (IMM: stationary + constant velocity) ---
    sigma_acc_vehicle: float = 3.0
    """CV model white-noise acceleration (m/s^2) for vehicles."""
    sigma_acc_vru: float = 1.5
    """Same for cyclists/pedestrians."""
    sigma_static: float = 0.02
    """Stationary model position random walk per frame (m)."""
    meas_sigma_base: float = 0.25
    """Centre measurement noise (m) = meas_sigma_base + meas_sigma_per_m * track length."""
    meas_sigma_per_m: float = 0.04
    init_vel_sigma_vehicle: float = 8.0
    """Initial velocity uncertainty (m/s) of a new vehicle track."""
    init_vel_sigma_vru: float = 2.5
    imm_p_stay: float = 0.95
    """IMM probability of staying in the same motion mode from one frame to the next."""

    # --- moving / stationary ---
    moving_on: float = 1.2
    """A track becomes ``moving`` when its speed exceeds moving_on ..."""
    moving_off: float = 0.8
    """... and stops being ``moving`` when it drops below moving_off (hysteresis around 1 m/s)."""
    lock_frames: int = 10
    """Frames of stationarity after which the box is frozen (stationary lock)."""
    lock_speed: float = 0.4
    """Max filtered speed for a frame to count as stationary."""
    lock_mu: float = 0.7
    """Min IMM stationary-mode probability for a frame to count as stationary."""
    lock_window: int = 100
    """The locked centre is the median of the last lock_window measured centres."""
    unlock_dist: float = 1.2
    """A lock is released after unlock_frames consecutive measurements farther than this from the
    locked centre while the IMM favours the moving model, or when the moving model clearly takes
    over (probability > 0.9 and speed > moving_on)."""
    unlock_frames: int = 3

    # --- shape ---
    dims_window: int = 60
    """Number of recent measurements used for the running medians of z, l, w, h."""
    yaw_gain: float = 0.4
    """Gain of the axis (yaw mod pi) update for non-locked tracks."""
    yaw_gain_static: float = 0.05
    """Same for stationary-locked tracks."""
    flip_gain: float = 0.15
    """Gain of the decaying long-axis vote (resolves 90 degree ambiguity)."""
    flip_init: float = 0.25
    """Initial trust in the first detection's long axis."""
    yaw_vel_min_speed: float = 2.0
    """Above this speed the heading of a vehicle is aligned with the velocity direction (for
    cyclists/pedestrians: above moving_on)."""

    # --- offline post-processing (track_sequence) ---
    min_track_frames: int = 12
    """Tracks are dropped unless (a) one of the online tracks they are made of has at least
    min_track_frames detections (merging extends tracks, it does not create them) and (b) they
    have min_track_frames detections within some window of core_window frames (a real object is
    seen densely at least once, a flickering false positive is not) ..."""
    core_window: int = 25
    min_track_frames_edge: int = 3
    """... unless they touch the sequence start/end or the ROI border and have at least this many."""
    edge_frames: int = 3
    """A track 'touches' the sequence start/end if it starts/ends within this many frames of it."""
    roi: tuple | None = None
    """Optional (xmin, xmax, ymin, ymax); tracks starting/ending within roi_margin of the border
    are allowed to be short."""
    roi_margin: float = 5.0
    min_mean_score: float = 0.35
    """Tracks with a lower mean detection score are dropped."""
    min_hit_ratio: float = 0.5
    """Tracks detected in less than this fraction of their lifetime are dropped."""
    max_merge_gap: int = 30
    """Max gap (frames) between the end of a fragment and the start of its continuation."""
    max_merge_gap_static: int = 300
    """Same for two stationary fragments at the same spot (a parked car occluded for long)."""
    merge_dist: float = 2.5
    """Max extrapolation error (m) for merging two fragments ..."""
    merge_dist_per_s: float = 1.5
    """... plus this much per second of gap."""
    extend_static: bool = False
    """Object permanence: vehicle tracks that are stationary over their whole lifetime (parked)
    are extended to the start/end of the sequence (``coasted``) unless another vehicle track
    occupies their spot.  Off by default: it helps counting parked cars that become occluded but
    also extends static false positives (lower precision on the real data)."""
    max_merge_overlap: int = 10
    """Max overlap (frames) of two fragments that can still be merged (hand-over after a drift)."""
    merge_dist_overlap: float = 2.0
    """Max distance (m) between two overlapping fragments at the hand-over."""
    merge_dist_static: float = 0.8
    """Max centre distance for merging two stationary fragments is
    merge_dist_static + merge_dist_static_per_m * length (car ~1.7 m, pedestrian ~0.9 m)."""
    merge_dist_static_per_m: float = 0.2
    dedup_margin: float = 0.5
    """A vehicle track whose centre lies inside a longer vehicle track's box (+ margin) ..."""
    dedup_fraction: float = 0.6
    """... in at least this fraction of their co-detected frames is the same object: it is merged
    into the longer track (split buses/trucks, track take-overs)."""
    static_radius: float = 0.8
    """A track is stationary for the whole sequence if 90 % of its centres lie within
    static_radius + static_radius_per_m * length of their median."""
    static_radius_per_m: float = 0.1
    static_speed: float = 0.5
    """Smoothed speed below which a frame belongs to a stationary segment."""
    min_static_frames: int = 10
    """Minimum length of a stationary segment (shorter ones are treated as moving)."""
    smooth_sigma_acc_vehicle: float = 2.0
    """White-noise acceleration of the RTS smoother for vehicles."""
    smooth_sigma_acc_vru: float = 1.0
    yaw_smooth_frames: float = 2.0
    """Gaussian sigma (frames) for smoothing the box-vs-velocity yaw deviation offline."""
    max_yaw_dev: float = 0.5
    """Max |box yaw - velocity direction| (rad) accepted from a detection of a moving object."""

    @classmethod
    def from_kwargs(cls, **kw) -> "TrackerParams":
        names = {f.name for f in fields(cls)}
        unknown = set(kw) - names
        if unknown:
            raise TypeError(f"unknown tracker parameter(s): {sorted(unknown)}")
        p = cls(**kw)
        if "gates" in kw:  # partial override of the gate table
            p.gates = {**DEFAULT_GATES, **kw["gates"]}
        return p


# --------------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------------


def _wrap(a, period: float = _TWO_PI):
    """Wrap angle(s) into [-period/2, period/2)."""
    return (np.asarray(a) + period / 2.0) % period - period / 2.0


def _wrapf(a: float, period: float = _TWO_PI) -> float:
    return (a + period / 2.0) % period - period / 2.0


def _long_axis(box: np.ndarray) -> float:
    """Direction of the box's long side, mod pi, in [-pi/2, pi/2)."""
    yaw = box[6] if box[3] >= box[4] else box[6] + _PI / 2.0
    return _wrapf(float(yaw), _PI)


def _corners(box) -> list[tuple[float, float]]:
    x, y, l, w, yaw = box[0], box[1], box[3], box[4], box[6]
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = l / 2.0, w / 2.0
    return [
        (x + c * px - s * py, y + s * px + c * py)
        for px, py in ((dx, dy), (-dx, dy), (-dx, -dy), (dx, -dy))
    ]


def _poly_area(poly) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def _clip(subject, clipper):
    """Sutherland-Hodgman clipping of a convex polygon by a convex CCW polygon."""
    out = subject
    n = len(clipper)
    for i in range(n):
        if not out:
            break
        ax, ay = clipper[i]
        bx, by = clipper[(i + 1) % n]
        inp, out = out, []

        def side(p):
            return (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)

        m = len(inp)
        for j in range(m):
            p, q = inp[j], inp[(j + 1) % m]
            sp, sq = side(p), side(q)
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def _bev_intersection(a, b) -> float:
    return _poly_area(_clip(_corners(a), _corners(b)))


def _extents(boxes, centre, axis: float):
    """(umin, umax, vmin, vmax) of the corners of ``boxes`` along / across ``axis`` from ``centre``."""
    cs, sn = math.cos(axis), math.sin(axis)
    pts = np.array([c for b in boxes for c in _corners(b)]) - np.asarray(centre)[:2]
    u = pts @ np.array([cs, sn])
    v = pts @ np.array([-sn, cs])
    return float(u.min()), float(u.max()), float(v.min()), float(v.max())


def bev_iou(a, b) -> float:
    """Bird's-eye-view IoU of two boxes ``[x, y, z, l, w, h, yaw]``."""
    inter = _bev_intersection(a, b)
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / union if union > 1e-9 else 0.0


# --------------------------------------------------------------------------------------------
# IMM filter (stationary + constant velocity) on [x, y, vx, vy]
# --------------------------------------------------------------------------------------------


def _cv_q(dt: float, sigma_acc: float) -> np.ndarray:
    g = np.array([[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]]) * sigma_acc**2
    q = np.zeros((4, 4))
    q[np.ix_([0, 2], [0, 2])] = g
    q[np.ix_([1, 3], [1, 3])] = g
    return q


class _IMM:
    """Two-model IMM.  Model 0: stationary (velocity reset to 0).  Model 1: constant velocity."""

    def __init__(self, xy, r_var, vel_sigma, dt, sigma_acc, sigma_static, p_stay):
        f1 = np.eye(4)
        f1[0, 2] = f1[1, 3] = dt
        self.F = (np.diag([1.0, 1.0, 0.0, 0.0]), f1)
        self.Q = (np.diag([sigma_static**2, sigma_static**2, 1e-4, 1e-4]), _cv_q(dt, sigma_acc))
        self.PI = np.array([[p_stay, 1 - p_stay], [1 - p_stay, p_stay]])
        x0 = np.array([xy[0], xy[1], 0.0, 0.0])
        self.x = [x0.copy(), x0.copy()]
        self.P = [np.diag([r_var, r_var, 1e-4, 1e-4]), np.diag([r_var, r_var, vel_sigma**2, vel_sigma**2])]
        self.mu = np.array([0.5, 0.5])
        self._combine()

    def _combine(self):
        self.xc = self.mu[0] * self.x[0] + self.mu[1] * self.x[1]
        P = np.zeros((4, 4))
        for j in range(2):
            d = self.x[j] - self.xc
            P += self.mu[j] * (self.P[j] + d[:, None] * d[None, :])
        self.Pc = P

    def predict(self):
        c = self.PI.T @ self.mu
        mix = self.PI * self.mu[:, None] / c[None, :]
        xs, Ps = [], []
        for j in range(2):
            x0 = mix[0, j] * self.x[0] + mix[1, j] * self.x[1]
            P0 = np.zeros((4, 4))
            for i in range(2):
                d = self.x[i] - x0
                P0 += mix[i, j] * (self.P[i] + d[:, None] * d[None, :])
            F = self.F[j]
            xs.append(F @ x0)
            Ps.append(F @ P0 @ F.T + self.Q[j])
        self.x, self.P, self.mu = xs, Ps, c
        self._combine()

    def update(self, z, r_var):
        ll = np.zeros(2)
        for j in range(2):
            x, P = self.x[j], self.P[j]
            S = P[:2, :2] + np.eye(2) * r_var
            nu = np.asarray(z[:2], float) - x[:2]
            Si = np.linalg.inv(S)
            K = P[:, :2] @ Si
            self.x[j] = x + K @ nu
            P = P - K @ S @ K.T
            self.P[j] = 0.5 * (P + P.T)
            det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
            ll[j] = -0.5 * float(nu @ Si @ nu) - 0.5 * math.log(max(det, 1e-12)) - math.log(_TWO_PI)
        lw = np.log(np.maximum(self.mu, 1e-12)) + ll
        lw -= lw.max()
        mu = np.exp(lw)
        mu /= mu.sum()
        self.mu = np.clip(mu, 1e-3, 1 - 1e-3)
        self.mu /= self.mu.sum()
        self._combine()

    def reset_velocity(self, vel_sigma):
        """Forget the motion state (used when a stationary lock is released)."""
        self.mu = np.array([0.3, 0.7])
        self.P[1][2:, 2:] = np.eye(2) * vel_sigma**2
        self._combine()

    @property
    def xy(self):
        return self.xc[:2]

    @property
    def vel(self):
        return self.xc[2:4]


# --------------------------------------------------------------------------------------------
# online tracker
# --------------------------------------------------------------------------------------------


class _Track:
    """State of one track: IMM filter, robust shape, yaw/axis estimate, class votes, history."""

    def __init__(self, tid: int, det: Detection, frame: int, p: TrackerParams, dt: float):
        self.id = tid
        self.p = p
        self.birth = frame
        self.meas: dict[int, Detection] = {}
        self.axes: dict[int, float] = {}  # frame -> filtered long axis after the update
        self.hits = 0
        self.tsu = 0  # frames since last update
        self.confirmed = False
        self.class_w: dict[str, float] = defaultdict(float)
        self.shape = deque(maxlen=p.dims_window)  # (l, w, h, z)
        self._dims = None
        self.score_ema = det.score
        self.moving = False
        self.locked = False
        self.static_run = 0
        self.static_buf: list[np.ndarray] = []
        self.lock_buf = deque(maxlen=p.lock_window)
        self.lock_centre = None
        self.dev_count = 0
        self.axis = _long_axis(det.box)
        b = det.box
        self.heading = _wrapf(float(b[6] if b[3] >= b[4] else b[6] + _PI / 2))
        self.flip_score = p.flip_init
        self.last_n_points = det.n_points
        grp = class_group(det.cls)
        veh = grp == "vehicle"
        l0 = max(det.box[3], det.box[4])
        self.imm = _IMM(
            det.box[:2],
            self._r_var(l0),
            p.init_vel_sigma_vehicle if veh else p.init_vel_sigma_vru,
            dt,
            p.sigma_acc_vehicle if veh else p.sigma_acc_vru,
            p.sigma_static,
            p.imm_p_stay,
        )
        self._absorb(det, frame)
        self.axes[frame] = self.axis

    # --- properties ---
    @property
    def cls(self) -> str:
        return max(self.class_w.items(), key=lambda kv: kv[1])[0]

    @property
    def group(self) -> str:
        return class_group(self.cls)

    @property
    def dims(self) -> np.ndarray:
        """Robust (l, w, h, z): running medians, cached until the next measurement."""
        if self._dims is None:
            self._dims = np.median(np.asarray(self.shape), axis=0)
        return self._dims

    @property
    def speed(self) -> float:
        return 0.0 if self.locked else float(np.hypot(*self.imm.vel))

    def _r_var(self, length: float) -> float:
        return (self.p.meas_sigma_base + self.p.meas_sigma_per_m * length) ** 2

    def centre(self) -> np.ndarray:
        if self.locked and self.lock_centre is not None:
            return self.lock_centre.copy()
        return self.imm.xy.copy()

    def box(self) -> np.ndarray:
        l, w, h, z = self.dims
        c = self.centre()
        return np.array([c[0], c[1], z, l, w, h, self.heading])

    def gate(self):
        """(inverse innovation covariance (2, 2), hard Euclidean cap) for association."""
        p = self.p
        base = p.gates.get(self.cls, 2.5)
        spd = p.gate_speed_factor * self.speed
        floor = (base + spd) / p.gate_sigma
        S = self.imm.Pc[:2, :2] + np.eye(2) * (floor**2 + self._r_var(self.dims[0]))
        return np.linalg.inv(S), p.gate_max_factor * base + spd

    # --- filter steps ---
    def predict(self):
        self.imm.predict()
        self.tsu += 1
        self._update_moving()
        self._update_heading()

    def update(self, det: Detection, frame: int):
        l = self.dims[0]
        xy, anchored = self._anchored_centre(det)
        if anchored:
            b = det.box.copy()
            b[:2] = xy
            det = Detection(b, det.cls, det.score, det.n_points)
        self.imm.update(xy, self._r_var(l) * (2.0 if anchored else 1.0))
        self.tsu = 0
        self._absorb(det, frame)
        self._update_lock(det.box[:2])
        self._update_moving()
        self._update_yaw(det)
        self.axes[frame] = self.axis

    def _anchored_centre(self, det: Detection):
        """Measured centre, corrected for partial / split / merged vehicle detections.

        If the detection is clearly shorter or longer (narrower or wider) than the track's robust
        dimensions, one end of the track box is aligned with the corresponding end of the
        detection -- the end whose alignment agrees best with the prediction.  If neither end is
        clearly better (symmetric shrink), the raw centre is kept.
        """
        p = self.p
        b = det.box
        if class_group(det.cls) != "vehicle" or self.group != "vehicle" or len(self.shape) < p.anchor_min_samples:
            return b[:2].copy(), False
        if abs(_wrapf(_long_axis(b) - self.axis, _PI)) >= _PI / 4:  # ambiguous L-shape, keep raw
            return b[:2].copy(), False
        l, w = self.dims[:2]
        c0 = self.imm.xy
        umin, umax, vmin, vmax = _extents([b], c0, self.axis)
        u, v = 0.5 * (umin + umax), 0.5 * (vmin + vmax)
        changed = False
        for lo, hi, size, thr, which in ((umin, umax, l, p.anchor_min_dl, "u"), (vmin, vmax, w, p.anchor_min_dw, "v")):
            d = abs((hi - lo) - size) / 2.0
            if 2.0 * d <= thr:
                continue
            c1, c2 = sorted((lo + size / 2.0, hi - size / 2.0), key=abs)
            if abs(c2) - abs(c1) > d:
                changed = True
                if which == "u":
                    u = c1
                else:
                    v = c1
        if not changed:
            return b[:2].copy(), False
        cs, sn = math.cos(self.axis), math.sin(self.axis)
        return np.array([c0[0] + u * cs - v * sn, c0[1] + u * sn + v * cs]), True

    def _absorb(self, det: Detection, frame: int):
        self.meas[frame] = det
        self.hits += 1
        b = det.box
        self.shape.append((max(b[3], b[4]), min(b[3], b[4]), b[5], b[2]))
        self._dims = None
        self.class_w[det.cls] += _class_weight(det.score, det.n_points)
        self.score_ema = 0.7 * self.score_ema + 0.3 * det.score
        self.last_n_points = det.n_points

    def _update_lock(self, xy):
        p = self.p
        xy = np.asarray(xy, float)
        spd = float(np.hypot(*self.imm.vel))
        if self.locked:
            self.lock_buf.append(xy)
            c = self.lock_centre = np.median(np.asarray(self.lock_buf), axis=0)
            self.dev_count = self.dev_count + 1 if np.hypot(*(xy - c)) > p.unlock_dist else 0
            if self.dev_count >= p.unlock_frames or (
                self.imm.mu[1] > 0.9 and spd > p.moving_on
            ):
                self.locked = False
                self.static_run = 0
                self.static_buf = []
                self.dev_count = 0
                veh = self.group == "vehicle"
                self.imm.reset_velocity(p.init_vel_sigma_vehicle if veh else p.init_vel_sigma_vru)
            return
        if self.imm.mu[0] > p.lock_mu and spd < p.lock_speed:
            self.static_run += 1
            self.static_buf.append(xy)
        else:
            self.static_run = 0
            self.static_buf = []
        if self.static_run >= p.lock_frames:
            self.locked = True
            self.lock_buf.clear()
            self.lock_buf.extend(self.static_buf)
            self.lock_centre = np.median(np.asarray(self.lock_buf), axis=0)
            self.static_buf = []

    def _update_moving(self):
        s = self.speed
        if self.moving and s < self.p.moving_off:
            self.moving = False
        elif not self.moving and s > self.p.moving_on:
            self.moving = True

    def _update_yaw(self, det: Detection):
        p = self.p
        alpha = _long_axis(det.box)
        gain = p.yaw_gain_static if self.locked else p.yaw_gain
        self.axis = _wrapf(self.axis + gain * _wrapf(alpha - self.axis, _PI / 2), _PI)
        vote = 1.0 if abs(_wrapf(alpha - self.axis, _PI)) < _PI / 4 else -1.0
        self.flip_score += p.flip_gain * (vote - self.flip_score)
        v = self.imm.vel
        if not self.locked and self.speed > self._vel_min_speed():
            vdir = math.atan2(v[1], v[0])
            vote = 1.0 if abs(_wrapf(vdir - self.axis, _PI)) < _PI / 4 else -1.0
            self.flip_score += p.flip_gain * (vote - self.flip_score)
        if self.flip_score < 0:
            self.axis = _wrapf(self.axis + _PI / 2, _PI)
            self.flip_score = -self.flip_score
        self._update_heading()

    def _vel_min_speed(self) -> float:
        return self.p.yaw_vel_min_speed if self.group == "vehicle" else self.p.moving_on

    def _update_heading(self):
        if not self.locked and self.speed > self._vel_min_speed():
            v = self.imm.vel
            ref = math.atan2(v[1], v[0])
        else:
            ref = self.heading
        a = self.axis
        cand = a if abs(_wrapf(a - ref)) <= _PI / 2 else a + _PI
        self.heading = _wrapf(cand)

    def to_obj(self) -> dict:
        coasted = self.tsu > 0
        return {
            "id": int(self.id),
            "cls": self.cls,
            "box": [float(v) for v in self.box()],
            "score": float(self.score_ema * (0.9**self.tsu)),
            "moving": bool(self.moving),
            "speed": float(self.speed),
            "n_points": 0 if coasted else int(self.last_n_points),
            "coasted": bool(coasted),
        }


class MultiObjectTracker:
    """Online multi-object tracker (see the module docstring for the design).

    Parameters
    ----------
    dt:
        Frame period in seconds (0.1 for the 10 Hz UrbanIng-V2X LiDARs).
    **params:
        Overrides of :class:`TrackerParams` fields.
    """

    def __init__(self, dt: float = 0.1, **params):
        self.dt = float(dt)
        self.params = TrackerParams.from_kwargs(**params)
        self.tracks: list[_Track] = []  # alive tracks
        self.all_tracks: list[_Track] = []  # every track ever created (for offline processing)
        self.frame = 0
        self._next_id = 1
        self._dup_count: dict[tuple[int, int], int] = {}

    # ------------------------------------------------------------------
    def step(self, detections: list[Detection]) -> list[dict]:
        """Process one frame.  Returns the confirmed tracks as contract objects."""
        p = self.params
        f = self.frame
        dets = [d if isinstance(d, Detection) else Detection.from_obj(d) for d in detections]
        dets = [d for d in dets if np.all(np.isfinite(d.box))]

        for t in self.tracks:
            t.predict()

        dets = self._fuse_split(dets)

        confirmed = [t for t in self.tracks if t.confirmed]
        tentative = [t for t in self.tracks if not t.confirmed]
        free = list(range(len(dets)))
        matches1, free = self._associate(confirmed, dets, free)
        for t, j in matches1:
            t.update(dets[j], f)
        # detections explained by a confirmed track (split / shadow / satellite clusters of the
        # same vehicle) neither feed tentative tracks nor spawn new ones
        free = [j for j in free if not self._covered(dets[j])]
        matches2, free = self._associate(tentative, dets, free)
        for t, j in matches2:
            t.update(dets[j], f)
        self._suppress_duplicates()

        # spawn new tracks
        for j in free:
            d = dets[j]
            if d.score < p.new_track_min_score:
                continue
            t = _Track(self._next_id, d, f, p, self.dt)
            self._next_id += 1
            self.tracks.append(t)
            self.all_tracks.append(t)

        # confirmation and deletion
        alive = []
        for t in self.tracks:
            if not t.confirmed:
                recent = sum(1 for k in range(f - p.confirm_window + 1, f + 1) if k in t.meas)
                if recent >= p.min_hits:
                    t.confirmed = True
                elif t.tsu > p.tentative_max_miss:
                    continue
            else:
                if t.locked:
                    max_age = p.max_age_static
                elif t.hits < p.young_hits:
                    max_age = p.max_age_young
                else:
                    max_age = p.max_age
                if t.tsu > max_age:
                    continue
            alive.append(t)
        self.tracks = alive

        self.frame += 1
        return [t.to_obj() for t in self.tracks if t.confirmed and t.tsu <= p.max_coast]

    # ------------------------------------------------------------------
    def _associate(self, tracks: list[_Track], dets: list[Detection], free: list[int]):
        if not tracks or not free:
            return [], free
        p = self.params
        tb = [t.box() for t in tracks]
        txy = np.array([t.imm.xy for t in tracks])
        dxy = np.array([dets[j].box[:2] for j in free])
        nu = dxy[None, :, :] - txy[:, None, :]
        dist = np.linalg.norm(nu, axis=2)
        g = [t.gate() for t in tracks]
        Si = np.array([x[0] for x in g])
        cap = np.array([x[1] for x in g])
        maha = np.sqrt(np.einsum("tdi,tij,tdj->td", nu, Si, nu))
        big = 1e6
        C = np.full(dist.shape, big)
        for i, j in zip(*np.nonzero((maha <= p.gate_sigma) & (dist <= cap[:, None]))):
            t, d = tracks[i], dets[free[j]]
            pen = class_penalty(t.cls, d.cls, p.class_mismatch_penalty)
            if not math.isfinite(pen):
                continue
            reach = 0.5 * (max(tb[i][3], tb[i][4]) + max(d.box[3], d.box[4]))
            iou = bev_iou(tb[i], d.box) if dist[i, j] < reach else 0.0
            C[i, j] = maha[i, j] / p.gate_sigma + p.w_iou * (1.0 - iou) + pen
        rows, cols = linear_sum_assignment(C)
        matches, used = [], set()
        for i, j in zip(rows, cols):
            if C[i, j] < big:
                matches.append((tracks[i], free[j]))
                used.add(free[j])
        return matches, [j for j in free if j not in used]

    def _fuse_split(self, dets: list[Detection]) -> list[Detection]:
        """Fuse several detections lying inside the predicted box of one long vehicle track.

        Buses and trucks are often split into two clusters (occlusion, windows, articulation).
        The parts are replaced by one detection spanning their extents along the track axis.
        """
        p = self.params
        big = [
            t for t in self.tracks
            if t.confirmed and t.tsu <= 2 and t.group == "vehicle" and t.dims[0] >= p.fuse_min_length
        ]
        if not big or len(dets) < 2:
            return dets
        used: set[int] = set()
        fused = []
        for t in sorted(big, key=lambda t: -t.dims[0]):
            l, w = t.dims[:2]
            c0 = t.imm.xy
            cs, sn = math.cos(t.axis), math.sin(t.axis)
            members = []
            for j, d in enumerate(dets):
                if j in used or class_group(d.cls) != "vehicle":
                    continue
                rx, ry = d.box[0] - c0[0], d.box[1] - c0[1]
                u, v = rx * cs + ry * sn, -rx * sn + ry * cs
                if (abs(u) <= l / 2 + p.fuse_margin and abs(v) <= w / 2 + p.fuse_margin
                        and max(d.box[3], d.box[4]) < 0.8 * l):
                    members.append(j)
            if len(members) < 2:
                continue
            parts = [dets[j] for j in members]
            umin, umax, vmin, vmax = _extents([d.box for d in parts], c0, t.axis)
            uc, vc = 0.5 * (umin + umax), 0.5 * (vmin + vmax)
            bottom = min(d.box[2] - d.box[5] / 2 for d in parts)
            top = max(d.box[2] + d.box[5] / 2 for d in parts)
            box = [
                c0[0] + uc * cs - vc * sn, c0[1] + uc * sn + vc * cs, 0.5 * (bottom + top),
                umax - umin, vmax - vmin, top - bottom, t.heading,
            ]
            fused.append(Detection(box, _vote(parts), max(d.score for d in parts), sum(d.n_points for d in parts)))
            used.update(members)
        if not used:
            return dets
        return [d for j, d in enumerate(dets) if j not in used] + fused

    def _covered(self, d: Detection) -> bool:
        """True if an unmatched detection is explained by an existing confirmed track.

        Either it lies mostly inside the track's box (``spawn_overlap``), or it is a *satellite*
        of a vehicle track: aligned with it, in its lane (lateral offset within the half width),
        touching it along its axis and with far fewer points than the track's own detection --
        the typical second cluster of a split bus/truck whose box was completed away from the
        sensor, or the sparse second half of a car.
        """
        p = self.params
        area = d.box[3] * d.box[4]
        if area <= 0:
            return False
        g = class_group(d.cls)
        ld = max(d.box[3], d.box[4])
        for t in self.tracks:
            if not t.confirmed or t.tsu > p.max_coast or t.group != g:
                continue
            b = t.box()
            dx, dy = d.box[0] - b[0], d.box[1] - b[1]
            reach = 0.5 * (b[3] + ld)
            if math.hypot(dx, dy) >= reach + p.satellite_margin:
                continue
            if _bev_intersection(b, d.box) / area > p.spawn_overlap:
                return True
            if g != "vehicle" or t.tsu > 0:
                continue
            # a moving vehicle cannot have another one touching it in its lane; a static one can
            # (queues, parking), so there the detection must also be much sparser
            if t.speed < p.lockstep_speed and d.n_points > p.satellite_points_ratio * t.last_n_points:
                continue
            cs, sn = math.cos(t.axis), math.sin(t.axis)
            u, v = abs(dx * cs + dy * sn), abs(-dx * sn + dy * cs)
            if v <= b[4] and u <= reach + p.satellite_margin:  # in its lane, touching it
                return True
        return False

    def _suppress_duplicates(self):
        """Delete the younger of two confirmed vehicle tracks that are the same vehicle.

        Two vehicles cannot overlap, and two vehicles moving at speed cannot touch each other in
        the same lane.  Tracks that overlap (or, when moving together, touch in lockstep) for
        ``dup_frames`` consecutive frames are the same vehicle seen as two clusters (sparse or
        split objects).  The history of the removed track is kept for the offline pass, which
        merges it into the surviving one.
        """
        p = self.params
        conf = [t for t in self.tracks if t.confirmed and t.tsu <= 1 and t.group == "vehicle"]
        boxes = [t.box() for t in conf]
        still = set()
        for i in range(len(conf)):
            for j in range(i + 1, len(conf)):
                a, b = boxes[i], boxes[j]
                dx, dy = b[0] - a[0], b[1] - a[1]
                reach = 0.5 * (a[3] + b[3])
                if math.hypot(dx, dy) >= reach + p.satellite_margin:
                    continue
                inter = _bev_intersection(a, b)
                same = inter / max(min(a[3] * a[4], b[3] * b[4]), 1e-6) > p.dup_overlap
                if not same:
                    ta, tb = conf[i], conf[j]
                    va, vb = ta.imm.vel, tb.imm.vel
                    if min(ta.speed, tb.speed) > p.lockstep_speed and np.hypot(*(va - vb)) < p.lockstep_dv:
                        cs, sn = math.cos(ta.axis), math.sin(ta.axis)
                        u, v = abs(dx * cs + dy * sn), abs(-dx * sn + dy * cs)
                        same = v <= a[4] / 2 + p.satellite_margin and u <= reach + p.satellite_margin
                if not same:
                    continue
                key = (min(conf[i].id, conf[j].id), max(conf[i].id, conf[j].id))
                still.add(key)
                self._dup_count[key] = self._dup_count.get(key, 0) + 1
                if self._dup_count[key] >= p.dup_frames:
                    ta, tb = conf[i], conf[j]
                    loser = ta if (ta.hits, -ta.id) < (tb.hits, -tb.id) else tb
                    loser.tsu = 10**9  # removed in the management step
        # leaky evidence: a frame where the pair does not look like one vehicle only decays it
        # (sparse detections make the relation flicker)
        alive = {t.id for t in self.tracks}
        self._dup_count = {
            k: (v if k in still else v - 0.5)
            for k, v in self._dup_count.items()
            if (k in still or v > 0.5) and k[0] in alive and k[1] in alive
        }


# --------------------------------------------------------------------------------------------
# offline post-processing
# --------------------------------------------------------------------------------------------


class _Frag:
    """A (possibly merged) track fragment: its detections keyed by frame."""

    dt = 0.1  # frame period, set by track_sequence

    def __init__(self, meas: dict[int, Detection], p: TrackerParams, axes: dict[int, float] | None = None):
        self.meas = dict(meas)
        self.axes = dict(axes or {})  # frame -> filtered long-axis direction of the online track
        self.p = p
        self.core = len(self.meas)  # detections of the largest online track it is made of
        self.refresh()

    def refresh(self):
        p = self.p
        self.frames = np.array(sorted(self.meas), dtype=int)
        self.dets = [self.meas[k] for k in self.frames]
        self.first, self.last = int(self.frames[0]), int(self.frames[-1])
        self.n = len(self.frames)
        self.xy = np.array([d.box[:2] for d in self.dets])
        self.scores = np.array([d.score for d in self.dets])
        self.cls = _vote(self.dets)
        self.group = class_group(self.cls)
        boxes = np.array([d.box for d in self.dets])
        self.l = float(np.median(np.maximum(boxes[:, 3], boxes[:, 4])))
        self.w = float(np.median(np.minimum(boxes[:, 3], boxes[:, 4])))
        self.h = float(np.median(boxes[:, 5]))
        self.med = np.median(self.xy, axis=0)
        r = np.linalg.norm(self.xy - self.med, axis=1)
        radius = p.static_radius + p.static_radius_per_m * self.l
        self.static = bool(np.quantile(r, 0.9) <= radius)
        self._speed = None
        self.p_start, self.v_start = self._endpoint(start=True)
        self.p_end, self.v_end = self._endpoint(start=False)

    def _endpoint(self, start: bool):
        """Position and velocity (m/frame) at the first/last detection (line fit)."""
        if self.static:
            return self.med.copy(), np.zeros(2)
        k = min(10, self.n)
        fr = self.frames[:k] if start else self.frames[-k:]
        xy = self.xy[:k] if start else self.xy[-k:]
        t0 = fr[0] if start else fr[-1]
        if k < 3 or fr[-1] - fr[0] < 2:
            return xy[0 if start else -1].copy(), np.zeros(2)
        A = np.stack([np.ones(k), fr - t0], axis=1)
        sol, *_ = np.linalg.lstsq(A, xy, rcond=None)
        return sol[0], sol[1]

    def speed_at(self, frame: int) -> float:
        """Speed (m/s) at ``frame`` from an RTS-smoothed trajectory of the detections (cached)."""
        if getattr(self, "_speed", None) is None:
            p = self.p
            veh = self.group == "vehicle"
            r = (p.meas_sigma_base + p.meas_sigma_per_m * self.l) ** 2
            _, vel = _rts_cv(
                self.frames - self.first, self.xy, np.full(self.n, r), self.last - self.first + 1,
                self.dt, p.smooth_sigma_acc_vehicle if veh else p.smooth_sigma_acc_vru,
            )
            self._speed = np.linalg.norm(vel, axis=1)
        return float(self._speed[min(max(frame - self.first, 0), self.last - self.first)])

    def pos_at(self, frame: int) -> np.ndarray:
        """Detected position at the last detection at or before ``frame`` (else the first)."""
        i = int(np.searchsorted(self.frames, frame, side="right")) - 1
        return self.xy[max(i, 0)]

    @property
    def mean_score(self) -> float:
        return float(self.scores.mean())


def _merge_fragments(frags: list[_Frag], p: TrackerParams, dt: float) -> list[_Frag]:
    """Link fragments A -> B of the same object.

    Either B starts after A ends and A's end extrapolates onto B's start (and vice versa), or --
    a track drifted onto clutter while a new one took over the object -- they overlap by at most
    ``max_merge_overlap`` frames and are within ``merge_dist_overlap`` of each other at the
    hand-over.  In overlapping frames the detection with more points is kept.
    """
    cands = []
    for ia, a in enumerate(frags):
        for ib, b in enumerate(frags):
            if ia == ib:
                continue
            g = b.first - a.last
            max_gap = p.max_merge_gap_static if (a.static and b.static) else p.max_merge_gap
            if g > max_gap or g < -p.max_merge_overlap:
                continue
            if not math.isfinite(class_penalty(a.cls, b.cls, p.class_mismatch_penalty)):
                continue
            if g < 1:
                if b.first <= a.first or b.last <= a.last:
                    continue
                err = min(
                    float(np.linalg.norm(a.pos_at(b.first) - b.p_start)),
                    float(np.linalg.norm(a.p_end - b.pos_at(a.last))),
                )
                thr = p.merge_dist_overlap
            elif a.static and b.static:
                err = float(np.linalg.norm(a.med - b.med))
                thr = p.merge_dist_static + p.merge_dist_static_per_m * max(a.l, b.l)
            else:
                e1 = np.linalg.norm(a.p_end + a.v_end * g - b.p_start)
                e2 = np.linalg.norm(b.p_start - b.v_start * g - a.p_end)
                e3 = np.linalg.norm((a.p_end + a.v_end * g / 2) - (b.p_start - b.v_start * g / 2))
                err = float(min(e1, e2, e3))
                thr = p.merge_dist + p.merge_dist_per_s * g * dt
            if err <= thr:
                cands.append((err / thr, ia, ib))
    cands.sort()
    succ, pred = {}, {}
    for _, ia, ib in cands:
        if ia in succ or ib in pred:
            continue
        # avoid cycles (cannot happen with strictly increasing time, kept for safety)
        succ[ia], pred[ib] = ib, ia
    out = []
    for i, f in enumerate(frags):
        if i in pred:
            continue
        if i not in succ:
            out.append(f)
            continue
        meas, axes, core = dict(f.meas), dict(f.axes), f.core
        j = i
        while j in succ:
            j = succ[j]
            for k, d in frags[j].meas.items():
                if k not in meas or d.n_points > meas[k].n_points:
                    meas[k] = d
                    if k in frags[j].axes:
                        axes[k] = frags[j].axes[k]
            core = max(core, frags[j].core)
        m = _Frag(meas, p, axes)
        m.core = core
        out.append(m)
    return out


def _keep(f: _Frag, n_frames: int, p: TrackerParams) -> bool:
    if f.mean_score < p.min_mean_score:
        return False
    if f.n / (f.last - f.first + 1) < p.min_hit_ratio:
        return False
    if f.n >= p.min_track_frames and f.core >= p.min_track_frames:
        fr = f.frames
        k = p.min_track_frames - 1
        if np.any(fr[k:] - fr[:-k] < p.core_window):
            return True
    edge = f.first <= p.edge_frames or f.last >= n_frames - 1 - p.edge_frames
    if not edge and p.roi is not None:
        xmin, xmax, ymin, ymax = p.roi
        for q in (f.xy[0], f.xy[-1]):
            dmin = min(q[0] - xmin, xmax - q[0], q[1] - ymin, ymax - q[1])
            if dmin < p.roi_margin:
                edge = True
    return edge and f.n >= p.min_track_frames_edge


def _dedup(frags: list[_Frag], p: TrackerParams) -> list[_Frag]:
    """Resolve vehicle tracks that duplicate a longer vehicle track.

    Two vehicles cannot overlap (nor touch in the same lane while moving), so a track whose centre
    lies inside a longer vehicle track's box (+ ``dedup_margin``) -- or touches it in its lane
    while both move faster than ``lockstep_speed`` -- in at least ``dedup_fraction`` of their
    co-detected frames is the same physical object: typically the second cluster of a split
    vehicle, or a track that took over from another one after an association error.  Its detections outside the longer track's
    lifetime are handed over to the longer track; the rest is dropped.
    """
    kept: list[_Frag] = []
    for b in sorted(frags, key=lambda f: -f.n):
        target = None
        if b.group == "vehicle":
            b_lo, b_hi = b.xy.min(axis=0), b.xy.max(axis=0)
            for a in kept:
                if a.group != "vehicle" or a.last < b.first or b.last < a.first:
                    continue
                reach = a.l / 2 + p.dedup_margin
                if np.any(b_lo > a.xy.max(axis=0) + reach) or np.any(b_hi < a.xy.min(axis=0) - reach):
                    continue  # trajectories never come close
                co = inside = 0
                hl, hw = a.l / 2 + p.dedup_margin, a.w / 2 + p.dedup_margin
                touch = 0.5 * (a.l + b.l) + p.satellite_margin
                for k, d in b.meas.items():
                    da = a.meas.get(k)
                    if da is None:
                        continue
                    co += 1
                    rel = d.box[:2] - da.box[:2]
                    ax = a.axes.get(k)
                    if ax is None:
                        ax = _long_axis(da.box)
                    c, s = math.cos(ax), math.sin(ax)
                    u, v = abs(c * rel[0] + s * rel[1]), abs(-s * rel[0] + c * rel[1])
                    if u <= hl and v <= hw:
                        inside += 1
                    elif u <= touch and v <= a.w and min(a.speed_at(k), b.speed_at(k)) > p.lockstep_speed:
                        inside += 1  # touching in its lane while both move: split vehicle
                    elif (math.hypot(*rel) <= touch and d.n_points <= p.satellite_points_ratio * da.n_points
                          and a.speed_at(k) > p.moving_off):
                        inside += 1  # sparse satellite cluster accompanying a moving vehicle
                if co >= min(3, b.n) and inside >= p.dedup_fraction * co:
                    target = a
                    break
        if target is None:
            kept.append(b)
            continue
        extra = {k: d for k, d in b.meas.items() if k < target.first or k > target.last}
        if extra:
            target.meas.update(extra)
            target.axes.update({k: b.axes[k] for k in extra if k in b.axes})
            target.refresh()
    return kept


def _rts_cv(rel: np.ndarray, z: np.ndarray, r_var: np.ndarray, T: int, dt: float, sigma_acc: float):
    """Kalman filter + Rauch-Tung-Striebel smoother, CV model, both axes share the covariance.

    ``rel`` are frame indices (0..T-1) of the measurements ``z`` (m, 2).
    Returns positions (T, 2) and velocities (T, 2).
    """
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * sigma_acc**2
    zi = {int(k): (z[i], r_var[i]) for i, k in enumerate(rel)}
    x = np.zeros((2, 2))  # rows: pos, vel; cols: x, y
    x[0] = z[0]
    k = min(5, len(rel))
    if k >= 2 and rel[k - 1] > rel[0]:
        A = np.stack([np.ones(k), (rel[:k] - rel[0]) * dt], axis=1)
        sol, *_ = np.linalg.lstsq(A, z[:k], rcond=None)
        x[1] = sol[1]
    P = np.diag([r_var[0], 4.0])
    xp = np.zeros((T, 2, 2))
    Pp = np.zeros((T, 2, 2))
    xf = np.zeros((T, 2, 2))
    Pf = np.zeros((T, 2, 2))
    for t in range(T):
        if t > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        xp[t], Pp[t] = x, P
        m = zi.get(t)
        if m is not None:
            zz, r = m
            s = P[0, 0] + r
            K = P[:, 0] / s
            x = x + np.outer(K, zz - x[0])
            P = P - np.outer(K, P[0, :])
        xf[t], Pf[t] = x, P
    xs = xf.copy()
    for t in range(T - 2, -1, -1):
        C = Pf[t] @ F.T @ np.linalg.inv(Pp[t + 1])
        xs[t] = xf[t] + C @ (xs[t + 1] - xp[t + 1])
    return xs[:, 0, :], xs[:, 1, :]


def _runs(mask: np.ndarray):
    """(start, end) inclusive index pairs of True runs."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def _hysteresis(speed: np.ndarray, on: float, off: float) -> np.ndarray:
    out = np.zeros(len(speed), bool)
    state = bool(len(speed) and speed[0] > 0.5 * (on + off))
    for i, s in enumerate(speed):
        if state and s < off:
            state = False
        elif not state and s > on:
            state = True
        out[i] = state
    return out


def _robust_axis(alpha: np.ndarray, w: np.ndarray) -> float:
    """Long-axis direction (mod pi) from measurements that may be flipped by pi/2."""
    psi = math.atan2(float(np.sum(w * np.sin(4 * alpha))), float(np.sum(w * np.cos(4 * alpha)))) / 4
    agree = np.abs(_wrap(alpha - psi, _PI)) < _PI / 4
    vote = float(np.sum(w[agree]) - np.sum(w[~agree]))
    return _wrapf(psi if vote >= 0 else psi + _PI / 2, _PI)


def _kernel_smooth(t_src, v_src, w_src, T: int, sigma: float) -> np.ndarray:
    if len(t_src) == 0:
        return np.zeros(T)
    t = np.arange(T)[:, None]
    k = np.exp(-0.5 * ((t - t_src[None, :]) / sigma) ** 2) * w_src[None, :]
    s = k.sum(axis=1)
    return np.where(s > 1e-9, (k * v_src[None, :]).sum(axis=1) / np.maximum(s, 1e-12), 0.0)


def _refine(f: _Frag, p: TrackerParams, dt: float) -> dict:
    """Smooth trajectory, stationary segments, speed, moving flags and yaw of one track."""
    T = f.last - f.first + 1
    rel = f.frames - f.first
    boxes = np.array([d.box for d in f.dets])
    wts = np.array([_class_weight(d.score, d.n_points) for d in f.dets])
    veh = f.group == "vehicle"
    r = (p.meas_sigma_base + p.meas_sigma_per_m * f.l) ** 2
    pos, vel = _rts_cv(
        rel, f.xy, np.full(f.n, r), T, dt, p.smooth_sigma_acc_vehicle if veh else p.smooth_sigma_acc_vru
    )
    speed = np.linalg.norm(vel, axis=1)

    # stationary segments
    if f.static:
        static = np.ones(T, bool)
    else:
        static = speed < p.static_speed
        for a, b in _runs(static):  # too short to be a stop
            if b - a + 1 < p.min_static_frames:
                static[a : b + 1] = False
        for a, b in _runs(~static):  # short moving blips between two stops
            if a > 0 and b < T - 1 and b - a + 1 < p.min_static_frames // 2:
                static[a : b + 1] = True
    for a, b in _runs(static):
        m = (rel >= a) & (rel <= b)
        pos[a : b + 1] = np.median(f.xy[m], axis=0) if m.any() else pos[a : b + 1].mean(axis=0)
        vel[a : b + 1] = 0.0
    speed = np.linalg.norm(vel, axis=1)
    moving = _hysteresis(speed, p.moving_on, p.moving_off)

    # height / z
    if f.static or f.n < 3:
        z = np.full(T, np.median(boxes[:, 2]))
    else:
        k = 7
        zm = np.array([np.median(boxes[max(0, i - k) : i + k + 1, 2]) for i in range(f.n)])
        z = np.interp(np.arange(T), rel, zm)

    # yaw
    alpha = np.array([_long_axis(b) for b in boxes])
    yaw_long = np.where(boxes[:, 3] >= boxes[:, 4], boxes[:, 6], boxes[:, 6] + _PI / 2)
    yaw = np.zeros(T)
    free = static.copy()  # frames whose yaw is estimated from the box axis alone
    # heading reference: velocity direction (VRUs: already at walking speed)
    v_min = p.yaw_vel_min_speed if veh else p.moving_on
    valid = (speed >= v_min) & ~static
    if valid.any():
        idx = np.arange(T)
        vi = np.flatnonzero(valid)
        k = np.searchsorted(vi, idx)
        left = vi[np.clip(k - 1, 0, len(vi) - 1)]
        right = vi[np.clip(k, 0, len(vi) - 1)]
        nearest = np.where(np.abs(idx - left) <= np.abs(right - idx), left, right)
        ref = np.arctan2(vel[nearest, 1], vel[nearest, 0])
        # vehicles: box yaw = velocity direction + smoothed measured deviation (slip, lag);
        # VRU boxes are too square to carry reliable orientation, use the velocity only
        dev = _wrap(alpha - ref[rel], _PI / 2)
        ok = (np.abs(dev) <= p.max_yaw_dev) & ~static[rel] & veh
        dev_s = _kernel_smooth(rel[ok], dev[ok], wts[ok], T, p.yaw_smooth_frames)
        yaw[~static] = ref[~static] + dev_s[~static]
    else:
        free[:] = True
    for a, b in _runs(free):
        m = (rel >= a) & (rel <= b)
        if not m.any():
            m = np.ones(f.n, bool)
        ax = _robust_axis(alpha[m], wts[m])
        if a > 0 and not free[a - 1]:
            ref_h = yaw[a - 1]
        elif b < T - 1 and not free[b + 1]:
            ref_h = yaw[b + 1]
        else:
            ref_h = ax if float(np.sum(wts[m] * np.cos(yaw_long[m] - ax))) >= 0 else ax + _PI
        yaw[a : b + 1] = ax if abs(_wrapf(ax - ref_h)) <= _PI / 2 else ax + _PI
    yaw = _wrap(yaw)

    out_boxes = np.zeros((T, 7))
    out_boxes[:, 0:2] = pos
    out_boxes[:, 2] = z
    out_boxes[:, 3], out_boxes[:, 4], out_boxes[:, 5] = f.l, f.w, f.h
    out_boxes[:, 6] = yaw
    coasted = np.ones(T, bool)
    coasted[rel] = False
    score = np.full(T, f.mean_score)
    score[rel] = 0.5 * (f.scores + f.mean_score)
    npts = np.zeros(T, int)
    npts[rel] = [d.n_points for d in f.dets]
    return {
        "boxes": out_boxes,
        "speed": speed,
        "moving": moving,
        "coasted": coasted,
        "score": score,
        "n_points": npts,
    }


def _extend_static(frags: list[_Frag], refined: list[dict], spans, n: int, p: TrackerParams):
    """Object permanence for parked vehicles: extend stationary vehicle tracks to the sequence
    start/end (as ``coasted``) while their spot is not taken by another vehicle track."""

    def occupied(k: int, idx: int) -> bool:
        me = refined[idx]["boxes"][0 if k < frags[idx].first else -1]
        for j, f in enumerate(frags):
            if j == idx or f.group != "vehicle" or not (f.first <= k <= f.last):
                continue
            o = refined[j]["boxes"][k - f.first]
            near = math.hypot(o[0] - me[0], o[1] - me[1]) < 0.5 * (max(o[3], o[4]) + max(me[3], me[4]))
            if near and _bev_intersection(o, me) > 0.0:
                return True
        return False

    out = list(spans)
    for idx, f in enumerate(frags):
        if f.group != "vehicle" or not f.static or f.n < p.min_track_frames:
            continue
        a, b = spans[idx]
        while a > 0 and not occupied(a - 1, idx):
            a -= 1
        while b < n - 1 and not occupied(b + 1, idx):
            b += 1
        out[idx] = (a, b)
    return out


def track_sequence(frames: list[list[Detection]], dt: float = 0.1, **params) -> tuple[list[list[dict]], dict]:
    """Offline tracking over a whole recorded sequence.

    Runs :class:`MultiObjectTracker` forward, then

    1. merges fragments of the same object (end of A consistent in space/time/motion with the
       start of B, or a short overlap at a hand-over; stationary fragments at the same spot;
       compatible class),
    2. drops short tracks (unless they touch the sequence start/end or the ROI border), tracks with
       a low mean score and tracks detected in less than half of their lifetime,
    3. merges vehicle tracks into a longer vehicle track they live inside of / touch in lockstep
       / accompany as a sparse satellite (split vehicles, take-overs),
    4. re-estimates every track from all its detections: RTS-smoothed trajectory, stationary
       segments frozen to the median position, robust dimensions, voted class, yaw resolved
       against the velocity direction; the tentative period is backfilled and gaps are filled
       (``coasted``: True),
    5. optionally (``extend_static``) extends parked vehicles to the sequence start/end.

    Returns ``(objects_per_frame, tracks)`` where ``tracks`` maps ``str(id)`` to
    ``{"cls", "group", "first", "last", "moving", "max_speed", "n_frames"}``.  Track ids are
    renumbered 1..N in order of appearance.
    """
    tr = MultiObjectTracker(dt=dt, **params)
    for dets in frames:
        tr.step(dets)
    p = tr.params
    n = len(frames)
    _Frag.dt = dt

    frags = [_Frag(t.meas, p, t.axes) for t in tr.all_tracks]
    frags = _merge_fragments(frags, p, dt)
    frags = [f for f in frags if _keep(f, n, p)]
    frags = _dedup(frags, p)
    frags.sort(key=lambda f: (f.first, float(f.med[0]), float(f.med[1])))

    refined = [_refine(f, p, dt) for f in frags]
    spans = [(f.first, f.last) for f in frags]
    if p.extend_static:
        spans = _extend_static(frags, refined, spans, n, p)

    out: list[list[dict]] = [[] for _ in range(n)]
    tracks: dict[str, dict] = {}
    for tid, (f, r, (a, b)) in enumerate(zip(frags, refined, spans), start=1):
        for k in range(a, b + 1):
            i = min(max(k - f.first, 0), f.last - f.first)  # outside the detected span: frozen box
            extended = k < f.first or k > f.last
            out[k].append(
                {
                    "id": tid,
                    "cls": f.cls,
                    "box": [round(float(v), 3) for v in r["boxes"][i]],
                    "score": round(float(r["score"][i]), 3),
                    "moving": bool(r["moving"][i]) and not extended,
                    "speed": 0.0 if extended else round(float(r["speed"][i]), 3),
                    "n_points": 0 if extended else int(r["n_points"][i]),
                    "coasted": bool(extended or r["coasted"][i]),
                }
            )
        tracks[str(tid)] = {
            "cls": f.cls,
            "group": f.group,
            "first": a,
            "last": b,
            "moving": bool(r["moving"].any()),
            "max_speed": round(float(r["speed"].max()), 3),
            "n_frames": b - a + 1,
        }
    return out, tracks
