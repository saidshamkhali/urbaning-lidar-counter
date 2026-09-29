"""Geometric object detection on fused infrastructure LiDAR frames.

Per frame:

1. Split points into ground / permanent structure / object candidates using the
   static :class:`~ulc.background.SceneModel`.
2. Cluster candidate points in bird's-eye view (DBSCAN).
3. Fit an oriented box to each cluster with search-based L-shape fitting
   (Zhang et al., "Efficient L-Shape Fitting for Vehicle Detection Using Laser Scanners", 2017).
4. Describe every cluster with a geometric/semantic feature vector. A learned classifier
   (:mod:`ulc.classify`) turns features into a class and a confidence; without a model a
   conservative rule-based classifier is used.
5. Complete partially-seen vehicles ("amodal" boxes): extend the box away from the
   sensor up to a class-specific size prior, because a LiDAR only sees the sides facing it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN

from .background import SceneModel
from .io import INTENSITY
from .semantic_map import SemanticMap

# (length, width, height) priors for amodal completion
SIZE_PRIOR = {
    "car": (4.5, 1.85, 1.55),
    "van": (5.2, 2.0, 2.1),
    "truck": (8.0, 2.5, 3.2),
    "bus": (12.0, 2.55, 3.1),
    "cyclist": (1.8, 0.7, 1.7),
    "pedestrian": (0.6, 0.6, 1.75),
}
DETECTOR_CLASSES = ["background", "car", "van", "truck", "bus", "cyclist", "pedestrian"]

FEATURE_NAMES = [
    "l_obs", "w_obs", "h_max", "h_p50", "h_min", "area", "log_n", "density", "range", "n_x_range2",
    "int_mean", "int_std", "int_max", "fg_frac", "freq_mean", "linearity", "planarity", "scatter",
    "top_frac", "z_std", "on_road", "on_walk", "on_green", "aspect", "n_lidars",
]


@dataclass
class Cluster:
    idx: np.ndarray                  # indices into the frame's point array
    box: np.ndarray                  # (7,) observed (tight) box
    features: np.ndarray
    cls: str = "background"
    score: float = 0.0
    probs: dict[str, float] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.idx)


# --------------------------------------------------------------------------- box fitting


def lshape_yaw(xy: np.ndarray, step_deg: float = 1.0, d0: float = 0.01, max_points: int = 600) -> float:
    """Heading (mod π/2) that best explains the points as an L-shape (closeness criterion)."""
    if len(xy) > max_points:
        xy = xy[np.random.default_rng(0).choice(len(xy), max_points, replace=False)]
    theta = np.deg2rad(np.arange(0.0, 90.0, step_deg))
    c, s = np.cos(theta), np.sin(theta)
    c1 = xy[:, :1] * c + xy[:, 1:] * s          # (N, A)
    c2 = -xy[:, :1] * s + xy[:, 1:] * c
    d1 = np.minimum(c1.max(0) - c1, c1 - c1.min(0))
    d2 = np.minimum(c2.max(0) - c2, c2 - c2.min(0))
    score = (1.0 / np.maximum(np.minimum(d1, d2), d0)).sum(0)
    return float(theta[int(np.argmax(score))])


def fit_box(xy: np.ndarray, zmin: float, zmax: float, yaw: float | None = None) -> np.ndarray:
    """Tight oriented box around points: [x, y, z, l, w, h, yaw] with l ≥ w.

    The heading comes from L-shape fitting unless it is given (e.g. from the lane prior),
    in which case `l` is measured along the given heading.
    """
    fixed = yaw is not None
    if len(xy) < 3 and not fixed:
        cx, cy = xy.mean(0)
        return np.array([cx, cy, (zmin + zmax) / 2, 0.3, 0.3, zmax - zmin, 0.0])
    if yaw is None:
        yaw = lshape_yaw(xy - xy.mean(0))
    c, s = np.cos(yaw), np.sin(yaw)
    u = xy[:, 0] * c + xy[:, 1] * s
    v = -xy[:, 0] * s + xy[:, 1] * c
    lu, lv = u.max() - u.min(), v.max() - v.min()
    cu, cv = (u.max() + u.min()) / 2, (v.max() + v.min()) / 2
    cx, cy = cu * c - cv * s, cu * s + cv * c
    if lv > lu and not fixed:  # make l the longer side
        lu, lv, yaw = lv, lu, yaw + np.pi / 2
    return np.array([cx, cy, (zmin + zmax) / 2, max(lu, 0.1), max(lv, 0.1), zmax - zmin, yaw])


def complete_box(box: np.ndarray, cls: str, sensor_xy: np.ndarray, ground: float) -> np.ndarray:
    """Grow a partially observed box to the class size prior, away from the sensor."""
    out = box.copy()
    if cls not in SIZE_PRIOR:
        return out
    L, W, H = SIZE_PRIOR[cls]
    c, s = np.cos(box[6]), np.sin(box[6])
    rel = sensor_xy - box[:2]
    su, sv = rel[0] * c + rel[1] * s, -rel[0] * s + rel[1] * c   # sensor in box frame
    du = dv = 0.0
    if cls in ("car", "van") or (cls in ("truck", "bus") and box[3] < 0.6 * L):
        if box[3] < L * 0.9:
            grow = L * 0.95 - box[3]
            du = -np.sign(su) * grow / 2 if abs(su) > box[3] / 2 else 0.0
            out[3] = box[3] + grow
    if box[4] < W * 0.85:
        grow = W * 0.95 - box[4]
        dv = -np.sign(sv) * grow / 2 if abs(sv) > box[4] / 2 else 0.0
        out[4] = box[4] + grow
    out[0] = box[0] + du * c - dv * s
    out[1] = box[1] + du * s + dv * c
    # boxes stand on the ground
    top = ground + max(box[5] + (box[2] - box[5] / 2 - ground), 0.5 * H)
    out[5] = top - ground
    out[2] = ground + out[5] / 2
    return out


# --------------------------------------------------------------------------- detector


class Detector:
    def __init__(self, scene: SceneModel, sensors_xyz: np.ndarray, classifier=None,
                 eps: float = 0.55, min_samples: int = 3, min_points: int = 5, max_range: float = 75.0,
                 lane_prior_points: int = 120, lane_prior_coherence: float = 0.75, junction_radius: float = 18.0):
        self.scene = scene
        self.sensors_xyz = sensors_xyz
        self.classifier = classifier
        self.eps, self.min_samples, self.min_points, self.max_range = eps, min_samples, min_points, max_range
        self.lane_prior_points, self.lane_prior_coherence = lane_prior_points, lane_prior_coherence
        self.junction_radius = junction_radius
        self.smap = SemanticMap(scene.crossing)
        # centre of the sensor constellation ~ centre of the intersection
        self.center = sensors_xyz[:, :2].mean(0)

    # -- clustering -------------------------------------------------------------
    def clusters(self, pts: np.ndarray, seq: str) -> tuple[list[Cluster], dict[str, np.ndarray]]:
        info = self.scene.classify_points(pts, seq)
        r = np.hypot(*(pts[:, :2] - self.center).T)
        cand = np.flatnonzero(info["candidate"] & (r < self.max_range))
        clusters: list[Cluster] = []
        if len(cand) < self.min_points:
            return clusters, info
        xy = pts[cand, :2].astype(np.float64)
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(xy)
        for lab in np.unique(labels[labels >= 0]):
            idx = cand[labels == lab]
            if len(idx) < self.min_points:
                continue
            for part in self._split_oversized(pts, idx):
                c = self._make_cluster(pts, part, info)
                if c is not None:
                    clusters.append(c)
        return clusters, info

    def _split_oversized(self, pts: np.ndarray, idx: np.ndarray) -> list[np.ndarray]:
        """Clusters much larger than a bus usually are several parked cars touching: re-split tighter."""
        xy = pts[idx, :2]
        ext = xy.max(0) - xy.min(0)
        if max(ext) < 16.0 and min(ext) < 5.0:
            return [idx]
        labels = DBSCAN(eps=self.eps * 0.6, min_samples=self.min_samples).fit_predict(xy)
        return [idx[labels == k] for k in np.unique(labels[labels >= 0]) if (labels == k).sum() >= self.min_points]

    def _make_cluster(self, pts: np.ndarray, idx: np.ndarray, info: dict) -> Cluster | None:
        p = pts[idx]
        h = info["h"][idx]
        ground = float(np.median(p[:, 2] - h))
        xy = p[:, :2].astype(np.float64)
        top = ground + float(np.percentile(h, 98))
        box = fit_box(xy, ground, top)
        # Sparse, partial views can't constrain the heading; on a road arm, vehicles follow the
        # lane. Inside the junction lanes cross, so there the tracker's motion direction decides.
        lane_yaw, coh = self.smap.lane_yaw(box[None, :2])
        in_junction = np.hypot(*(box[:2] - self.center)) < self.junction_radius
        if len(idx) < self.lane_prior_points and coh[0] >= self.lane_prior_coherence and not in_junction:
            box = fit_box(xy, ground, top, yaw=float(lane_yaw[0]))
        if box[3] > 25 or box[4] > 8:  # walls/hedges that escaped the permanent mask
            return None
        return Cluster(idx=idx, box=box, features=self._features(p, h, box, info, idx))

    def _features(self, p: np.ndarray, h: np.ndarray, box: np.ndarray, info: dict, idx: np.ndarray) -> np.ndarray:
        n = len(p)
        rng = float(np.min(np.hypot(*(self.sensors_xyz[:, :2] - box[:2]).T)))
        area = box[3] * box[4]
        inten = p[:, INTENSITY]
        xyz = p[:, :3] - p[:, :3].mean(0)
        ev = np.sort(np.linalg.eigvalsh(np.cov(xyz.T) + 1e-9 * np.eye(3)))[::-1]
        ev = ev / max(ev[0], 1e-9)
        hmax = float(np.percentile(h, 98))
        xy_c = box[:2][None]
        return np.array([
            box[3], box[4], hmax, float(np.median(h)), float(np.percentile(h, 2)), area, np.log(n),
            n / max(area, 0.05), rng, n * (rng / 20.0) ** 2 / 100.0,
            inten.mean(), inten.std(), inten.max(),
            float(info["foreground"][idx].mean()), float(info["freq"][idx].mean()),
            1 - ev[1], ev[1] - ev[2], ev[2],
            float((h > 0.8 * hmax).mean()), float(h.std()),
            float(self.smap.lookup("road", xy_c)[0]), float(self.smap.lookup("walk", xy_c)[0]),
            float(self.smap.lookup("green", xy_c)[0]),
            box[3] / max(box[4], 0.05), float(len(np.unique(p[:, 4]))),
        ], dtype=np.float32)

    # -- classification ---------------------------------------------------------
    def classify(self, clusters: list[Cluster]) -> None:
        if not clusters:
            return
        X = np.stack([c.features for c in clusters])
        if self.classifier is not None:
            proba = self.classifier.predict_proba(X)
            classes = list(self.classifier.classes_)
            for c, pr in zip(clusters, proba):
                k = int(np.argmax(pr))
                c.cls, c.score = classes[k], float(pr[k])
                c.probs = {cl: float(v) for cl, v in zip(classes, pr)}
                if c.cls == "background":
                    # keep the best object hypothesis around with its (low) confidence
                    obj = [(v, cl) for cl, v in c.probs.items() if cl != "background"]
                    v, cl = max(obj)
                    c.score = float(v)
                    c.cls = cl if v > 0.25 else "background"
        else:
            for c in clusters:
                c.cls, c.score = rule_based_class(c.features)

    # -- full pipeline ----------------------------------------------------------
    def detect(self, pts: np.ndarray, seq: str, min_score: float = 0.0) -> tuple[list[Cluster], dict]:
        clusters, info = self.clusters(pts, seq)
        self.classify(clusters)
        dets = [c for c in clusters if c.cls != "background" and c.score >= min_score]
        for c in dets:
            self._complete(c)
        dets = self._merge_fragments(pts, dets, info)
        return dets, info

    def _complete(self, c: Cluster) -> None:
        k = int(np.argmin(np.hypot(*(self.sensors_xyz[:, :2] - c.box[:2]).T)))
        c.box = complete_box(c.box, c.cls, self.sensors_xyz[k, :2], c.box[2] - c.box[5] / 2)

    def _merge_fragments(self, pts: np.ndarray, dets: list[Cluster], info: dict) -> list[Cluster]:
        """Merge vehicle detections whose completed boxes overlap: they are pieces of one vehicle
        (e.g. a car split by an occluding pole, or the front and back of a bus)."""
        from .geometry import bev_intersection_matrix

        vehicle = {"car", "van", "truck", "bus"}
        changed = True
        while changed and len(dets) > 1:
            changed = False
            boxes = np.stack([d.box for d in dets])
            inter = bev_intersection_matrix(boxes, boxes)
            area = boxes[:, 3] * boxes[:, 4]
            overlap = inter / np.minimum(area[:, None], area[None, :])
            np.fill_diagonal(overlap, 0)
            for a, b in zip(*np.nonzero(np.triu(overlap > 0.25))):
                da, db = dets[a], dets[b]
                if da.cls not in vehicle or db.cls not in vehicle:
                    continue
                idx = np.concatenate([da.idx, db.idx])
                merged = self._make_cluster(pts, idx, info)
                if merged is None or merged.box[3] > 14.0:
                    continue
                self.classify([merged])
                if merged.cls not in vehicle:  # the union doesn't look like one vehicle
                    main = da if da.n_points >= db.n_points else db
                    merged.cls = main.cls
                    merged.score = main.score
                merged.score = max(merged.score, da.score, db.score)
                self._complete(merged)
                dets = [d for k, d in enumerate(dets) if k not in (a, b)] + [merged]
                changed = True
                break
        return dets


def rule_based_class(f: np.ndarray) -> tuple[str, float]:
    """Hand-tuned fallback classifier on (observed) cluster geometry."""
    fd = dict(zip(FEATURE_NAMES, f))
    l, w, h = fd["l_obs"], fd["w_obs"], fd["h_max"]
    if h < 0.7 or fd["h_min"] > 1.2 or fd["n_x_range2"] < 0.02:
        return "background", 0.0
    if l > 9.0 and h > 2.4 and w > 1.8:
        return "bus", 0.7
    if l > 6.0 and h > 2.3:
        return "truck", 0.6
    if 2.0 < l < 6.5 and 1.0 < w < 2.6 and 1.0 < h < 2.3:
        return ("van", 0.6) if h > 1.85 else ("car", 0.7)
    if l < 2.2 and w < 1.1 and 1.2 < h < 2.1:
        return ("cyclist", 0.4) if l > 1.2 else ("pedestrian", 0.4)
    if 1.5 < l < 6.5 and w < 2.6 and h > 0.9 and fd["on_road"]:
        return "car", 0.4
    return "background", 0.0
