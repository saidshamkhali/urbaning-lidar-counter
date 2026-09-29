"""Box geometry: corners, rotated BEV / 3D IoU and point-in-box tests.

Boxes are ``(N, 7)`` float arrays ``[x, y, z, l, w, h, yaw]`` with ``z`` the box centre and
``yaw`` counter-clockwise from +x (see ``docs/DATA_FORMAT.md``).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "wrap_angle",
    "box_corners_bev",
    "bev_intersection_matrix",
    "bev_iou_matrix",
    "iou_3d_matrix",
    "points_in_boxes",
]

# Unit-box corners in the box frame, counter-clockwise, starting front-right.
_UNIT_CORNERS = np.array([[0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5]])


def _as_boxes(boxes: np.ndarray) -> np.ndarray:
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape(0, 7)
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(f"expected boxes of shape (N, 7), got {arr.shape}")
    return arr


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) in radians to the interval (-pi, pi]."""
    wrapped = np.pi - np.mod(np.pi - np.asarray(a, dtype=np.float64), 2.0 * np.pi)
    return float(wrapped) if np.ndim(wrapped) == 0 else wrapped


def box_corners_bev(boxes: np.ndarray) -> np.ndarray:
    """Return the ``(N, 4, 2)`` bird's-eye-view corners of ``boxes`` in counter-clockwise order."""
    boxes = _as_boxes(boxes)
    local = _UNIT_CORNERS[None] * boxes[:, None, 3:5]  # (N, 4, 2) scaled by (l, w)
    c, s = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)  # (N, 2, 2)
    return np.einsum("nij,nkj->nki", rot, local) + boxes[:, None, :2]


def _polygon_area(poly: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Shoelace area of ``K`` padded polygons ``(K, M, 2)`` whose first ``count`` vertices are valid."""
    k, m = poly.shape[:2]
    if m == 0:
        return np.zeros(k)
    idx = np.arange(m)[None, :]
    nxt = np.where(idx + 1 < count[:, None], idx + 1, 0)
    p_next = np.take_along_axis(poly, nxt[..., None], axis=1)
    cross = poly[..., 0] * p_next[..., 1] - poly[..., 1] * p_next[..., 0]
    area = 0.5 * np.where(idx < count[:, None], cross, 0.0).sum(axis=1)
    return np.where(count >= 3, np.abs(area), 0.0)


def _clip_half_plane(
    poly: np.ndarray, count: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One Sutherland-Hodgman step: clip each polygon to the left of the directed edge ``a -> b``."""
    k, m = poly.shape[:2]
    idx = np.arange(m)[None, :]
    valid = idx < count[:, None]
    nxt = np.where(idx + 1 < count[:, None], idx + 1, 0)
    cur, nex = poly, np.take_along_axis(poly, nxt[..., None], axis=1)

    edge = (b - a)[:, None, :]
    side_cur = edge[..., 0] * (cur[..., 1] - a[:, None, 1]) - edge[..., 1] * (cur[..., 0] - a[:, None, 0])
    side_nex = edge[..., 0] * (nex[..., 1] - a[:, None, 1]) - edge[..., 1] * (nex[..., 0] - a[:, None, 0])
    in_cur, in_nex = side_cur >= 0.0, side_nex >= 0.0

    denom = side_cur - side_nex
    t = np.divide(side_cur, denom, out=np.zeros_like(denom), where=denom != 0.0)
    inter = cur + t[..., None] * (nex - cur)

    # Each input vertex emits itself (if inside) followed by the edge crossing (if any).
    cand = np.stack([cur, inter], axis=2).reshape(k, 2 * m, 2)
    keep = np.stack([valid & in_cur, valid & (in_cur != in_nex)], axis=2).reshape(k, 2 * m)
    order = np.argsort(~keep, axis=1, kind="stable")
    new_count = keep.sum(axis=1)
    width = int(new_count.max()) if k else 0
    return np.take_along_axis(cand, order[:, :width, None], axis=1), new_count


def _pairwise_intersection_area(ca: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """Exact intersection area of ``K`` pairs of convex CCW quadrilaterals ``(K, 4, 2)``."""
    poly, count = ca, np.full(len(ca), 4)
    for j in range(4):
        poly, count = _clip_half_plane(poly, count, cb[:, j], cb[:, (j + 1) % 4])
    return _polygon_area(poly, count)


def bev_intersection_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise ``(N, M)`` bird's-eye-view intersection areas of rotated boxes."""
    a, b = _as_boxes(a), _as_boxes(b)
    inter = np.zeros((len(a), len(b)))
    if len(a) == 0 or len(b) == 0:
        return inter
    # Prefilter: only pairs whose circumscribed circles overlap can intersect.
    ra = 0.5 * np.hypot(a[:, 3], a[:, 4])
    rb = 0.5 * np.hypot(b[:, 3], b[:, 4])
    dist = np.linalg.norm(a[:, None, :2] - b[None, :, :2], axis=-1)
    ia, ib = np.nonzero(dist < ra[:, None] + rb[None, :])
    if len(ia):
        inter[ia, ib] = _pairwise_intersection_area(box_corners_bev(a)[ia], box_corners_bev(b)[ib])
    return inter


def bev_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise ``(N, M)`` exact bird's-eye-view IoU of rotated boxes."""
    a, b = _as_boxes(a), _as_boxes(b)
    inter = bev_intersection_matrix(a, b)
    union = (a[:, 3] * a[:, 4])[:, None] + (b[:, 3] * b[:, 4])[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)


def iou_3d_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise ``(N, M)`` 3D IoU: BEV intersection x vertical overlap / union volume."""
    a, b = _as_boxes(a), _as_boxes(b)
    inter_bev = bev_intersection_matrix(a, b)
    top = np.minimum((a[:, 2] + a[:, 5] / 2)[:, None], (b[:, 2] + b[:, 5] / 2)[None, :])
    bottom = np.maximum((a[:, 2] - a[:, 5] / 2)[:, None], (b[:, 2] - b[:, 5] / 2)[None, :])
    inter = inter_bev * np.clip(top - bottom, 0.0, None)
    vol_a = a[:, 3] * a[:, 4] * a[:, 5]
    vol_b = b[:, 3] * b[:, 4] * b[:, 5]
    union = vol_a[:, None] + vol_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)


def points_in_boxes(points: np.ndarray, boxes: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Index of the box containing each point, or -1 (``(P,)`` int32).

    ``margin`` (metres) enlarges every box on all sides. Where boxes overlap, the lowest box
    index wins.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    boxes = _as_boxes(boxes)
    out = np.full(len(points), -1, dtype=np.int32)
    if len(points) == 0:
        return out
    for i, (x, y, z, l, w, h, yaw) in enumerate(boxes):
        half_l, half_w, half_h = l / 2 + margin, w / 2 + margin, h / 2 + margin
        radius = np.hypot(half_l, half_w)
        cand = np.flatnonzero(
            (out < 0)
            & (np.abs(points[:, 2] - z) <= half_h)
            & (np.abs(points[:, 0] - x) <= radius)
            & (np.abs(points[:, 1] - y) <= radius)
        )
        if len(cand) == 0:
            continue
        dx, dy = points[cand, 0] - x, points[cand, 1] - y
        c, s = np.cos(yaw), np.sin(yaw)
        inside = (np.abs(c * dx + s * dy) <= half_l) & (np.abs(-s * dx + c * dy) <= half_w)
        out[cand[inside]] = i
    return out
