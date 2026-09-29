import numpy as np
import pytest

from ulc.geometry import (
    bev_iou_matrix,
    box_corners_bev,
    iou_3d_matrix,
    points_in_boxes,
    wrap_angle,
)


def box(x=0.0, y=0.0, z=0.0, l=4.0, w=2.0, h=1.5, yaw=0.0):
    return np.array([[x, y, z, l, w, h, yaw]], dtype=float)


def random_boxes(rng, n, spread=10.0):
    return np.column_stack([
        rng.uniform(-spread, spread, (n, 2)),
        rng.uniform(0, 2, n),
        rng.uniform(1, 6, n),
        rng.uniform(0.5, 3, n),
        rng.uniform(1, 3, n),
        rng.uniform(-np.pi, np.pi, n),
    ])


def test_wrap_angle():
    assert wrap_angle(np.pi) == pytest.approx(np.pi)
    assert wrap_angle(-np.pi) == pytest.approx(np.pi)
    assert wrap_angle(3 * np.pi) == pytest.approx(np.pi)
    assert wrap_angle(0.0) == 0.0
    a = np.linspace(-20, 20, 1001)
    w = wrap_angle(a)
    assert np.all(w > -np.pi) and np.all(w <= np.pi)
    assert np.allclose(np.cos(w), np.cos(a)) and np.allclose(np.sin(w), np.sin(a))


def test_corners_ccw_and_area():
    rng = np.random.default_rng(0)
    b = random_boxes(rng, 20)
    c = box_corners_bev(b)
    assert c.shape == (20, 4, 2)
    x, y = c[..., 0], c[..., 1]
    signed = 0.5 * np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, axis=1)
    assert np.allclose(signed, b[:, 3] * b[:, 4])  # positive => counter-clockwise
    assert np.allclose(c.mean(axis=1), b[:, :2])


def test_corners_heading():
    c = box_corners_bev(box(l=4, w=2, yaw=np.pi / 2))[0]
    # Length now runs along +y.
    assert np.allclose(np.ptp(c[:, 1]), 4) and np.allclose(np.ptp(c[:, 0]), 2)


def test_identical_boxes():
    rng = np.random.default_rng(1)
    b = random_boxes(rng, 30)
    assert np.allclose(np.diag(bev_iou_matrix(b, b)), 1.0)
    assert np.allclose(np.diag(iou_3d_matrix(b, b)), 1.0)


def test_rotated_square_invariance():
    a = box(l=2, w=2)
    for yaw in (np.pi / 2, np.pi, -np.pi / 2, 2 * np.pi):
        assert bev_iou_matrix(a, box(l=2, w=2, yaw=yaw))[0, 0] == pytest.approx(1.0)
    # A 4x2 box rotated by 180 deg is the same rectangle; by 90 deg it overlaps in a 2x2 square.
    assert bev_iou_matrix(box(), box(yaw=np.pi))[0, 0] == pytest.approx(1.0)
    assert bev_iou_matrix(box(), box(yaw=np.pi / 2))[0, 0] == pytest.approx(4 / 12)


def test_known_overlaps():
    # Shift along length: intersection 3x2, union 10.
    assert bev_iou_matrix(box(), box(x=1))[0, 0] == pytest.approx(0.6)
    # Shift along width: intersection 4x1.5, union 10.
    assert bev_iou_matrix(box(), box(y=0.5))[0, 0] == pytest.approx(0.6)
    # Same configuration rotated as a whole is invariant.
    yaw = 0.7
    shifted = box(x=np.cos(yaw), y=np.sin(yaw), yaw=yaw)
    assert bev_iou_matrix(box(yaw=yaw), shifted)[0, 0] == pytest.approx(0.6)
    # Unit square vs itself rotated 45 deg: regular octagon of area 2(sqrt2 - 1).
    inter = 2 * (np.sqrt(2) - 1)
    assert bev_iou_matrix(box(l=1, w=1), box(l=1, w=1, yaw=np.pi / 4))[0, 0] == pytest.approx(inter / (2 - inter))
    # Small box fully inside a big one.
    assert bev_iou_matrix(box(l=4, w=4), box(l=2, w=2, yaw=0.3))[0, 0] == pytest.approx(4 / 16)


def test_disjoint_and_touching():
    assert bev_iou_matrix(box(), box(x=10))[0, 0] == 0.0
    assert bev_iou_matrix(box(), box(x=4))[0, 0] == pytest.approx(0.0, abs=1e-12)
    # Circumscribed circles overlap but the rectangles do not.
    assert bev_iou_matrix(box(yaw=0.0), box(x=3.2, y=2.5, yaw=np.pi / 2))[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert iou_3d_matrix(box(), box(z=5))[0, 0] == 0.0


def test_iou_3d_vertical():
    # Same footprint, shifted up by half the height: overlap h/2, union 1.5 * volume.
    assert iou_3d_matrix(box(h=2), box(z=1, h=2))[0, 0] == pytest.approx(1 / 3)
    # BEV IoU ignores z.
    assert bev_iou_matrix(box(h=2), box(z=1, h=2))[0, 0] == pytest.approx(1.0)


def test_empty_inputs():
    assert bev_iou_matrix(np.zeros((0, 7)), box()).shape == (0, 1)
    assert iou_3d_matrix(box(), np.zeros((0, 7))).shape == (1, 0)
    assert points_in_boxes(np.zeros((5, 3)), np.zeros((0, 7))).tolist() == [-1] * 5


def test_iou_matches_rasterisation():
    """Random boxes: exact BEV IoU agrees with a fine-grid area estimate."""
    rng = np.random.default_rng(2)
    a = random_boxes(rng, 12, spread=2)
    b = random_boxes(rng, 12, spread=2)
    iou = bev_iou_matrix(a, b)
    g = np.arange(-8, 8, 0.02) + 0.01
    gx, gy = np.meshgrid(g, g)
    pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    flat = lambda bx: np.column_stack([bx[:, :2], np.zeros(len(bx)), bx[:, 3:5], np.ones(len(bx)), bx[:, 6]])  # noqa: E731
    fa, fb = flat(a), flat(b)
    inside_a = np.stack([points_in_boxes(pts, fa[i:i + 1]) == 0 for i in range(len(a))])
    inside_b = np.stack([points_in_boxes(pts, fb[j:j + 1]) == 0 for j in range(len(b))])
    inter = inside_a.astype(float) @ inside_b.T.astype(float)
    union = inside_a.sum(1)[:, None] + inside_b.sum(1)[None, :] - inter
    assert np.abs(iou - inter / union).max() < 0.02
    assert (iou > 0).sum() > 20  # the test actually exercises overlaps


def test_points_in_boxes():
    boxes = np.vstack([box(), box(x=10, yaw=np.pi / 2)])
    pts = np.array([
        [0, 0, 0],       # centre of box 0
        [1.9, 0.9, 0.7],  # near a corner of box 0
        [2.1, 0, 0],     # just outside box 0 along length
        [0, 0, 0.8],     # above box 0
        [10, 1.9, 0],    # inside box 1 (length along y)
        [11.5, 0, 0],    # outside box 1 (width is along x)
        [50, 50, 0],
    ])
    assert points_in_boxes(pts, boxes).tolist() == [0, 0, -1, -1, 1, -1, -1]
    assert points_in_boxes(pts, boxes, margin=0.2).tolist() == [0, 0, 0, 0, 1, -1, -1]
    assert points_in_boxes(pts, boxes).dtype == np.int32


def test_points_in_overlapping_boxes_lowest_index_wins():
    boxes = np.vstack([box(), box(x=1)])
    assert points_in_boxes(np.array([[1.0, 0, 0], [2.5, 0, 0]]), boxes).tolist() == [0, 1]
