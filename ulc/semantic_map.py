"""Rasterised HD-map layers (road, walkway, vegetation, …) used as detector features."""
from __future__ import annotations

import cv2
import numpy as np

from .io import load_lanelet_map

LAYERS = {
    "road": ("road", "bicycle_lane"),
    "walk": ("walkway", "shared_walkway", "traffic_island"),
    "green": ("vegetation", "keepout"),
}


class SemanticMap:
    def __init__(self, crossing: str = "crossing1", radius: float = 100.0, cell: float = 0.5):
        self.radius, self.cell = radius, cell
        self.n = int(round(2 * radius / cell))
        self.map = load_lanelet_map(crossing=crossing)
        self.layers: dict[str, np.ndarray] = {}
        for name, kinds in LAYERS.items():
            img = np.zeros((self.n, self.n), np.uint8)
            polys = [self._pix(np.asarray(a["polygon"])) for a in self.map["areas"] if a["kind"] in kinds]
            if polys:
                cv2.fillPoly(img, polys, 1)
            self.layers[name] = img.astype(bool)

        self._build_lane_field()

    def _build_lane_field(self, sigma_m: float = 2.0) -> None:
        """Dominant lane direction (mod π) per cell from lane markings, curbs and road borders.

        Segment directions are accumulated as doubled-angle vectors (cos 2θ, sin 2θ) so that
        opposite directions reinforce each other, then blurred. The vector length after
        normalisation is a coherence in [0, 1]: ~1 along straight roads, low inside the
        intersection where several directions meet.
        """
        kinds = {"line_thin", "line_thick", "curbstone", "road_border"}
        acc = np.zeros((2, self.n, self.n), np.float32)
        wsum = np.zeros((self.n, self.n), np.float32)
        for ln in self.map["lines"]:
            if ln["kind"] not in kinds:
                continue
            pts = np.asarray(ln["points"])[:, :2]
            for a, b in zip(pts[:-1], pts[1:]):
                d = b - a
                seg = float(np.hypot(*d))
                if seg < 1e-3:
                    continue
                th = np.arctan2(d[1], d[0])
                n_s = max(2, int(seg / (self.cell * 0.5)))
                q = a + np.linspace(0, 1, n_s)[:, None] * d
                ix = ((q[:, 0] + self.radius) / self.cell).astype(int)
                iy = ((q[:, 1] + self.radius) / self.cell).astype(int)
                ok = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
                np.add.at(acc[0], (ix[ok], iy[ok]), np.cos(2 * th))
                np.add.at(acc[1], (ix[ok], iy[ok]), np.sin(2 * th))
                np.add.at(wsum, (ix[ok], iy[ok]), 1.0)
        s = sigma_m / self.cell
        acc = np.stack([cv2.GaussianBlur(a, (0, 0), s) for a in acc])
        wsum = cv2.GaussianBlur(wsum, (0, 0), s)
        mag = np.hypot(acc[0], acc[1])
        self.lane_theta = 0.5 * np.arctan2(acc[1], acc[0])
        self.lane_coherence = np.where(wsum > 1e-3, mag / np.maximum(wsum, 1e-6), 0.0).astype(np.float32)
        self.lane_support = wsum

    def lane_yaw(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lane direction (mod π) and its coherence at each (x, y); coherence 0 off the map."""
        ix = ((xy[:, 0] + self.radius) / self.cell).astype(np.int64)
        iy = ((xy[:, 1] + self.radius) / self.cell).astype(np.int64)
        ok = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        yaw = np.zeros(len(xy))
        coh = np.zeros(len(xy))
        yaw[ok] = self.lane_theta[ix[ok], iy[ok]]
        coh[ok] = self.lane_coherence[ix[ok], iy[ok]] * np.clip(self.lane_support[ix[ok], iy[ok]] / 0.05, 0, 1)
        return yaw, coh

    def _pix(self, xy: np.ndarray) -> np.ndarray:
        # image row = x index, column = y index, so cv2 (col,row) = (y, x)
        ij = np.round((xy + self.radius) / self.cell).astype(np.int32)
        return ij[:, ::-1].reshape(-1, 1, 2)

    def lookup(self, layer: str, xy: np.ndarray) -> np.ndarray:
        ix = ((xy[:, 0] + self.radius) / self.cell).astype(np.int64)
        iy = ((xy[:, 1] + self.radius) / self.cell).astype(np.int64)
        ok = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        out = np.zeros(len(xy), bool)
        out[ok] = self.layers[layer][ix[ok], iy[ok]]
        return out
