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
