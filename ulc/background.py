"""Static scene model for fixed infrastructure LiDARs.

Because the sensors never move, statistics over many frames tell us what the empty
scene looks like:

* **Ground elevation map** – a robust low percentile of point heights per 0.5 m cell,
  used to refine the devkit's quadratic ground model (curbs, sidewalks, road camber).
* **Occupancy frequency** – for every 0.25 m voxel, the fraction of frames in which it
  contains points. Computed per sequence and combined across sequences:

  * *permanent* voxels (occupied in every sequence, recorded on different days) are
    buildings, poles, trees, signs → removed before clustering;
  * *static* voxels (occupied in most frames of this sequence only) are mostly
    parked vehicles → kept, and flagged as static;
  * everything else is *transient* foreground: moving traffic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .io import ROOT, Sequence, ground_z

CACHE_DIR = ROOT / "outputs" / "cache"


@dataclass
class GridSpec:
    radius: float = 80.0      # half-size of the square area of interest (m)
    cell: float = 0.25        # voxel xy size (m)
    zcell: float = 0.25       # voxel z size (m)
    zmax: float = 4.5         # highest height above ground considered (m)
    hmin: float = 0.25        # points below this height above ground are ground
    gcell: float = 0.5        # ground elevation map cell (m)

    @property
    def n(self) -> int:
        return int(round(2 * self.radius / self.cell))

    @property
    def nz(self) -> int:
        return int(round(self.zmax / self.zcell))

    @property
    def ng(self) -> int:
        return int(round(2 * self.radius / self.gcell))


class SceneModel:
    """Ground map + occupancy statistics for one crossing, built from several sequences."""

    def __init__(self, spec: GridSpec, ground_offset: np.ndarray, occupancy: dict[str, np.ndarray],
                 crossing: str = "crossing1"):
        self.spec = spec
        self.crossing = crossing
        self.ground_offset = ground_offset        # (ng, ng) float32, metres above quadratic model
        self.occupancy = occupancy                # seq -> (n, n, nz) float16 frequency in [0, 1]
        freqs = np.stack([o.astype(np.float32) for o in occupancy.values()])
        # permanent: seen in (almost) every frame of *every* sequence
        self.permanent = freqs.min(0) >= 0.35
        # grow a little so that the sparse edges of walls/trees don't survive as clusters
        self.permanent = ndimage.binary_dilation(self.permanent, iterations=1)

    # ------------------------------------------------------------------ building
    @classmethod
    def build(cls, sequences: list[str], spec: GridSpec | None = None, verbose: bool = True) -> "SceneModel":
        spec = spec or GridSpec()
        crossing = sequences[0].split("_")[2]
        # pass 1: ground elevation map (low percentile of heights near the ground)
        ng = spec.ng
        hist_bins = np.linspace(-0.6, 0.6, 49)
        hist = np.zeros((ng * ng, len(hist_bins) - 1), np.uint32)
        seqs = [Sequence(s) for s in sequences]
        for seq in seqs:
            for i in range(0, len(seq), 2):
                p = seq.load_frame(i)
                h = p[:, 2] - ground_z(p[:, :2], crossing)
                gx = ((p[:, 0] + spec.radius) / spec.gcell).astype(np.int64)
                gy = ((p[:, 1] + spec.radius) / spec.gcell).astype(np.int64)
                m = (gx >= 0) & (gx < ng) & (gy >= 0) & (gy < ng) & (np.abs(h) < 0.6)
                hb = np.clip(np.digitize(h[m], hist_bins) - 1, 0, len(hist_bins) - 2)
                np.add.at(hist, (gx[m] * ng + gy[m], hb), 1)
        cum = np.cumsum(hist, 1)
        total = cum[:, -1]
        centers = 0.5 * (hist_bins[1:] + hist_bins[:-1])
        idx = (cum >= np.maximum(1, 0.15 * total)[:, None]).argmax(1)
        off = np.where(total >= 20, centers[idx], np.nan).reshape(ng, ng)
        # fill holes (cells never seen, e.g. under permanently parked cars) from neighbours
        off = _fill_nan(off, iterations=12)
        off = np.nan_to_num(off, nan=0.0)
        off = np.clip(ndimage.median_filter(off, size=3), -0.3, 0.35).astype(np.float32)

        model = cls(spec, off, {"_": np.zeros((1,), np.float16)}, crossing)  # temporary for height()
        occupancy = {}
        for seq in seqs:
            cnt = np.zeros((spec.n, spec.n, spec.nz), np.uint16)
            for i in range(len(seq)):
                vox, ok = model.voxelize(seq.load_frame(i))
                v = np.zeros((spec.n, spec.n, spec.nz), bool)
                v[vox[ok, 0], vox[ok, 1], vox[ok, 2]] = True
                cnt += v
            occupancy[seq.name] = (cnt / len(seq)).astype(np.float16)
            if verbose:
                print(f"  occupancy {seq.name}: {len(seq)} frames")
        return cls(spec, off, occupancy, crossing)

    # ------------------------------------------------------------------ queries
    def height(self, pts: np.ndarray) -> np.ndarray:
        """Height above the (refined) ground for each point."""
        s = self.spec
        h = pts[:, 2] - ground_z(pts[:, :2], self.crossing)
        gx = np.clip(((pts[:, 0] + s.radius) / s.gcell).astype(np.int64), 0, s.ng - 1)
        gy = np.clip(((pts[:, 1] + s.radius) / s.gcell).astype(np.int64), 0, s.ng - 1)
        return h - self.ground_offset[gx, gy]

    def ground_at(self, xy: np.ndarray) -> np.ndarray:
        s = self.spec
        gx = np.clip(((xy[:, 0] + s.radius) / s.gcell).astype(np.int64), 0, s.ng - 1)
        gy = np.clip(((xy[:, 1] + s.radius) / s.gcell).astype(np.int64), 0, s.ng - 1)
        return ground_z(xy, self.crossing) + self.ground_offset[gx, gy]

    def voxelize(self, pts: np.ndarray, h: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Voxel indices (N, 3) and a validity mask for points inside the grid's height band."""
        s = self.spec
        h = self.height(pts) if h is None else h
        vox = np.empty((len(pts), 3), np.int64)
        vox[:, 0] = ((pts[:, 0] + s.radius) / s.cell).astype(np.int64)
        vox[:, 1] = ((pts[:, 1] + s.radius) / s.cell).astype(np.int64)
        vox[:, 2] = (h / s.zcell).astype(np.int64)
        ok = ((vox[:, 0] >= 0) & (vox[:, 0] < s.n) & (vox[:, 1] >= 0) & (vox[:, 1] < s.n)
              & (h > s.hmin) & (vox[:, 2] < s.nz))
        return vox, ok

    def classify_points(self, pts: np.ndarray, seq: str) -> dict[str, np.ndarray]:
        """Per-point masks: height above ground, ground, candidate (object band), permanent, foreground."""
        h = self.height(pts)
        vox, ok = self.voxelize(pts, h)
        v = vox[ok]
        permanent = np.zeros(len(pts), bool)
        permanent[ok] = self.permanent[v[:, 0], v[:, 1], v[:, 2]]
        freq = np.zeros(len(pts), np.float32)
        occ = self.occupancy.get(seq)
        if occ is not None:
            freq[ok] = occ[v[:, 0], v[:, 1], v[:, 2]]
        return {
            "h": h,
            "ground": h <= self.spec.hmin,
            "candidate": ok & ~permanent,
            "permanent": permanent,
            "foreground": ok & ~permanent & (freq < 0.6),
            "freq": freq,
        }

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ground_offset=self.ground_offset, crossing=self.crossing,
                            spec=np.array([self.spec.radius, self.spec.cell, self.spec.zcell, self.spec.zmax,
                                           self.spec.hmin, self.spec.gcell]),
                            **{f"occ__{k}": v for k, v in self.occupancy.items()})

    @classmethod
    def load(cls, path: Path) -> "SceneModel":
        d = np.load(path)
        spec = GridSpec(*[float(x) for x in d["spec"]])
        occ = {k[5:]: d[k] for k in d.files if k.startswith("occ__")}
        return cls(spec, d["ground_offset"], occ, str(d["crossing"]))

    @classmethod
    def get(cls, sequences: list[str], rebuild: bool = False) -> "SceneModel":
        crossing = sequences[0].split("_")[2]
        path = CACHE_DIR / f"scene_{crossing}.npz"
        if path.exists() and not rebuild:
            model = cls.load(path)
            if set(sequences) <= set(model.occupancy):
                return model
        model = cls.build(sequences)
        model.save(path)
        return model


def _fill_nan(a: np.ndarray, iterations: int = 10) -> np.ndarray:
    """Iteratively fill NaNs with the mean of their valid 3x3 neighbours."""
    a = a.copy()
    for _ in range(iterations):
        nan = np.isnan(a)
        if not nan.any():
            break
        v = np.where(nan, 0.0, a)
        w = (~nan).astype(np.float64)
        k = np.ones((3, 3))
        s = ndimage.convolve(v, k, mode="nearest")
        c = ndimage.convolve(w, k, mode="nearest")
        fill = nan & (c > 0)
        a[fill] = s[fill] / c[fill]
    return a
