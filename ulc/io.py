"""Loading UrbanIng-V2X infrastructure LiDAR sequences and the Lanelet2 HD map.

Only the fixed infrastructure LiDARs are used. Every sweep is transformed into the
dataset's global frame (local ENU metres around the crossing1 GPS origin) with the
`gTl` extrinsics from `calibration.json`, and the sweeps listed for one time step in
`timesync_info.csv` are concatenated into a single fused point cloud.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# GPS origin of each crossing's global frame (from the official devkit, urbaning/data/info.py)
GPS_ORIGINS = {
    "crossing1": (48.771731, 11.438043, 419.0),
    "crossing2": (48.772450, 11.441743, 419.0),
    "crossing3": (48.769060, 11.438518, 419.0),
}
# Quadratic ground model z = a x² + b y² + c xy + d x + e y + f (devkit)
GROUND_PARAMS = {
    "crossing1": (-2.17646045e-05, -2.54249987e-05, -8.18657797e-05, 3.81083323e-03, 8.94509838e-04, 4.57465017e-01),
    "crossing2": (-1.06222423e-05, -2.00344008e-05, 2.03389314e-05, -1.28649645e-03, -3.67867551e-04, 5.02631044e-01),
    "crossing3": (-1.21443378e-05, -1.32533272e-05, 3.58418801e-05, 1.78362570e-03, 2.67349162e-03, -7.98810367e-02),
}

# Columns of a fused point array
X, Y, Z, INTENSITY, LIDAR, TOFFSET = range(6)


def ground_z(xy: np.ndarray, crossing: str = "crossing1") -> np.ndarray:
    """Height of the road surface at the given (x, y) positions (devkit ground model)."""
    a, b, c, d, e, f = GROUND_PARAMS[crossing]
    x, y = xy[..., 0], xy[..., 1]
    return a * x * x + b * y * y + c * x * y + d * x + e * y + f


@dataclass
class Sequence:
    """One UrbanIng-V2X sequence restricted to its infrastructure LiDARs."""

    name: str
    data_dir: Path = DATA_DIR
    lidars: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.root = self.data_dir / "dataset" / self.name
        self.crossing = self.name.split("_")[2]
        calib = json.loads((self.root / "calibration.json").read_text())
        self.lidars = sorted(k for k in calib if k.startswith(self.crossing) and k.endswith("_lidar"))
        self.gTl = {k: np.asarray(calib[k]["extrinsics"]["gTl"], dtype=np.float64) for k in self.lidars}
        sync = pd.read_csv(self.root / "timesync_info.csv").set_index("Unnamed: 0")
        self.timestamps_ms: list[int] = [int(t) for t in sync.loc["timestamp_ms"]]
        self._files = {k: list(sync.loc[k]) for k in self.lidars}

    def __len__(self) -> int:
        return len(self.timestamps_ms)

    @property
    def label_path(self) -> Path:
        return self.data_dir / "labels" / f"{self.name}.json"

    @cached_property
    def lidar_positions(self) -> np.ndarray:
        """(L, 3) sensor origins in the global frame."""
        return np.stack([self.gTl[k][:3, 3] for k in self.lidars])

    def load_sweep(self, lidar: str, i: int) -> np.ndarray:
        """One LiDAR sweep in the global frame as (N, 6) float32 [x, y, z, intensity, lidar, t_offset]."""
        raw = np.load(self.root / lidar / self._files[lidar][i])
        xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float64)
        T = self.gTl[lidar]
        g = xyz @ T[:3, :3].T + T[:3, 3]
        out = np.empty((len(g), 6), dtype=np.float32)
        out[:, :3] = g
        out[:, INTENSITY] = raw["intensity"]
        out[:, LIDAR] = self.lidars.index(lidar)
        out[:, TOFFSET] = raw["time_offset_ms"]
        return out

    def load_frame(self, i: int) -> np.ndarray:
        """Fused point cloud of all infrastructure LiDARs at time step `i`, (N, 6) float32."""
        pts = np.concatenate([self.load_sweep(k, i) for k in self.lidars])
        ok = np.isfinite(pts[:, :3]).all(1)
        return pts[ok]

    def __iter__(self):
        for i in range(len(self)):
            yield self.load_frame(i)


# --------------------------------------------------------------------------- HD map


def _local_projector(crossing: str = "crossing1"):
    """lat/lon -> global frame (same convention as lanelet2's UtmProjector with the crossing origin)."""
    from pyproj import Proj

    lat0, lon0, _ = GPS_ORIGINS[crossing]
    utm = Proj(proj="utm", zone=32, ellps="WGS84")
    x0, y0 = utm(lon0, lat0)

    def project(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = utm(lon, lat)
        return np.asarray(x) - x0, np.asarray(y) - y0

    return project


def load_lanelet_map(path: Path | str = DATA_DIR / "crossings_lanelet2map.osm", crossing: str = "crossing1",
                     radius: float = 150.0) -> dict:
    """Parse the Lanelet2 OSM file into polylines and lanelet polygons near `crossing`.

    Returns the `map_<crossing>.json` structure from docs/DATA_FORMAT.md.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    project = _local_projector(crossing)

    ids, lats, lons = [], [], []
    for n in root.iter("node"):
        ids.append(n.get("id"))
        lats.append(float(n.get("lat")))
        lons.append(float(n.get("lon")))
    xs, ys = project(np.array(lats), np.array(lons))
    node_xy = {nid: (float(x), float(y)) for nid, x, y in zip(ids, xs, ys)}

    def tags(el) -> dict:
        return {t.get("k"): t.get("v") for t in el.findall("tag")}

    ways: dict[str, tuple[dict, np.ndarray]] = {}
    for w in root.iter("way"):
        pts = np.array([node_xy[nd.get("ref")] for nd in w.findall("nd") if nd.get("ref") in node_xy])
        if len(pts) >= 2:
            ways[w.get("id")] = (tags(w), pts)

    def near(pts: np.ndarray) -> bool:
        return bool((np.hypot(pts[:, 0], pts[:, 1]) < radius).any())

    def with_z(pts: np.ndarray) -> list[list[float]]:
        z = ground_z(pts, crossing)
        return [[round(float(x), 3), round(float(y), 3), round(float(zz), 3)] for (x, y), zz in zip(pts, z)]

    lines = []
    for t, pts in ways.values():
        if near(pts) and t.get("type"):
            lines.append({"kind": t["type"], "subtype": t.get("subtype", ""), "points": with_z(pts)})

    areas = []
    for r in root.iter("relation"):
        t = tags(r)
        if t.get("type") not in ("lanelet", "multipolygon"):
            continue
        members = {m.get("role"): m.get("ref") for m in r.findall("member") if m.get("type") == "way"}
        if t.get("type") == "lanelet" and "left" in members and "right" in members:
            if members["left"] not in ways or members["right"] not in ways:
                continue
            left, right = ways[members["left"]][1], ways[members["right"]][1]
            # bounds share direction; make sure the polygon doesn't self-intersect
            if np.linalg.norm(left[0] - right[0]) > np.linalg.norm(left[0] - right[-1]):
                right = right[::-1]
            poly = np.vstack([left, right[::-1]])
        elif t.get("type") == "multipolygon" and "outer" in members and members["outer"] in ways:
            poly = ways[members["outer"]][1]
        else:
            continue
        if near(poly):
            kind = t.get("subtype", t.get("area_type", t.get("type")))
            areas.append({"kind": kind, "polygon": [[round(float(x), 3), round(float(y), 3)] for x, y in poly]})

    return {"crossing": crossing, "lines": lines, "areas": areas}
