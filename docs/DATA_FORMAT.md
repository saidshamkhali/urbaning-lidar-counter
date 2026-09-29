# Data contract

This file is the contract between the Python pipeline (`ulc/`) and the web viewer (`web/`).
Everything described here is **generated locally** by `scripts/export_web.py` into
`web/public/data/` and is git-ignored (the UrbanIng-V2X licence is CC BY-NC-ND 4.0, so derived
point clouds are never committed or published).

## Coordinate frame

All geometry is in the dataset's **global frame**: a local metric ENU frame whose origin is the
GPS origin of `crossing1` (48.771731 N, 11.438043 E, 419 m). `x` = east, `y` = north, `z` = up,
metres. The ground at the intersection is at roughly `z ≈ 0.45`.

## Boxes

A box is 7 numbers: `[x, y, z, l, w, h, yaw]`

* `x, y, z` – centre of the box (z is the *centre*, not the bottom)
* `l, w, h` – length (along heading), width, height in metres
* `yaw` – heading in radians, counter-clockwise around +z, measured from +x

In Python, arrays of boxes are `np.ndarray` of shape `(N, 7)`, `float64`.

## Classes

Fine classes (as in the UrbanIng-V2X labels, lower-cased):
`car, van, truck, bus, trailer, motorcycle, cyclist, escooter, pedestrian, other`

Groups used for counting:

| group     | classes                                       |
|-----------|-----------------------------------------------|
| `vehicle` | car, van, truck, bus, trailer, motorcycle     |
| `vru`     | cyclist, escooter, pedestrian (other road users) |
| `other`   | other                                         |

The detector predicts a reduced set: `car, van, truck, bus, cyclist, pedestrian`
(`truck` also covers trailers). Colours in the viewer:
car/van = cyan-green, truck/bus/trailer = amber/yellow, cyclist/pedestrian = magenta/pink.

## Files in `web/public/data/`

```
index.json                 list of sequences
map_crossing1.json         HD-map polylines/areas (from the Lanelet2 map)
<seq>/meta.json            per-sequence metadata
<seq>/points/000000.bin    fused point cloud for frame 0 (binary, see below)
<seq>/points/000001.bin    ...
<seq>/detections.json      our detections + tracks + counts
<seq>/gt.json              ground-truth labels, same schema as detections.json
<seq>/metrics.json         evaluation results
```

### `index.json`

```json
{
  "generated_at": "2026-09-29T14:00:00Z",
  "sequences": [
    {
      "id": "20241126_0024_crossing1_09",
      "crossing": "crossing1",
      "n_frames": 200,
      "duration_s": 20.0,
      "unique_vehicles": 41,        // detector, unique tracked vehicles
      "unique_vehicles_gt": 44      // ground truth
    }
  ]
}
```

### `<seq>/meta.json`

```json
{
  "id": "20241126_0024_crossing1_09",
  "crossing": "crossing1",
  "map": "map_crossing1.json",
  "fps": 10,
  "n_frames": 200,
  "timestamps": [1732640002.0, 1732640002.1],
  "bounds": {"xmin": -80, "xmax": 80, "ymin": -80, "ymax": 80, "zmin": -2, "zmax": 12},
  "lidars": [{"name": "crossing1_11_lidar", "index": 0, "position": [x, y, z]}],
  "point_format": {
    "stride_bytes": 8,
    "scale": 0.01,
    "origin": [0.0, 0.0, 0.0],
    "fields": ["x:int16", "y:int16", "z:int16", "intensity:uint8", "flags:uint8"]
  },
  "frames": [{"file": "points/000000.bin", "n_points": 123456}]
}
```

### Point files `<seq>/points/NNNNNN.bin`

Little-endian, `n_points × 8` bytes, one record per point:

| bytes | type    | meaning                                            |
|-------|---------|----------------------------------------------------|
| 0–1   | int16   | x = value × scale + origin[0]  (scale = 0.01 m)    |
| 2–3   | int16   | y = value × scale + origin[1]                      |
| 4–5   | int16   | z = value × scale + origin[2]                      |
| 6     | uint8   | intensity, 0–255 (normalised)                      |
| 7     | uint8   | flags (bit field, below)                           |

`flags`:

* bits 0–1: LiDAR index (0–3), same order as `meta.lidars`
* bit 2 (`4`): foreground (differs from the learned static background)
* bit 3 (`8`): ground point
* bit 4 (`16`): inside a detected box

### `<seq>/detections.json` and `<seq>/gt.json`

```json
{
  "source": "detector",            // or "gt"
  "frames": [                        // one list per frame, index = frame number
    [
      {"id": 12, "cls": "car", "box": [x, y, z, l, w, h, yaw],
       "score": 0.93, "moving": true, "speed": 8.4, "n_points": 312}
    ]
  ],
  "tracks": {
    "12": {"cls": "car", "group": "vehicle", "first": 0, "last": 143,
           "moving": true, "max_speed": 11.2, "n_frames": 144}
  },
  "counts": [                        // one entry per frame
    {"vehicle": 31, "vru": 6, "other": 0, "moving": 9, "parked": 22,
     "by_class": {"car": 27, "van": 2, "truck": 1, "bus": 1}}
  ],
  "unique": {"vehicle": 41, "vru": 9, "other": 0,
             "by_class": {"car": 35, "van": 3, "truck": 2, "bus": 1}}
}
```

`speed` is m/s (estimated by the tracker for detections; from consecutive label positions for GT).
`moving` is true when the track's speed is above 1.0 m/s for the current frame.
`score` is 1.0 for GT; `n_points` may be 0 for GT.

### `<seq>/metrics.json`

```json
{
  "detection": {
    "iou_thresholds": [0.3, 0.5],
    "vehicle": {"ap": {"0.3": 0.81, "0.5": 0.66}, "precision": 0.9, "recall": 0.8, "f1": 0.85},
    "per_class": {"car": {"ap": {"0.3": 0.8, "0.5": 0.6}, "n_gt": 5000, "n_pred": 4800}},
    "by_range": [{"range": "0-20", "recall": 0.95}, {"range": "20-40", "recall": 0.85}],
    "pr_curve": {"recall": [0.0, 0.1], "precision": [1.0, 0.99]}
  },
  "counting": {
    "mae": 1.2, "rmse": 1.6, "bias": -0.8, "within_1": 0.72,
    "pred": [31, 30], "gt": [33, 33]
  },
  "tracking": {"mota": 0.7, "motp": 0.8, "id_switches": 4, "unique_pred": 41, "unique_gt": 44}
}
```

### `map_crossing1.json`

```json
{
  "crossing": "crossing1",
  "lines": [
    {"kind": "curbstone", "subtype": "high", "points": [[x, y, z], [x, y, z]]}
  ],
  "areas": [
    {"kind": "road", "polygon": [[x, y], [x, y]]}
  ]
}
```

`lines[].kind` is the Lanelet2 `type` tag of the way (`curbstone`, `road_border`, `line_thin`,
`line_thick`, `virtual`, `stop_line`, `zebra_marking`, …) and `subtype` its `subtype` tag
(`solid`, `dashed`, `high`, `low`, …). `areas` are lanelet polygons (left bound + reversed right
bound); `kind` is the lanelet `subtype` (`road`, `walkway`, `bicycle_lane`, `crosswalk`, …).
