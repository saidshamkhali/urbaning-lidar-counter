# urbaning-lidar-counter

Detecting, boxing and counting vehicles at a real urban intersection using only the
**fixed infrastructure LiDARs** of the [UrbanIng-V2X](https://github.com/thi-ad/UrbanIng-V2X)
cooperative-perception dataset (Ingolstadt, Germany; [paper](https://arxiv.org/abs/2510.23478)).

> Work in progress: the pipeline, evaluation and 3D viewer are being built.

Sequences used (all from `crossing1`):

- `20241126_0024_crossing1_09`
- `20241126_0008_crossing1_01`
- `20241127_0000_crossing1_00`

## Getting the data

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows
python scripts/download_data.py
```

The download is ~11 GB. Only the infrastructure LiDAR sweeps, calibration and time-sync files
are extracted into `data/dataset/`.

## Data licence

UrbanIng-V2X is released under **CC BY-NC-ND 4.0**. This repository contains **only code and
our own results**; no point clouds or other derived data are committed. Run the scripts to
reproduce everything locally.

## Licence

Code: MIT.
