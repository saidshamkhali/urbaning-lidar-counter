"""Download and extract the UrbanIng-V2X sequences used in this project.

Only the infrastructure LiDAR sweeps, calibration and time-sync files are
extracted from the archives; camera images and vehicle sensors are skipped.

    python scripts/download_data.py                 # the 3 default sequences
    python scripts/download_data.py --seq 20241126_0008_crossing1_01
    python scripts/download_data.py --keep-archives # don't delete .7z after extraction

Data: UrbanIng-V2X (Harvard Dataverse, doi:10.7910/DVN/A9LPY7), CC BY-NC-ND 4.0.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DATAVERSE = "https://dataverse.harvard.edu"
DOI = "doi:10.7910/DVN/A9LPY7"
DEFAULT_SEQUENCES = [
    "20241126_0024_crossing1_09",
    "20241126_0008_crossing1_01",
    "20241127_0000_crossing1_00",
]
EXTRA_FILES = ["crossings_lanelet2map.osm"]
ROOT = Path(__file__).resolve().parents[1]
HEADERS = {"User-Agent": "urbaning-lidar-counter/0.1 (+https://github.com/saidshamkhali/urbaning-lidar-counter)"}


def list_files() -> dict[str, dict]:
    url = f"{DATAVERSE}/api/datasets/:persistentId/?persistentId={DOI}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
        meta = json.load(r)
    out = {}
    for f in meta["data"]["latestVersion"]["files"]:
        df = f["dataFile"]
        out[df["filename"]] = {"id": df["id"], "size": df.get("filesize", 0), "dir": f.get("directoryLabel", "")}
    return out


def download(file_id: int, size: int, dest: Path) -> None:
    """Resumable download using HTTP Range requests."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if size and have == size:
        print(f"  ok   {dest.name} (already downloaded)")
        return
    for attempt in range(20):
        have = dest.stat().st_size if dest.exists() else 0
        req = urllib.request.Request(f"{DATAVERSE}/api/access/datafile/{file_id}", headers=HEADERS)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "ab" if have else "wb") as fh:
                if have and r.status != 206:  # server ignored the range: restart
                    fh.truncate(0)
                    have = 0
                t0, done, last = time.time(), have, 0.0
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if time.time() - last > 10:
                        last = time.time()
                        rate = (done - have) / max(1e-6, last - t0) / 1e6
                        print(f"  ...  {dest.name} {done / 1e9:5.2f}/{size / 1e9:.2f} GB  {rate:5.1f} MB/s", flush=True)
            if not size or dest.stat().st_size == size:
                print(f"  done {dest.name}", flush=True)
                return
        except Exception as exc:  # network hiccup: retry with resume
            print(f"  retry {dest.name} after error: {exc}", flush=True)
            time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f"failed to download {dest.name}")


def find_7z() -> str:
    for cand in ("7z", "7za", r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"):
        if shutil.which(cand) or Path(cand).exists():
            return cand
    sys.exit("7-Zip not found. Install it (https://www.7-zip.org) and re-run.")


def extract(archive: Path, out_dir: Path, seq: str) -> None:
    """Extract only infrastructure LiDAR folders + calibration/time-sync metadata."""
    patterns = [f"{seq}/crossing*_lidar/*", f"{seq}/calibration.json", f"{seq}/timesync_info.csv", f"{seq}/weather_data.json"]
    cmd = [find_7z(), "x", str(archive), f"-o{out_dir}", "-y", "-r"] + patterns
    print("  extracting", archive.name, "->", out_dir, flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", nargs="*", default=DEFAULT_SEQUENCES, help="sequence ids to fetch")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--keep-archives", action="store_true")
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()

    files = list_files()
    raw = args.data_dir / "raw"
    for name in EXTRA_FILES:
        download(files[name]["id"], files[name]["size"], args.data_dir / name)

    for seq in args.seq:
        print(f"[{seq}]", flush=True)
        label = f"{seq}.json"
        download(files[label]["id"], files[label]["size"], args.data_dir / "labels" / label)
        parts = sorted(n for n in files if n.startswith(f"{seq}.7z."))
        if not parts:
            sys.exit(f"sequence {seq} not found on Dataverse")
        if (args.data_dir / "dataset" / seq / "calibration.json").exists():
            print("  already extracted")
            continue
        for p in parts:
            download(files[p]["id"], files[p]["size"], raw / p)
        if args.no_extract:
            continue
        extract(raw / parts[0], args.data_dir / "dataset", seq)
        if not args.keep_archives:
            for p in parts:
                (raw / p).unlink(missing_ok=True)
    print("all done")


if __name__ == "__main__":
    main()
