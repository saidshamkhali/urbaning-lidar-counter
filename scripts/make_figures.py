"""Result figures for the README (from outputs/results, i.e. our own detections and metrics).

    python scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
FIG = ROOT / "docs" / "figures"
SEQUENCES = ["20241126_0024_crossing1_09", "20241126_0008_crossing1_01", "20241127_0000_crossing1_00"]

BG, PANEL, GRID, TEXT, MUTED = "#070a10", "#0d121b", "#1c2533", "#dfe7f2", "#7d8da3"
CYAN, AMBER, VIOLET, SLATE = "#35e0c8", "#ffb547", "#9d8cff", "#55657a"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT, "grid.color": GRID,
    "axes.grid": True, "grid.linewidth": 0.6, "font.size": 10, "axes.titleweight": "bold",
    "axes.titlesize": 11, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})


def short(seq: str) -> str:
    return seq.split("_", 1)[1]


def load(seq: str, name: str) -> dict:
    return json.loads((RESULTS / seq / f"{name}.json").read_text())


def counts_over_time() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    for ax, seq in zip(axes, SEQUENCES):
        det, gt = load(seq, "detections"), load(seq, "gt")
        t = np.arange(len(det["counts"])) / 10.0
        ax.plot(t, [c["vehicle"] for c in gt["counts"]], color=SLATE, lw=1, ls=":", label="all labels (incl. never visible)")
        ax.plot(t, [c["vehicle"] for c in gt["counts_visible"]], color=VIOLET, lw=1.2, ls="--", label="GT visible in frame (≥5 pts)")
        ax.plot(t, [c["vehicle"] for c in gt["counts_trackable"]], color=AMBER, lw=1.6, label="GT in coverage (trackable)")
        ax.plot(t, [c["vehicle"] for c in det["counts"]], color=CYAN, lw=2.2, label="ours (tracked)")
        ax.set_title(short(seq), loc="left")
        ax.set_ylabel("vehicles in frame")
        ax.set_ylim(0, None)
    axes[0].legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.45), fontsize=9)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(FIG / "counts_over_time.png", dpi=150)
    plt.close(fig)


def pr_and_range() -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = [CYAN, AMBER, VIOLET]
    width = 0.26
    for k, (seq, col) in enumerate(zip(SEQUENCES, colors)):
        m = load(seq, "metrics")["detection"]
        pr = m["pr_curve"]
        a1.plot(pr["recall"], pr["precision"], color=col, lw=2,
                label=f"{short(seq)}  AP@0.3 = {m['vehicle']['ap']['0.3']:.2f}")
        rb = m["by_range"]
        x = np.arange(len(rb))
        a2.bar(x + (k - 1) * width, [b["recall"] or 0 for b in rb], width, color=col, alpha=0.9, label=short(seq))
        a2.set_xticks(x, [b["range"] + " m" for b in rb])
    a1.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="recall", ylabel="precision")
    a1.set_title("Vehicle detection – precision/recall (BEV IoU 0.3)", loc="left")
    a1.legend(loc="lower left", fontsize=9)
    a2.set(ylim=(0, 1), ylabel="recall", xlabel="distance from junction centre")
    a2.set_title("Recall by distance", loc="left")
    a2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "detection_quality.png", dpi=150)
    plt.close(fig)


def unique_counts() -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(SEQUENCES))
    vals = []
    for seq in SEQUENCES:
        det, gt = load(seq, "detections"), load(seq, "gt")
        vals.append((det["unique"]["vehicle"], gt["unique_trackable"]["vehicle"], gt["unique"]["vehicle"]))
    vals = np.array(vals)
    w = 0.27
    for k, (lab, col) in enumerate([("ours", CYAN), ("GT in coverage", AMBER), ("all labels", SLATE)]):
        bars = ax.bar(x + (k - 1) * w, vals[:, k], w, color=col, label=lab)
        ax.bar_label(bars, color=TEXT, fontsize=9, padding=2)
    ax.set_xticks(x, [short(s) for s in SEQUENCES])
    ax.set_ylabel("unique vehicles in 20 s")
    ax.set_title("Unique vehicles counted per sequence", loc="left")
    ax.legend(ncol=3, fontsize=9, loc="upper right")
    ax.set_ylim(0, vals.max() * 1.2)
    fig.tight_layout()
    fig.savefig(FIG / "unique_counts.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    counts_over_time()
    pr_and_range()
    unique_counts()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
