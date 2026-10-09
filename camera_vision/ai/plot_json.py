"""Charting tool for JSON outputs from analyze_video
using seaborn. Draws heat_mean (left axis) and
tiles_above_threshold (right axis) against frame index,
with the peak frame marked.

Usage::

    python -m ai.plot_json recordings\\clip_annotated.json
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a graph with JSON data"
    )
    parser.add_argument("json_path", help="the _annotated.json")
    parser.add_argument(
        "--title", default=None,
        help="chart title, default is video name",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.json_path)
    records = json.loads(src.read_text())
    frames = [r["frame"] for r in records]
    mean = [r["heat_mean"] for r in records]
    above = [r["tiles_above_threshold"] for r in records]

    seaborn.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 4.2))
    seaborn.lineplot(x=frames, y=mean, ax=ax1,
                     color="#D85A30", linewidth=2)
    ax1.set_xlabel("frame index")
    ax1.set_ylabel("heat_mean", color="#D85A30")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    seaborn.lineplot(x=frames, y=above, ax=ax2,
                     color="#534AB7", linewidth=1.5, alpha=0.8)
    ax2.set_ylabel("tiles above threshold", color="#534AB7")
    ax2.grid(False)

    peak = frames[mean.index(max(mean))]
    ax1.axvline(peak, color="gray", linestyle="--", linewidth=1)
    ax1.set_title(args.title if args.title else src.stem)
    ax1.text(
        0.02, 0.95,
        f"peak heat_mean {max(mean):.3f} at frame {peak}",
        transform=ax1.transAxes,
        va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white",
                  edgecolor="#534AB7", alpha=0.8),
    )

    fig.tight_layout()

    out = src.with_name(src.stem + "_plot.png")
    if args.title:
        temp = re.sub(r"[^\w-]+", "_", args.title).strip("_").lower()
        out = src.with_name(f"{temp}_plot.png")

    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    main()
