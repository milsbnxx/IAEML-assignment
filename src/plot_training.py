"""Plot measured PPO updates from a training directory."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


PANELS = (("mean_episode_return", "Episode return"), ("success_rate", "Episodes reaching the goal"),
          ("policy_loss", "Policy loss"), ("value_loss", "Value loss"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/training"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    with (args.run_dir / "updates.csv").open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        parser.error("The training log has no updates.")
    maps = sorted({int(r["map_id"]) for r in rows})
    colors = ("#156b74", "#b15128", "#6654a3")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8))
    fig.subplots_adjust(left=0.085, right=0.98, top=0.82, bottom=0.10, hspace=0.50, wspace=0.24)
    fig.suptitle("PPO learning curves", x=0.085, y=0.97, ha="left", fontsize=22, weight="bold")
    fig.text(0.085, 0.89, "Faint lines show each update. Bold lines average eight updates per map.",
             fontsize=10, color="#435459")
    first_full = next((int(r["transitions"]) / 1000 for r in rows if float(r["short_probability"]) == 0), None)
    for ax, (field, label) in zip(axes.flat, PANELS):
        for i, map_id in enumerate(maps):
            part = [r for r in rows if int(r["map_id"]) == map_id and r[field] != ""]
            if not part:
                continue
            x = np.array([int(r["transitions"]) / 1000 for r in part])
            y = np.array([float(r[field]) for r in part])
            weights = np.array([int(r["episodes"]) if field in ("mean_episode_return", "success_rate") else 1
                                for r in part])
            window = np.ones(min(8, len(y)))
            smoothed = np.convolve(y * weights, window, "valid") / np.convolve(weights, window, "valid")
            color = colors[i % len(colors)]
            ax.plot(x, y, color=color, lw=0.8, alpha=0.22)
            ax.plot(x[len(window)-1:], smoothed, color=color, lw=2.1, label=f"Map {map_id}")
        if first_full is not None:
            ax.axvline(first_full, color="#657880", ls=":", lw=1)
        ax.set_title(label, loc="left", fontsize=13, weight="bold")
        ax.set_xlabel("Environment steps (thousands)", fontsize=9)
        ax.grid(axis="y", alpha=0.19)
        ax.spines[["top", "right"]].set_visible(False)
        if field == "success_rate":
            ax.set_ylim(-0.05, 1.05)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=8)
    if first_full is not None:
        fig.text(0.085, 0.02, "The dotted line marks the first update with full-route resets only.",
                 fontsize=9, color="#435459")
    dest = args.output or args.run_dir / "training_curves.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160, facecolor="white")
    plt.close(fig)
    print(dest)


if __name__ == "__main__":
    main()
