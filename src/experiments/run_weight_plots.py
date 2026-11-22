"""Figures from the pretrained-checkpoint weight statistics.

Reads analysis/checkpoint_weight_stats.csv and produces:
  * weight-std vs fan-in scatter overlaid with Xavier / He reference bands,
  * per-family boxplot of conv weight std.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))


def main() -> None:
    figs = os.path.join(ROOT, "analysis", "figs")
    os.makedirs(figs, exist_ok=True)
    df = pd.read_csv(os.path.join(ROOT, "analysis", "checkpoint_weight_stats.csv"))
    w = df[df["role"].isin(["conv", "head", "gate", "qk"])].copy()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Scatter: empirical std vs fan-in, reference theory curves.
    fig, ax = plt.subplots(figsize=(8, 5))
    fi = np.linspace(max(10, w["fan_in"].min()), w["fan_in"].max(), 100)
    ax.plot(fi, np.sqrt(2.0 / fi), "--", color="gray", label="He/Kaiming std")
    ax.plot(fi, np.sqrt(2.0 / (fi + fi)), ":", color="black", label="Xavier std (square)")
    for family, sub in w.groupby("family"):
        ax.scatter(sub["fan_in"], sub["std"], s=14, alpha=0.7, label=family)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("fan_in")
    ax.set_ylabel("weight std (trained checkpoint)")
    ax.set_title("Pretrained dehazing weights: empirical std vs fan-in")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(figs, "weight_std_vs_fanin.png"), dpi=150)
    plt.close(fig)

    # Boxplot per family/split.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = []
    data = []
    for (fam, split), sub in w.groupby(["family", "split"]):
        labels.append(f"{fam}\n{split}")
        data.append(sub["std"].values)
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("conv weight std")
    ax.set_title("Per-checkpoint conv weight std distribution")
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(figs, "weight_std_boxplot.png"), dpi=150)
    plt.close(fig)

    print("saved figures to", figs)


if __name__ == "__main__":
    main()
