#!/usr/bin/env python
"""T6 vs T7 v1 vs T7 v2 comparison figures (for slides / discussion).

Shows the full picture: greedy / pass@8 / SC@8 across the three checkpoints.
Output → docs/figures/ppt/slides/
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "ppt" / "slides"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 18, "axes.labelsize": 14,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 13, "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def fig_t6_vs_t7():
    """Three groups (T6 / T7 v1 / T7 v2), 3 bars each (greedy / SC / pass)."""
    groups = ["T6\nstep_336", "T7 v1\n(shortest, 6 ep)", "T7 v2\n(first, 2 ep)"]
    # (greedy, SC@8 best, pass@8 T=1.0)
    greedy = [28.1, 27.2, 27.5]
    sc8    = [38.4, 36.3, 29.6]
    pass8  = [65.9, 59.2, 56.8]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(groups))
    w = 0.27

    b1 = ax.bar(x - w, greedy, w, color="#95A5A6", edgecolor="black",
                linewidth=1.5, label="Greedy (T=0)")
    b2 = ax.bar(x,     sc8,    w, color="#3498DB", edgecolor="black",
                linewidth=1.5, label="SC@8 (deployable)")
    b3 = ax.bar(x + w, pass8,  w, color="#9B59B6", edgecolor="black",
                linewidth=1.5, label="pass@8 (oracle)")

    for bars, vals in [(b1, greedy), (b2, sc8), (b3, pass8)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 1.0, f"{v:.1f}",
                    ha="center", fontsize=12, fontweight="bold")

    # Highlight regressions vs T6
    for i in [1, 2]:
        for j, (vals, base) in enumerate([(greedy, 28.1), (sc8, 38.4), (pass8, 65.9)]):
            delta = vals[i] - base
            if delta < -1.0:
                ax.text(x[i] + (j - 1) * w, vals[i] - 4.5, f"{delta:+.1f}",
                        ha="center", fontsize=10, color="darkred",
                        fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("fail rescue (%)")
    ax.set_ylim(0, 78)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    out = OUT / "s6_t6_vs_t7.png"
    plt.savefig(out); plt.close()
    print(f"  → {out}")


def fig_t7_failure_mode():
    """Single chart showing what self-distill does to the distribution:
    greedy nearly unchanged but pass@N AND SC@N both drop."""
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ["Greedy\n(deployable mode)",
               "SC@8\n(majority of 8 samples)",
               "pass@8\n(any of 8 samples)"]
    t6_vals  = [28.1, 38.4, 65.9]
    t7v2_vals = [27.5, 29.6, 56.8]

    x = np.arange(len(metrics))
    w = 0.36
    ax.bar(x - w/2, t6_vals,  w, color="#27AE60", edgecolor="black",
           linewidth=1.5, label="T6 (token-level SFT)")
    ax.bar(x + w/2, t7v2_vals, w, color="#E74C3C", edgecolor="black",
           linewidth=1.5, label="T7 v2 (self-distill SFT)")

    for i, (v6, v7) in enumerate(zip(t6_vals, t7v2_vals)):
        ax.text(x[i] - w/2, v6 + 1.2, f"{v6:.1f}",
                ha="center", fontsize=12, fontweight="bold")
        ax.text(x[i] + w/2, v7 + 1.2, f"{v7:.1f}",
                ha="center", fontsize=12, fontweight="bold")
        delta = v7 - v6
        ax.annotate("", xy=(x[i] + w/2, v7), xytext=(x[i] + w/2, v6),
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=2))
        ax.text(x[i] + w/2 + 0.18, (v6 + v7)/2, f"{delta:+.1f}",
                fontsize=11, color="darkred", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("fail rescue (%)")
    ax.set_ylim(0, 78)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    out = OUT / "s7_t7_failure_mode.png"
    plt.savefig(out); plt.close()
    print(f"  → {out}")


if __name__ == "__main__":
    print(f"Generating T7 comparison figures → {OUT}")
    fig_t6_vs_t7()
    fig_t7_failure_mode()
    print("Done.")
