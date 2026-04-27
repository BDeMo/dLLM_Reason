#!/usr/bin/env python
"""PPT-quality figures for the logic chain. Larger fonts, cleaner layout,
minimal text density — designed to read at slide-deck distance.

Output → docs/figures/ppt/
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "ppt"
OUT.mkdir(parents=True, exist_ok=True)

# PPT-friendly defaults
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ──────────────── Figure 1: 4-stage logic flow (the master diagram) ────────
def fig_logic_flow():
    fig, ax = plt.subplots(figsize=(16, 8))

    stages = [
        {
            "title": "Stage 1\nA-axis",
            "subtitle": "Inference-only tweaks",
            "key": "pass@N: 92%\n(needs oracle)",
            "ceiling": "Ceiling-5\n(5 prompts)",
            "color": "#4A90E2",
            "x": 1.0,
        },
        {
            "title": "Stage 2\nT6 SFT",
            "subtitle": "Train on Qwen traces",
            "key": "greedy: 28%\nbest of 24 ckpts: 50%",
            "ceiling": "Hardset\n(166 prompts)",
            "color": "#E55934",
            "x": 5.5,
        },
        {
            "title": "Stage 3\nP2 decode",
            "subtitle": "T6 + sampling",
            "key": "pass@N: 66%\nSC@N: 38%",
            "ceiling": "Gap 27%\n(hidden signal)",
            "color": "#9B59B6",
            "x": 10.0,
        },
        {
            "title": "Stage 4\nT7 distill",
            "subtitle": "Bake pass@N → greedy",
            "key": "Target ≥ 45%\n(running)",
            "ceiling": "TBD",
            "color": "#27AE60",
            "x": 14.5,
        },
    ]

    for s in stages:
        # Stage box
        box = FancyBboxPatch((s["x"] - 1.6, 2.2), 3.2, 4.2,
                              boxstyle="round,pad=0.08",
                              facecolor=s["color"], alpha=0.85,
                              edgecolor="black", linewidth=2)
        ax.add_patch(box)
        # Title
        ax.text(s["x"], 5.85, s["title"], ha="center", va="center",
                fontsize=20, fontweight="bold", color="white")
        # Subtitle
        ax.text(s["x"], 5.05, s["subtitle"], ha="center", va="center",
                fontsize=12, color="white", style="italic")
        # Key result
        ax.text(s["x"], 4.0, s["key"], ha="center", va="center",
                fontsize=14, fontweight="bold", color="white")
        # Ceiling
        ax.text(s["x"], 2.7, "ceiling: " + s["ceiling"],
                ha="center", va="center", fontsize=11, color="white",
                style="italic")

    # Arrows BETWEEN boxes (in the gap below them, NOT cutting through tops)
    arrows = [
        (1, 2, "needs oracle\n→ try training"),
        (2, 3, "training caps 50%\n→ add sampling"),
        (3, 4, "27% gap\n→ distill it"),
    ]
    for src, dst, label in arrows:
        x_src = stages[src - 1]["x"] + 1.6
        x_dst = stages[dst - 1]["x"] - 1.6
        arrow = FancyArrowPatch((x_src, 4.3), (x_dst, 4.3),
                                arrowstyle="-|>", mutation_scale=28,
                                color="black", linewidth=2.5)
        ax.add_patch(arrow)
        # Label BELOW the arrow with a white box so it doesn't clash with stage box
        ax.text((x_src + x_dst) / 2, 4.3 - 1.0, label,
                ha="center", va="top", fontsize=10,
                style="italic", color="dimgray",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", edgecolor="lightgray", alpha=0.95))

    # Bottom narrative banner
    ax.text(8, 0.8,
            "Inference rescues 92% but needs oracle  →  Training reaches 50% greedy ceiling  →  "
            "Sampling lifts to 66% but SC only cashes 38%  →  Distill the gap?",
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF9E7",
                      edgecolor="goldenrod", linewidth=1.5))

    # Title
    ax.text(8, 7.6, "Logic Chain — A-axis → T6 → P2 → T7",
            ha="center", va="center", fontsize=24, fontweight="bold")

    ax.set_xlim(-0.5, 16.5); ax.set_ylim(-0.2, 8.4)
    ax.axis("off")
    out = OUT / "logic_flow.png"
    plt.savefig(out); plt.close()
    print(f"  → {out}")


# ──────────────── Figure 2: Capacity ladder (PPT version) ──────────────────
def fig_capacity_ladder_ppt():
    levels = [
        ("Untrained\ngreedy",          0.0,   "#BDC3C7"),
        ("Trained\ngreedy",            28.1,  "#3498DB"),
        ("Sampling\n+ majority",       38.4,  "#5DADE2"),
        ("Best-of\nckpts (T=0)",       50.2,  "#F39C12"),
        ("Sampling\n+ oracle",         65.9,  "#9B59B6"),
        ("Inference\nunion (A-axis)",  91.67, "#27AE60"),
        ("Combined\npotential",        100,   "white"),
    ]
    fig, ax = plt.subplots(figsize=(14, 7))
    xs = np.arange(len(levels))
    ys = [l[1] for l in levels]
    cs = [l[2] for l in levels]

    bars = ax.bar(xs, ys, color=cs, edgecolor="black", linewidth=2,
                  alpha=0.92)

    # Number on top of each bar
    for i, (b, y) in enumerate(zip(bars, ys)):
        if i < len(levels) - 1:
            ax.text(b.get_x() + b.get_width()/2, y + 1.5, f"{y:.1f}%",
                    ha="center", fontsize=18, fontweight="bold",
                    color=cs[i] if y > 5 else "black")
        else:
            ax.text(b.get_x() + b.get_width()/2, 50, "?",
                    ha="center", va="center",
                    fontsize=48, fontweight="bold", color="gray")

    # +10% (deploy gain) — arrow on RIGHT side of bar 1 (Trained greedy) → bar 2 (SC)
    ax.annotate("", xy=(2 - 0.45, 38.4), xytext=(1 + 0.45, 28.1),
                arrowprops=dict(arrowstyle="->", color="green", lw=2.2))
    ax.text(1.5, 19, "+10%\ndeployable\ntoday",
            fontsize=11, color="green", fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F0FFF0",
                      edgecolor="green", linewidth=1.2))

    # +27.5% (locked-in capacity) — arc from SC bar top to Sampling-oracle bar top
    # Use connectionstyle to curve OVER the Best-of-ckpts orange bar between them
    ax.annotate("", xy=(4, 65.0), xytext=(2, 39),
                arrowprops=dict(arrowstyle="->", color="darkred", lw=2.5,
                                connectionstyle="arc3,rad=-0.35"))
    ax.text(0.5, 95,
            "+27.5% locked-in\ncapacity (oracle only)",
            fontsize=13, color="darkred", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF5F5",
                      edgecolor="darkred", linewidth=1.5))

    ax.set_xticks(xs)
    ax.set_xticklabels([l[0] for l in levels], fontsize=12)
    ax.set_ylabel("fail rescue (% of fail set)", fontsize=14)
    ax.set_title("Capacity ladder — what each axis can deliver",
                 fontsize=20, fontweight="bold", pad=15)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    out = OUT / "capacity_ladder.png"
    plt.savefig(out); plt.close()
    print(f"  → {out}")


# ──────────────── Figure 3: pass vs SC gap (the key motivator) ─────────────
def fig_pass_vs_sc_dramatic():
    """One-shot dramatic visualization of the 27% gap on best ckpt."""
    fig, ax = plt.subplots(figsize=(11, 7))
    ckpt = "Best T6 ckpt"
    bars_data = [
        ("Greedy\n(T=0 pass@1)",     28.1, "#95A5A6"),
        ("Majority Vote\n(SC@8)",    38.4, "#3498DB"),
        ("Oracle Sampling\n(pass@8)",65.9, "#9B59B6"),
    ]
    xs = np.arange(len(bars_data))
    ys = [b[1] for b in bars_data]
    cs = [b[2] for b in bars_data]
    bars = ax.bar(xs, ys, color=cs, edgecolor="black", linewidth=2.5,
                  alpha=0.92, width=0.55)
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width()/2, y + 1.5, f"{y:.1f}%",
                ha="center", fontsize=22, fontweight="bold")

    # Two annotations: "deployed today" and "the hidden gap"
    ax.annotate("", xy=(1, 38.4), xytext=(0, 28.1),
                arrowprops=dict(arrowstyle="->", color="green", lw=2.5))
    ax.text(0.5, 35.5, "+10%\ndeployable", ha="center", fontsize=14,
            color="green", fontweight="bold")

    ax.annotate("", xy=(2, 65.9), xytext=(1, 38.4),
                arrowprops=dict(arrowstyle="->", color="darkred", lw=2.5))
    ax.text(1.5, 53, "+27.5%\nlocked in\ncapacity", ha="center", fontsize=14,
            color="darkred", fontweight="bold")

    # Tag baseline above column 0
    ax.text(0, 28.1 + 7, "current\nproduction", ha="center", fontsize=10,
            color="dimgray", style="italic")
    ax.text(2, 65.9 + 7, "needs oracle\n(can't deploy)", ha="center",
            fontsize=10, color="purple", style="italic")

    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars_data], fontsize=14)
    ax.set_ylabel("fail rescue rate (%)", fontsize=15)
    ax.set_title("The 27% gap — model has the capacity, decoding can't tap it\n"
                 f"({ckpt}, full scope = 331 prompts)",
                 fontsize=18, fontweight="bold", pad=15)
    ax.set_ylim(0, 80)
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    out = OUT / "pass_sc_gap_drama.png"
    plt.savefig(out); plt.close()
    print(f"  → {out}")


# ──────────────── Figure 4: Cross-axis disjointness (PPT) ──────────────────
def fig_disjoint_ppt():
    fig, ax = plt.subplots(figsize=(13, 8))
    LEFT = (-1.8, 0); LR = 1.8
    RIGHT = (3.0, 0); RR = 3.0

    c1 = Circle(LEFT, LR, alpha=0.45, color="#3498DB", linewidth=3,
                edgecolor="navy")
    c2 = Circle(RIGHT, RR, alpha=0.45, color="#E55934", linewidth=3,
                edgecolor="darkred")
    ax.add_patch(c1); ax.add_patch(c2)

    # Left labels ABOVE circle, inside number, prompts under
    ax.text(LEFT[0], LEFT[1] + 2.5, "Inference\ncannot rescue",
            ha="center", fontsize=18, fontweight="bold", color="navy")
    ax.text(LEFT[0], LEFT[1] + 0.0, "5", ha="center", fontsize=44,
            fontweight="bold", color="navy")
    ax.text(LEFT[0], LEFT[1] - 1.0, "prompts", ha="center", fontsize=12,
            color="navy")

    # Right labels
    ax.text(RIGHT[0], RIGHT[1] + 3.7, "Training\ncannot rescue",
            ha="center", fontsize=18, fontweight="bold", color="darkred")
    ax.text(RIGHT[0], RIGHT[1] + 0.0, "166", ha="center", fontsize=44,
            fontweight="bold", color="darkred")
    ax.text(RIGHT[0], RIGHT[1] - 1.4, "prompts", ha="center", fontsize=12,
            color="darkred")

    # Arrow + label
    TOUCH = 0
    ax.annotate("intersection is empty",
                xy=(TOUCH, 0), xytext=(TOUCH, -3.5),
                fontsize=18, color="green", fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color="green", lw=2.5))

    # Punchline well above (separated from "Inference cannot rescue" label)
    ax.text(0.6, 5.5,
            "Two ceilings are orthogonal → combined > either alone",
            ha="center", fontsize=18, fontweight="bold")

    ax.set_xlim(-5, 7); ax.set_ylim(-4.5, 6.2)
    ax.set_aspect("equal"); ax.axis("off")
    out = OUT / "disjoint_axes.png"
    plt.savefig(out); plt.close()
    print(f"  → {out}")


# ────────────────────────────── Main ───────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating PPT figures → {OUT}")
    fig_logic_flow()
    fig_capacity_ladder_ppt()
    fig_pass_vs_sc_dramatic()
    fig_disjoint_ppt()
    print("Done.")
