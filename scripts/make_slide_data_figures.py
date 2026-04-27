#!/usr/bin/env python
"""Pure data figures for PPT slides 2-5 (slide 1 already done by user).

Each figure shows ONLY data — no captions, no internal text explaining
the result. User adds descriptive text in PPT.

Output → docs/figures/ppt/slides/
"""
from __future__ import annotations
from pathlib import Path
import json
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "ppt" / "slides"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 17,
    "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 14, "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.4,
})


# ─────────────── Slide 2: A-axis methods (60-scope) ───────────────────────
def slide2_a_axis():
    """All A-axis methods + pass@N + union. Color-coded by category."""
    data = [
        ("A1\nDAG edge",        0.0,  "#95A5A6"),
        ("A2\ntoken",           0.0,  "#95A5A6"),
        ("A3\nspan",            0.0,  "#95A5A6"),
        ("A4\nblock",           8.33, "#7F8C8D"),
        ("A5\ntemplate",       13.33, "#9B59B6"),
        ("A6\ngen length",     20.0,  "#9B59B6"),
        ("H3\npass@N",         86.67, "#8E44AD"),
        ("union",              91.67, "#27AE60"),
    ]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    xs = np.arange(len(data))
    ys = [d[1] for d in data]
    cs = [d[2] for d in data]
    bars = ax.bar(xs, ys, color=cs, edgecolor="black", linewidth=1.6,
                  alpha=0.92, width=0.7)
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width()/2, y + 2, f"{y:.1f}%",
                ha="center", fontsize=15, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([d[0] for d in data], fontsize=14)
    ax.set_ylabel("rescue rate (%)")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    out = OUT / "s2_a_axis_methods.png"
    plt.savefig(out); plt.close()
    print(f"  → {out}")


# ─────────────── Slide 3: T6 24-ckpt landscape + union saturation ─────────
def slide3_t6_landscape():
    """Two-panel: (a) per-ckpt greedy rescue, (b) cumulative union curve."""
    per_ckpt_path = ROOT / "runs/validation/t6_hardset/per_ckpt.json"
    if not per_ckpt_path.exists():
        print("  (skip — no per_ckpt.json)"); return
    raw = json.loads(per_ckpt_path.read_text(encoding="utf-8"))
    inner = raw.get("per_ckpt", raw)
    entries = []
    for label, info in inner.items():
        if not isinstance(info, dict): continue
        rescued = set(info.get("rescued") or [])
        entries.append((label, len(rescued), rescued))
    n_fail = 331

    # Order by individual rescue count (descending)
    entries_sorted = sorted(entries, key=lambda e: -e[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    # Panel (a): per-ckpt rescue
    labels = [e[0].replace("full_", "Full ").replace("lora_", "LoRA ").replace("_", " ")
              for e in entries_sorted]
    counts = [e[1] for e in entries_sorted]
    pcts = [100 * c / n_fail for c in counts]
    cs = ["#E55934" if "Full" in l else "#3498DB" for l in labels]

    xs1 = np.arange(len(labels))
    ax1.bar(xs1, pcts, color=cs, edgecolor="black", linewidth=1.0, alpha=0.92)
    ax1.set_xticks(xs1)
    ax1.set_xticklabels(labels, rotation=70, ha="right", fontsize=10)
    ax1.set_ylabel("greedy fail rescue (%)")
    ax1.set_ylim(0, 35)
    ax1.grid(axis="y", alpha=0.25, linestyle=":")
    # Legend
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(facecolor="#E55934", edgecolor="black", label="Full-SFT"),
        Patch(facecolor="#3498DB", edgecolor="black", label="LoRA"),
    ], loc="upper right")

    # Panel (b): greedy max-cover cumulative union
    chosen, union, remaining, cumul_y = [], set(), entries.copy(), []
    while remaining:
        best_i, best_gain = None, -1
        for i, (lab, c, r) in enumerate(remaining):
            gain = len(r - union)
            if gain > best_gain: best_gain = gain; best_i = i
        chosen.append(remaining[best_i][0])
        union |= remaining[best_i][2]
        cumul_y.append(len(union))
        remaining.pop(best_i)
    cumul_pct = [100 * y / n_fail for y in cumul_y]
    plateau = cumul_pct[-1]

    xs2 = np.arange(1, len(chosen) + 1)
    ax2.plot(xs2, cumul_pct, marker="o", color="#27AE60", linewidth=2.6,
             markersize=8, markerfacecolor="white", markeredgewidth=2)
    ax2.axhline(plateau, color="darkgreen", linestyle="--", lw=1.5, alpha=0.6)
    ax2.set_xlabel("# ckpts in ensemble (greedy max-cover)")
    ax2.set_ylabel("union rescue (%)")
    ax2.set_xticks(np.arange(0, 25, 4))
    ax2.set_ylim(0, 60)
    ax2.grid(alpha=0.25, linestyle=":")

    out = OUT / "s3_t6_landscape.png"
    plt.savefig(out); plt.close()
    print(f"  → {out}")


# ─────────────── Slide 4: pass@N vs SC@N gap on T6 best ──────────────────
def slide4_combined():
    """3 bars on Best T6 ckpt: greedy / SC@8 / pass@8 (T=0 / T=0.7 / T=1.0)."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars_data = [
        ("Greedy\nT=0",             28.1, "#95A5A6"),
        ("Majority Vote\nSC@8",     38.4, "#3498DB"),
        ("Oracle Sampling\npass@8", 65.9, "#9B59B6"),
    ]
    xs = np.arange(len(bars_data))
    ys = [b[1] for b in bars_data]
    cs = [b[2] for b in bars_data]
    bars = ax.bar(xs, ys, color=cs, edgecolor="black", linewidth=1.8,
                  alpha=0.92, width=0.55)
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width()/2, y + 1.5, f"{y:.1f}%",
                ha="center", fontsize=22, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars_data])
    ax.set_ylabel("fail rescue (%)")
    ax.set_ylim(0, 80)
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    out = OUT / "s4_pass_vs_sc.png"
    plt.savefig(out); plt.close()
    print(f"  → {out}")


# ─────────────── Slide 5: 2x2 matrix of methods × cells ──────────────────
def slide5_matrix():
    """2x2 grid: rows = inference / training; cols = position-only / context-aware.
    Each cell shows the methods we tried + their best rescue %.
    Bottom-right cell (training × context) is the frontier."""
    fig, ax = plt.subplots(figsize=(14, 8))

    cell_data = [
        # row 0 = Inference, col 0 = Position
        {"row": 0, "col": 0,
         "title": "Inference × Position",
         "methods": ["A1 DAG edge      0.0%",
                     "A2 token revise  0.0%",
                     "A3 span revise   0.0%",
                     "A4 block layout  8.3%"],
         "best": "8.3%",
         "color": "#FAD7A0"},
        # row 0, col 1 = Inference × Context
        {"row": 0, "col": 1,
         "title": "Inference × Context",
         "methods": ["A5 prompt template 13.3%",
                     "A6 gen length     20.0%",
                     "H3 pass@N (oracle) 86.7%",
                     "SC@N (deployable) 38.4%"],
         "best": "86.7% (oracle)",
         "color": "#D5A6BD"},
        # row 1, col 0 = Training × Position (token-level)
        {"row": 1, "col": 0,
         "title": "Training × Token-level",
         "methods": ["T6 Full-SFT (greedy)  28.1%",
                     "T6 LoRA r=1           10.6%",
                     "24-ckpt union (oracle) 49.8%",
                     ""],
         "best": "49.8%",
         "color": "#A9CCE3"},
        # row 1, col 1 = Training × Trajectory  (FRONTIER)
        {"row": 1, "col": 1,
         "title": "Training × Trajectory",
         "methods": ["T7 self-distill  ?",
                     "ORM + BoN        ?",
                     "PRM-RL           ?",
                     ""],
         "best": "TBD",
         "color": "#FFFFFF",
         "frontier": True},
    ]

    cell_w, cell_h = 5.5, 3.0
    base_x, base_y = 1.5, 0.5

    from matplotlib.patches import FancyBboxPatch
    for c in cell_data:
        x0 = base_x + c["col"] * (cell_w + 0.2)
        y0 = base_y + (1 - c["row"]) * (cell_h + 0.2)
        edge = "darkred" if c.get("frontier") else "black"
        edge_w = 3 if c.get("frontier") else 1.5
        box = FancyBboxPatch((x0, y0), cell_w, cell_h,
                             boxstyle="round,pad=0.05",
                             facecolor=c["color"], edgecolor=edge,
                             linewidth=edge_w, alpha=0.85)
        ax.add_patch(box)
        # Title
        ax.text(x0 + cell_w / 2, y0 + cell_h - 0.35, c["title"],
                ha="center", fontsize=17, fontweight="bold")
        # Methods
        for i, m in enumerate(c["methods"]):
            if not m: continue
            ax.text(x0 + 0.25, y0 + cell_h - 0.85 - 0.4 * i, m,
                    ha="left", fontsize=13, family="monospace")
        # Best at bottom of cell
        ax.text(x0 + cell_w / 2, y0 + 0.25, f"best: {c['best']}",
                ha="center", fontsize=15, fontweight="bold",
                color="darkred" if c.get("frontier") else "black",
                style="italic")

    # Axis labels (matrix headers)
    # Column headers (top)
    ax.text(base_x + cell_w / 2, base_y + 2 * cell_h + 0.5,
            "Position-only", ha="center",
            fontsize=18, fontweight="bold")
    ax.text(base_x + cell_w + 0.2 + cell_w / 2,
            base_y + 2 * cell_h + 0.5,
            "Context-aware", ha="center",
            fontsize=18, fontweight="bold")
    # Row labels (left)
    ax.text(base_x - 0.5, base_y + cell_h * 1.5 + 0.1,
            "Inference\n(no training)",
            ha="right", va="center", fontsize=18, fontweight="bold")
    ax.text(base_x - 0.5, base_y + cell_h * 0.5,
            "Training\n(modify model)",
            ha="right", va="center", fontsize=18, fontweight="bold")

    ax.set_xlim(0, base_x + 2 * cell_w + 1)
    ax.set_ylim(0, base_y + 2 * cell_h + 1.2)
    ax.set_aspect("equal")
    ax.axis("off")

    out = OUT / "s5_method_matrix.png"
    plt.savefig(out); plt.close()
    print(f"  → {out}")


# ────────────────────────────── Main ───────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating slide data figures → {OUT}")
    slide2_a_axis()
    slide3_t6_landscape()
    slide4_combined()
    slide5_matrix()
    print("Done.")
