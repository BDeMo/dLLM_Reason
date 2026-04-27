#!/usr/bin/env python
"""Generate figures for docs/archive/logic_chain_a_axis_to_p2.zh.md.

Reads existing run summaries and ablation results, produces PNG figures
under docs/figures/. Idempotent — re-run after new data lands.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
})


# ─────────────────────────── Figure 1: T6 Pareto ────────────────────────────
def fig_t6_pareto():
    """Full-SFT vs LoRA ckpts — fail rescue vs ok retention."""
    full_path = ROOT / "runs/validation/t6_ablate/summary.md"
    lora_path = ROOT / "runs/validation/t6_lora_ablate/summary.md"
    rows = []  # (mode, label, fail_pct, ok_pct, size)

    pat_full = re.compile(r"\| (\d+) \| ([\d.]+) \| \d+/\d+ \(([\d.]+)%\) \| \d+/\d+ \(([\d.]+)%\)")
    if full_path.exists():
        for line in full_path.read_text(encoding="utf-8").splitlines():
            m = pat_full.match(line)
            if m:
                rows.append(("full", f"ep={m.group(2)}",
                             float(m.group(3)), float(m.group(4)),
                             float(m.group(2))))
    pat_lora = re.compile(r"\| (\d+) \| ([\d.]+) \| \d+ \| \d+/\d+ \(([\d.]+)%\) \| \d+/\d+ \(([\d.]+)%\)")
    if lora_path.exists():
        for line in lora_path.read_text(encoding="utf-8").splitlines():
            m = pat_lora.match(line)
            if m:
                rank = int(m.group(1)); ep = float(m.group(2))
                rows.append((f"lora_r{rank}", f"r{rank} ep{ep}",
                             float(m.group(3)), float(m.group(4)), ep))

    fig, ax = plt.subplots(figsize=(8, 6))
    # Full-SFT in red square
    full = [(r[2], r[3], r[4], r[1]) for r in rows if r[0] == "full"]
    if full:
        xs = [r[1] for r in full]; ys = [r[0] for r in full]
        ax.scatter(xs, ys, s=[80 + 30*r[2] for r in full],
                   marker="s", color="C3", edgecolors="black",
                   label="Full-SFT", zorder=5)
        for r in full:
            ax.annotate(r[3], (r[1], r[0]), fontsize=8,
                        xytext=(5, 5), textcoords="offset points")
    # LoRA per rank in different colors
    rank_colors = {1: "C0", 2: "C9", 4: "C2", 8: "C1", 16: "C4"}
    for rank in [1, 2, 4, 8, 16]:
        lora_r = [(r[2], r[3], r[4], r[1])
                  for r in rows if r[0] == f"lora_r{rank}"]
        if not lora_r:
            continue
        xs = [r[1] for r in lora_r]; ys = [r[0] for r in lora_r]
        ax.scatter(xs, ys, s=[40 + 20*r[2] for r in lora_r],
                   marker="o", color=rank_colors[rank], alpha=0.75,
                   label=f"LoRA r={rank}", zorder=4)

    # Reference points
    ax.scatter([100], [0], marker="^", s=120, color="gray",
               label="baseline (vanilla LLaDA)")
    # Target zone
    ax.axvspan(95, 100, alpha=0.08, color="green",
               label="ok retention ≥ 95%")

    ax.set_xlabel("ok retention (%)")
    ax.set_ylabel("fail rescue (%)")
    ax.set_title("Trade-off landscape — 24 trained checkpoints\n(point size ∝ training epochs)")
    ax.set_xlim(85, 101); ax.set_ylim(-2, 35)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    out = FIG_DIR / "t6_pareto.png"
    plt.savefig(out); plt.close()
    print(f"  → {out}")


# ──────────────────────── Figure 2: A-axis bar chart ────────────────────────
def fig_a_axis_bars():
    """Rescue rate of each A-axis method on the 60-prompt scope."""
    methods = [
        ("A1\nDAG search\n(edge)", 0.0, "red"),
        ("A2/H1\ntoken revise", 0.0, "red"),
        ("A3\nspan revise", 0.0, "red"),
        ("A4\nblock layout", 8.33, "tab:orange"),
        ("A5\nprompt template", 13.33, "tab:orange"),
        ("A6\ngen length", 20.00, "tab:orange"),
        ("H3\npass@N (T>0)", 86.67, "green"),
        ("A4×A5\njoint", 16.67, "tab:olive"),
        ("All-A\nunion", 91.67, "darkgreen"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    xs = np.arange(len(methods))
    ys = [m[1] for m in methods]
    cs = [m[2] for m in methods]
    bars = ax.bar(xs, ys, color=cs, edgecolor="black", alpha=0.85)
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width()/2, y + 1, f"{y:.1f}%",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([m[0] for m in methods], fontsize=9)
    ax.set_ylabel("rescue rate on 60-prompt fail scope (%)")
    ax.set_title("Inference-only rescue methods (no model training)\n"
                 "fine grain DEAD → coarse grain works → sampling oracle ceiling")
    ax.set_ylim(0, 105)
    ax.axhline(91.67, color="darkgreen", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(0, 93, "union ceiling 91.67%  (Ceiling-5 = 5 prompts unsalvageable)",
            fontsize=9, color="darkgreen")
    ax.grid(axis="y", alpha=0.3)
    out = FIG_DIR / "a_axis_methods.png"
    plt.savefig(out); plt.close()
    print(f"  → {out}")


# ─────────────────────── Figure 3: Hardset histogram ───────────────────────
def fig_hardset_histogram():
    """Distribution: how many of 24 ckpts rescue each fail prompt."""
    md = (ROOT / "runs/validation/t6_hardset/hardset.md")
    if not md.exists():
        print("  (skip — no hardset.md)"); return
    txt = md.read_text(encoding="utf-8")
    # Parse the histogram table
    pat = re.compile(r"\| (\d+) \| (\d+) \|")
    counts = {}
    in_table = False
    for line in txt.splitlines():
        if "# ckpts rescuing" in line:
            in_table = True; continue
        if in_table:
            m = pat.match(line.strip())
            if m: counts[int(m.group(1))] = int(m.group(2))
    # Ensure all bins 0-23 exist
    full = {k: counts.get(k, 0) for k in range(24)}
    fig, ax = plt.subplots(figsize=(11, 4.5))
    xs = list(full.keys()); ys = list(full.values())
    bar_colors = ["red" if k == 0 else
                  "tab:orange" if k <= 2 else
                  "tab:olive" if k <= 6 else "tab:green" for k in xs]
    ax.bar(xs, ys, color=bar_colors, edgecolor="black", alpha=0.85)
    # Annotate hardset count
    ax.text(0, full[0] + 4, f"hardset\n{full[0]} prompts", ha="center",
            fontsize=10, color="red", fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xlabel("# of trained checkpoints that rescue this prompt")
    ax.set_ylabel("# of failing prompts")
    ax.set_title("Per-prompt rescue coverage across 24 checkpoints\n"
                 f"hardset = 166/331 prompts no checkpoint rescues (greedy ceiling = 49.8%)")
    ax.grid(axis="y", alpha=0.3)
    out = FIG_DIR / "t6_hardset_histogram.png"
    plt.savefig(out); plt.close()
    print(f"  → {out}")


# ───────────────────── Figure 4: pass@N vs SC@N gap ────────────────────────
def fig_passN_vs_sc():
    """For each (ckpt, T) cell: pass@8 vs SC@8 bars side by side."""
    base = ROOT / "runs/validation/t6_decode_ablate"
    cells = []  # (ckpt_label, T, pass8, sc8)
    if not base.is_dir():
        print("  (skip — no t6_decode_ablate)"); return

    # Compact labels to avoid x-axis overlap. Full chart already conveys
    # the model identity via grouping; T= ... is the within-group axis.
    label_short = {
        "v161_t6_ablate_hf_step_336": "Full ep=2",
        "v161_t6_ablate_hf_step_84":  "Full ep=0.5",
        "v161_t6_lora_r1_hf_step_336_merged": "LoRA r=1 ep=2",
    }
    for d in sorted(base.iterdir()):
        if not d.is_dir(): continue
        ckpt = label_short.get(d.name, d.name)
        # Read pass@8 from summary.md
        sm = d / "summary.md"
        sc = d / "sc_summary.md"
        if not sm.exists() or not sc.exists(): continue
        # Parse SC summary which has BOTH pass@8 and SC@8 for fail rows.
        # Format:
        #   | run | temp | group | n | pass@1 | pass@4 | pass@8 | SC@1 | SC@4 | SC@8 | has_ans |
        for line in sc.read_text(encoding="utf-8").splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 12:
                continue
            # parts[1]=run, [2]=temp, [3]=group, [4]=n, [5..7]=pass@{1,4,8}, [8..10]=SC@{1,4,8}
            try:
                if parts[3] != "fail":
                    continue
                T = float(parts[2])
                pass8 = float(parts[7].rstrip("%"))
                sc8 = float(parts[10].rstrip("%"))
                cells.append((ckpt, T, pass8, sc8))
            except (ValueError, IndexError):
                continue

    cells = [c for c in cells if c[3] is not None]
    if not cells:
        print("  (skip — no parsed cells)"); return

    fig, ax = plt.subplots(figsize=(13, 6))
    n = len(cells)
    xs = np.arange(n)
    pass8 = [c[2] for c in cells]
    sc8 = [c[3] for c in cells]
    w = 0.38
    pass_bars = ax.bar(xs - w/2, pass8, w, color="steelblue",
                        edgecolor="black", label="Sampling oracle (pass@8)")
    sc_bars = ax.bar(xs + w/2, sc8, w, color="indianred",
                      edgecolor="black", label="Majority vote (SC@8)")

    # Number on TOP of each bar
    for b, v in zip(pass_bars, pass8):
        ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.1f}",
                ha="center", fontsize=8, color="steelblue")
    for b, v in zip(sc_bars, sc8):
        ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.1f}",
                ha="center", fontsize=8, color="indianred")

    # Gap label CENTERED ABOVE the bar pair (above pass@8 since it's higher)
    for i, (p, s) in enumerate(zip(pass8, sc8)):
        ax.text(i, p + 4.5, f"gap −{p - s:.1f}",
                ha="center", fontsize=9, color="dimgray", fontweight="bold")

    ax.set_xticks(xs)
    # Two-line label: model on top, T on bottom; rotated 25° to avoid overlap
    ax.set_xticklabels([f"{c[0]}\nT={c[1]}" for c in cells],
                       fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("fail rescue (%)")
    ax.set_title("Sampling oracle vs deployable majority-vote on full scope\n"
                 "gap 25-30% = model capacity exists but cannot be tapped without oracle")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 80)
    plt.subplots_adjust(bottom=0.22)
    out = FIG_DIR / "passN_vs_SC.png"
    plt.savefig(out); plt.close()
    print(f"  → {out}")


# ────────────────────── Figure 5: Capacity ladder ──────────────────────────
def fig_capacity_ladder():
    """Vertical bars showing each level of fail rescue achievable."""
    levels = [
        ("Untrained\ngreedy", 0.0, "lightgray"),
        ("Trained\ngreedy", 28.1, "tab:blue"),
        ("Sampling\n+ majority vote", 38.4, "tab:cyan"),
        ("Sampling\n+ oracle", 65.9, "tab:purple"),
        ("Best-of\ncheckpoints", 49.8, "tab:olive"),
        ("Inference\ntweaks", 91.67, "tab:green"),
        ("Combined\npotential", 100, "white"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    xs = np.arange(len(levels))
    ys = [l[1] for l in levels]
    cs = [l[2] for l in levels]
    bars = ax.bar(xs, ys, color=cs, edgecolor="black", alpha=0.85)
    for i, (b, y) in enumerate(zip(bars, ys)):
        if i < 6:
            ax.text(b.get_x() + b.get_width()/2, y + 2, f"{y:.1f}%",
                    ha="center", fontsize=10, fontweight="bold")
        else:
            ax.text(b.get_x() + b.get_width()/2, 50, "?", ha="center",
                    fontsize=20, fontweight="bold", color="gray")
    ax.set_xticks(xs)
    ax.set_xticklabels([l[0] for l in levels], fontsize=9)
    ax.set_ylabel("fail rescue (% of scope_fail)")
    ax.set_title("Capacity ladder — what each axis can deliver on fail rescue\n"
                 "(60-scope vs 331-scope mixed, see notes)")
    ax.set_ylim(0, 105)
    ax.axhline(50, color="gray", linestyle=":", alpha=0.5)
    ax.text(0, 51, "T6 hardset boundary (50%)", fontsize=8, color="gray")
    ax.grid(axis="y", alpha=0.3)
    # Note
    ax.text(0.02, 0.95,
            "Note: A-union 91.67% on old 60-scope; others on new 331-scope.\n"
            "Crucial fact: A∩T6 ceilings disjoint → combined potential > either alone.",
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))
    out = FIG_DIR / "capacity_ladder.png"
    plt.savefig(out); plt.close()
    print(f"  → {out}")


# ──────────────────── Figure 6: Cross-axis Venn diagram ────────────────────
def fig_cross_axis_venn():
    """Two disjoint sets — circles do NOT overlap; arrow into the gap
    between them with 'empty' label, no marker icon."""
    fig, ax = plt.subplots(figsize=(10, 6))
    from matplotlib.patches import Circle

    # Touching (tangent) circles: right edge of left = left edge of right.
    # Single touch point at x=0 — geometrically still ∩=∅ (a point has
    # measure zero) and visually "they touch but don't overlap".
    LEFT_CTR = (-1.5, 0); LEFT_R = 1.5
    RIGHT_CTR = (2.2, 0); RIGHT_R = 2.2
    c1 = Circle(LEFT_CTR, LEFT_R, alpha=0.35, color="tab:blue", linewidth=2,
                edgecolor="navy")
    c2 = Circle(RIGHT_CTR, RIGHT_R, alpha=0.35, color="tab:red", linewidth=2,
                edgecolor="darkred")
    ax.add_patch(c1); ax.add_patch(c2)

    # Left circle: inference-only ceiling
    ax.text(LEFT_CTR[0], LEFT_CTR[1] + 1.85, "Inference cannot rescue",
            ha="center", fontsize=12, fontweight="bold", color="navy")
    ax.text(LEFT_CTR[0], LEFT_CTR[1] + 0.35, "{4, 5, 14, 41, 42}",
            ha="center", fontsize=11)
    ax.text(LEFT_CTR[0], LEFT_CTR[1] - 0.45, "5 prompts",
            ha="center", fontsize=10, color="navy")

    # Right circle: training-only ceiling
    ax.text(RIGHT_CTR[0], RIGHT_CTR[1] + 2.5, "Training cannot rescue",
            ha="center", fontsize=12, fontweight="bold", color="darkred")
    ax.text(RIGHT_CTR[0], RIGHT_CTR[1] + 0.3, "166 prompts",
            ha="center", fontsize=11)
    ax.text(RIGHT_CTR[0], RIGHT_CTR[1] - 0.5, "(across 24 trained ckpts)",
            ha="center", fontsize=9, color="darkred")

    # Arrow points from below to the tangent point (x=0, y=0)
    TOUCH_X = 0
    ax.annotate("intersection is empty",
                xy=(TOUCH_X, 0),
                xytext=(TOUCH_X, -2.6),
                fontsize=14, color="green", fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color="green", lw=2))

    ax.text(0, 3.7,
            "Two ceilings are disjoint → combining axes can break both",
            ha="center", fontsize=12, fontweight="bold")

    ax.set_xlim(-4, 5); ax.set_ylim(-3.2, 4.4)
    ax.set_aspect("equal")
    ax.axis("off")
    out = FIG_DIR / "cross_axis_venn.png"
    plt.savefig(out); plt.close()
    print(f"  → {out}")


# ──────────────────────────── Main ────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures →", FIG_DIR)
    fig_t6_pareto()
    fig_a_axis_bars()
    fig_hardset_histogram()
    fig_passN_vs_sc()
    fig_capacity_ladder()
    fig_cross_axis_venn()
    print("Done.")
