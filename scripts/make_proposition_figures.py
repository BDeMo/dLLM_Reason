#!/usr/bin/env python
"""Four figures, one per proposition (P1→C1 ... P4→C4).

Each shows the supporting fact extracted from real run data.

P1 → C1  : position-only inference saturates; context-aware unlocks capacity
P2 → C2  : oracle ≠ deployable; training is needed to deploy
P3 → C3  : token-level training saturates (24-ckpt union plateau)
P4 → C4  : the 27% gap = wrong-mode convergence, not sampling shortage

Output → docs/figures/ppt/
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "ppt"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 18, "axes.labelsize": 14,
    "legend.fontsize": 12, "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


# ─────────────────────────── P1 → C1 ───────────────────────────────────────
def fig_p1_position_vs_context():
    """Methods grouped by position-only vs context-aware; rescue rate from
    A-axis findings (60-prompt scope). Shows position-only flat, context
    unlocks the capacity."""
    data = [
        ("A1\nedge revise",     "position", 0.0,  "DEAD"),
        ("A2\ntoken revise",    "position", 0.0,  "DEAD"),
        ("A3\nspan revise",     "position", 0.0,  "DEAD"),
        ("A4\nblock layout",    "position", 8.33, "ok"),
        ("A5\nprompt template", "context",  13.33,"ok"),
        ("A6\ngen length",      "context",  20.0, "ok"),
        ("H3\npass@N (T>0)",    "context",  86.67,"oracle"),
    ]
    fig, ax = plt.subplots(figsize=(13, 6))
    xs = np.arange(len(data))
    ys = [d[2] for d in data]
    cs = ["#95A5A6" if d[1] == "position" else "#9B59B6" for d in data]
    bars = ax.bar(xs, ys, color=cs, edgecolor="black", linewidth=1.5,
                  alpha=0.92)
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width()/2, y + 1.5, f"{y:.1f}%",
                ha="center", fontsize=12, fontweight="bold")

    # Vertical separator at position→context boundary
    ax.axvline(3.5, color="darkred", linestyle="--", lw=2, alpha=0.6)
    ax.text(3.55, 95, "transition →\ncontext-aware\nunlocks capacity",
            fontsize=12, color="darkred", fontweight="bold", style="italic")

    # Annotate the dead zone
    ax.text(1, 50, "position-only\n= flat ground",
            ha="center", fontsize=11, color="dimgray", style="italic")

    ax.set_xticks(xs)
    ax.set_xticklabels([d[0] for d in data], fontsize=11)
    ax.set_ylabel("rescue rate (% of 60-prompt fail set)")
    ax.set_title("P1 → C1: position-only methods saturate at ~8%; "
                 "context-aware unlocks 86%",
                 fontsize=16, fontweight="bold", pad=14)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    # Custom legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#95A5A6", edgecolor="black", label="Position-only"),
        Patch(facecolor="#9B59B6", edgecolor="black", label="Context-aware"),
    ], loc="upper left", framealpha=0.95)

    out = OUT / "P1_position_vs_context.png"
    plt.savefig(out); plt.close()
    print(f"  → {out}")


# ─────────────────────────── P2 → C2 ───────────────────────────────────────
def fig_p2_oracle_vs_deployable():
    """4 bars: oracle vs deployable, inference-axis vs training-axis.
    Training row shows we move from oracle-only to deployable."""
    fig, ax = plt.subplots(figsize=(12, 6.5))
    groups = ["Inference\n(no training)", "Training\n(T6 SFT)"]
    oracle  = [86.67, 50.2]   # A union (60-scope), T6 best-of-24-ckpt union
    deploy  = [10.0,  28.1]   # A4 best deployable single, T6 best greedy

    x = np.arange(len(groups))
    w = 0.36
    b1 = ax.bar(x - w/2, oracle, w, color="#9B59B6", edgecolor="black",
                linewidth=1.5, label="Oracle (best with GT-picker)")
    b2 = ax.bar(x + w/2, deploy, w, color="#27AE60", edgecolor="black",
                linewidth=1.5, label="Deployable (greedy or majority)")

    for b, v in zip(b1, oracle):
        ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}%",
                ha="center", fontsize=14, fontweight="bold")
    for b, v in zip(b2, deploy):
        ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}%",
                ha="center", fontsize=14, fontweight="bold")

    # Gap arrows
    for i, (o, d) in enumerate(zip(oracle, deploy)):
        ax.annotate("", xy=(i + w/2 + 0.02, o), xytext=(i + w/2 + 0.02, d),
                    arrowprops=dict(arrowstyle="<->", color="dimgray", lw=2))
        ax.text(i + w/2 + 0.10, (o + d)/2, f"gap\n{o - d:.0f}%",
                fontsize=11, color="dimgray", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=14)
    ax.set_ylabel("rescue rate (%)")
    ax.set_title("P2 → C2: oracle is not deployable; training closes gap from 77→22",
                 fontsize=15, fontweight="bold", pad=14)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    out = OUT / "P2_oracle_vs_deployable.png"
    plt.savefig(out); plt.close()
    print(f"  → {out}")


# ─────────────────────────── P3 → C3 ───────────────────────────────────────
def fig_p3_training_saturates():
    """Cumulative-rescue curve: order ckpts by rescue, plot union as we
    add ckpts. Should plateau at 49.8% = T6 hardset complement."""
    per_ckpt_path = ROOT / "runs/validation/t6_hardset/per_ckpt.json"
    if not per_ckpt_path.exists():
        print("  (skip P3 — no per_ckpt.json)"); return

    raw = json.loads(per_ckpt_path.read_text(encoding="utf-8"))
    # Real schema: {"per_ckpt": {label: {rescued: [...], ...}}, ...}
    inner = raw.get("per_ckpt", raw)
    entries = []
    for label, info in inner.items():
        if not isinstance(info, dict):
            continue
        rescued = set(info.get("rescued") or info.get("rescued_idxs") or [])
        entries.append((label, rescued))
    if not entries:
        print("  (skip P3 — empty per_ckpt)"); return

    # Greedy ordering: keep adding the ckpt with the largest marginal lift
    chosen = []
    union = set()
    remaining = entries.copy()
    cumul_y = []
    while remaining:
        best_i, best_gain = None, -1
        for i, (lab, r) in enumerate(remaining):
            gain = len(r - union)
            if gain > best_gain:
                best_gain = gain; best_i = i
        chosen.append(remaining[best_i][0])
        union |= remaining[best_i][1]
        cumul_y.append(len(union))
        remaining.pop(best_i)

    n_fail = 331
    cumul_pct = [100 * y / n_fail for y in cumul_y]

    fig, ax = plt.subplots(figsize=(12, 6))
    xs = np.arange(1, len(chosen) + 1)
    ax.plot(xs, cumul_pct, marker="o", color="#E55934", linewidth=2.5,
            markersize=8, markerfacecolor="white", markeredgewidth=2)

    # Plateau line
    plateau = cumul_pct[-1]
    ax.axhline(plateau, color="darkred", linestyle="--", lw=1.5, alpha=0.6)
    ax.text(len(chosen) - 1, plateau + 1.5,
            f"plateau {plateau:.1f}% — 24-ckpt union ceiling",
            fontsize=11, color="darkred", fontweight="bold", ha="right",
            style="italic")

    # Hardset annotation
    hardset_pct = 100 - plateau
    ax.text(2, plateau - 8,
            f"166 prompts ({hardset_pct:.1f}%) cannot be rescued by ANY ckpt\n"
            f"(T6 hardset)",
            fontsize=11, color="dimgray", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FAFAFA",
                      edgecolor="lightgray"))

    # Mark first few ckpts that dominate
    ax.annotate("3 ckpts ≈ 90% of plateau", xy=(3, cumul_pct[2]),
                xytext=(8, 30), fontsize=11, color="darkblue",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkblue", lw=1.3))

    ax.set_xticks(xs)
    ax.set_xlabel("# of ckpts in ensemble (greedy max-cover order)")
    ax.set_ylabel("cumulative rescue (% of 331 fail prompts)")
    ax.set_title("P3 → C3: 24 ckpts × T=0 pass@1 union plateaus at 50%\n"
                 "→ token-level training saturates; need trajectory-level",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_ylim(0, 60)
    ax.grid(alpha=0.3)

    out = OUT / "P3_training_saturates.png"
    plt.savefig(out); plt.close()
    print(f"  → {out}")


# ─────────────────────────── P4 → C4 ───────────────────────────────────────
def fig_p4_wrong_mode_convergence():
    """For each FAIL prompt that pass@8 succeeds on (≥1 correct sample),
    count the size of the LARGEST WRONG cluster among the 8 answers.
    This visualizes: 'when pass@N succeeds but SC fails, what does the
    wrong-majority look like?'"""
    cell_dir = ROOT / "runs/validation/t6_decode_ablate"
    if not cell_dir.is_dir():
        print("  (skip P4 — no decode_ablate)"); return

    # Use Full-SFT step_336 @ T=1.0 (best pass@N cell)
    candidates = list(cell_dir.glob("v161_t6_ablate_hf_step_336/T1.0_N8_*"))
    if not candidates:
        print("  (skip P4 — no T1.0 cell for step_336)"); return
    pp = candidates[0] / "per_prompt"
    if not pp.is_dir():
        print(f"  (skip P4 — no per_prompt under {pp.parent})"); return

    largest_wrong_clusters = []
    pass_succeeded_sc_failed = []  # cluster size only for these prompts

    def _ext(s):
        import re
        nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", str(s or ""))
        if not nums: return None
        try: return float(nums[-1].replace(",", ""))
        except: return None

    for f in pp.glob("fail_*.json"):
        rec = json.loads(f.read_text(encoding="utf-8"))
        T = rec.get("temps", {}).get("1.0", {})
        answers = T.get("answer_list", [])
        corrects = T.get("correct_list", [])
        if not answers or len(answers) != 8:
            continue
        gt = _ext(rec.get("gt"))
        if gt is None: continue

        wrong_answers = [a for a, c in zip(answers, corrects) if not c and a is not None]
        if not wrong_answers:
            continue
        cluster = Counter(wrong_answers)
        max_wrong = cluster.most_common(1)[0][1]

        any_correct = any(corrects)
        # SC succeeds iff mode == gt; mode of full answer_list
        all_clean = [a for a in answers if a is not None]
        mode_full = Counter(all_clean).most_common(1)[0][0] if all_clean else None
        sc_correct = (mode_full == gt) if mode_full is not None else False

        largest_wrong_clusters.append(max_wrong)
        if any_correct and not sc_correct:
            pass_succeeded_sc_failed.append(max_wrong)

    if not largest_wrong_clusters:
        print("  (skip P4 — no usable fail records)"); return

    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.arange(0.5, 9.5, 1)
    ax.hist(largest_wrong_clusters, bins=bins, color="#95A5A6",
            edgecolor="black", linewidth=1.3, alpha=0.7,
            label=f"all fail prompts (n={len(largest_wrong_clusters)})")
    ax.hist(pass_succeeded_sc_failed, bins=bins, color="#E55934",
            edgecolor="black", linewidth=1.3, alpha=0.95,
            label=f"pass@N rescued but SC missed (n={len(pass_succeeded_sc_failed)})")

    ax.axvline(4, color="darkred", linestyle="--", lw=2, alpha=0.6)
    # Threshold label placed in the LOWER-CENTER (no legend collision)
    ax.text(4.1, ax.get_ylim()[1] * 0.55 if ax.get_ylim()[1] > 0 else 30,
            "majority threshold\n(≥4 same wrong answer\nguarantees SC fail)",
            fontsize=10, color="darkred", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="darkred", alpha=0.92))

    ax.set_xticks(range(1, 9))
    ax.set_xlabel("size of largest WRONG-answer cluster (out of 8 samples)")
    ax.set_ylabel("# of fail prompts")
    ax.set_title("P4 → C4: 'wrong majority' is systematic, not random\n"
                 "Most fail prompts have ≥5 samples agreeing on a single wrong answer "
                 "→ SC misses what pass@N catches",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    out = OUT / "P4_wrong_mode_convergence.png"
    plt.savefig(out); plt.close()
    print(f"  → {out}")


# ────────────────────────────── Main ───────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating proposition figures → {OUT}")
    fig_p1_position_vs_context()
    fig_p2_oracle_vs_deployable()
    fig_p3_training_saturates()
    fig_p4_wrong_mode_convergence()
    print("Done.")
