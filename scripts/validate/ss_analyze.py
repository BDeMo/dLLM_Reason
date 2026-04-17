"""ss_analyze.py — Analyze strategy_search run output, paper-ready.

Reads a finished `runs/validation/strategy_search_<ts>/` directory and emits:
  - <run_dir>/analysis_report.md    human-readable findings
  - <run_dir>/analysis_stats.json   machine-readable stats for downstream

Answers (in order):
  1. Headline: oracle rate, distillable count, ceiling sanity check.
  2. vs A-union / H3 / full-method-union: does SS break 91.67%?
  3. template_position novelty: unique rescues from suffix_scaffold / mid_anchor.
  4. Winner-kind distribution + strategy entropy (distill difficulty proxy).
  5. FAIL18 per-prompt breakdown (which SS-config rescued each).
  6. Per-dim marginals (universal sweet spots vs per-prompt diversity).
  7. Difficulty: histogram of n_correct_configs per prompt.

Usage:
  python scripts/validate/ss_analyze.py --run_dir runs/validation/strategy_search_<ts>
  python scripts/validate/ss_analyze.py --run_dir <path> --print    # also print to stdout

Notes:
  - Baseline invariants (FAIL18, ceiling 5, A-axis rescues) are hardcoded below
    from P6 crossref on n=60 (last verified 2026-04-16 via
    docs/archive/ablation_index.zh.md). Re-verify if baseline runs change.
  - FAIL group indices are 0..59 (matches scope_fail_prompts.json[:60]).
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


# ── Hardcoded baseline invariants (n=60, P6-authoritative) ────────────────────
# Source: docs/archive/ablation_index.zh.md + p6_h3_crossref output.
FAIL18 = {0, 4, 5, 8, 10, 13, 14, 15, 19, 28, 35, 41, 42, 48, 51, 53, 55, 59}
CEILING_5 = {4, 5, 14, 41, 42}  # true capacity ceiling — unsalvageable by any method pre-SS
A4_RESCUE = {0, 8, 13, 15, 28}                           # 5 / 60
# A5 rescue set — recompute from a5 per_prompt if you need exactness; this is
# the canonical set referenced in finding_a_axis_exploration.
A5_RESCUE = {0, 8, 10, 13, 15, 28, 53, 55}               # 8 / 60 (approx; verify)
A6_RESCUE = {0, 10, 13, 15, 19, 28, 35, 48, 51, 53, 55, 59}  # 12 / 60
A_UNION = A4_RESCUE | A5_RESCUE | A6_RESCUE              # 13 / 60
# H3 rescue at n=60 (per P6 h3_only=42 + covered in FAIL18). Full set has 52
# prompts; we only list the 10 that intersect FAIL18 since the other 42 are
# outside FAIL18 (i.e. baseline-correct prompts). For "new-lever" comparisons
# we care about the FAIL18 slice.
H3_FAIL18 = {0, 8, 10, 13, 15, 28, 35, 53, 55, 59}       # all already covered by A-union
FULL_METHOD_UNION = A_UNION | H3_FAIL18                   # 13 / 60 (FAIL18 view)


# ── IO ────────────────────────────────────────────────────────────────────────

def load_run(run_dir: Path):
    """Load summary + winners + every per-prompt record."""
    pp_dir = run_dir / "per_prompt"
    summary_path = run_dir / "summary.json"
    winners_path = pp_dir / "winners.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"summary.json missing at {summary_path} — did the final aggregate pass run?"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    winners = (
        json.loads(winners_path.read_text(encoding="utf-8"))
        if winners_path.exists() else []
    )
    per_prompt_recs = []
    for f in sorted(pp_dir.glob("*.json")):
        if f.name == "winners.json":
            continue
        per_prompt_recs.append(json.loads(f.read_text(encoding="utf-8")))
    return summary, winners, per_prompt_recs


# ── Analysis sections ─────────────────────────────────────────────────────────

def headline_stats(summary: dict, winners: list, recs: list) -> dict:
    """Section 1: oracle rate, distillable count, ceiling check."""
    total = len(recs)
    oracle = sum(1 for r in recs if r["n_configs_passed"] > 0)
    oracle_fail = sum(
        1 for r in recs if r.get("group") == "fail" and r["n_configs_passed"] > 0
    )
    n_fail = sum(1 for r in recs if r.get("group") == "fail")
    n_ok = sum(1 for r in recs if r.get("group") == "ok")
    oracle_ok = sum(
        1 for r in recs if r.get("group") == "ok" and r["n_configs_passed"] > 0
    )
    ss_fail_rescued = {
        r["idx"] for r in recs if r.get("group") == "fail" and r["n_configs_passed"] > 0
    }
    ceiling_broken = CEILING_5 & ss_fail_rescued
    ceiling_held = CEILING_5 - ss_fail_rescued
    return dict(
        total=total, n_fail=n_fail, n_ok=n_ok,
        oracle_all=oracle, oracle_fail=oracle_fail, oracle_ok=oracle_ok,
        oracle_rate_all=oracle / max(total, 1),
        oracle_rate_fail=oracle_fail / max(n_fail, 1),
        oracle_rate_ok=oracle_ok / max(n_ok, 1),
        ss_fail_rescued=sorted(ss_fail_rescued),
        ceiling_broken_by_ss=sorted(ceiling_broken),
        ceiling_held_under_ss=sorted(ceiling_held),
    )


def vs_prior_baselines(ss_fail_rescued: set) -> dict:
    """Section 2: compare SS fail-rescue set with A-union / H3_FAIL18 / full union."""
    ss_vs_a_union_new = ss_fail_rescued - A_UNION
    ss_vs_fullunion_new = ss_fail_rescued - FULL_METHOD_UNION
    missing_from_ss = A_UNION - ss_fail_rescued  # regression check
    return dict(
        a_union=sorted(A_UNION), a_union_size=len(A_UNION),
        full_method_union=sorted(FULL_METHOD_UNION),
        full_method_union_size=len(FULL_METHOD_UNION),
        ss_rescued_size=len(ss_fail_rescued),
        ss_new_over_a_union=sorted(ss_vs_a_union_new),
        ss_new_over_a_union_size=len(ss_vs_a_union_new),
        ss_new_over_full_union=sorted(ss_vs_fullunion_new),
        ss_new_over_full_union_size=len(ss_vs_fullunion_new),
        regressed=sorted(missing_from_ss),
        regressed_size=len(missing_from_ss),
    )


def template_position_novelty(recs: list) -> dict:
    """Section 3: partition rescues by template_position; count position-only rescues.

    Diffusion-LM-unique paper claim: suffix_scaffold / mid_anchor (= inpainting-
    style scaffolding) rescue prompts that prefix cannot.
    """
    # For each fail prompt, collect the set of template_positions among its
    # correct configs.
    position_rescue_sets: dict[str, set[int]] = defaultdict(set)
    for r in recs:
        if r.get("group") != "fail":
            continue
        idx = r["idx"]
        if r["n_configs_passed"] == 0:
            continue
        for res in r["results"]:
            if res.get("pass@1", 0) >= 1.0:
                pos = res["config"]["template_position"]
                position_rescue_sets[pos].add(idx)

    positions = ["prefix", "suffix_scaffold", "mid_anchor", "none"]
    unique_per_position: dict[str, list[int]] = {}
    for p in positions:
        others = set()
        for q in positions:
            if q != p:
                others |= position_rescue_sets.get(q, set())
        unique_per_position[p] = sorted(position_rescue_sets.get(p, set()) - others)

    return dict(
        rescue_sets_by_position={
            p: sorted(position_rescue_sets.get(p, set())) for p in positions
        },
        sizes_by_position={
            p: len(position_rescue_sets.get(p, set())) for p in positions
        },
        unique_per_position=unique_per_position,
        unique_size_per_position={p: len(v) for p, v in unique_per_position.items()},
        # Diffusion-LM-novel: any rescue from {suffix_scaffold, mid_anchor}
        # that prefix cannot achieve.
        prefix_rescue_set=sorted(position_rescue_sets.get("prefix", set())),
        inpaint_novel_set=sorted(
            (position_rescue_sets.get("suffix_scaffold", set())
             | position_rescue_sets.get("mid_anchor", set()))
            - position_rescue_sets.get("prefix", set())
        ),
    )


def winner_kind_stats(winners: list) -> dict:
    """Section 4: winner-kind distribution + strategy entropy (distill proxy)."""
    kinds = ("cheapest", "shortest", "most_reliable", "deterministic")
    stats: dict = {}
    for k in kinds:
        config_ids = [
            w["winners"][k]["config_id"] for w in winners
            if k in w.get("winners", {})
        ]
        if not config_ids:
            stats[k] = dict(n=0, entropy_bits=0.0, top1_frac=0.0, distinct=0)
            continue
        cnt = Counter(config_ids)
        total = sum(cnt.values())
        entropy = -sum((v / total) * _log2(v / total) for v in cnt.values())
        top1_frac = cnt.most_common(1)[0][1] / total
        stats[k] = dict(
            n=total, entropy_bits=entropy, top1_frac=top1_frac,
            distinct=len(cnt),
            top5=[(c, n, n / total) for c, n in cnt.most_common(5)],
        )
    return stats


def _log2(x: float) -> float:
    import math
    return math.log2(x) if x > 0 else 0.0


def fail18_breakdown(recs: list, winners: list) -> list:
    """Section 5: per-prompt breakdown for each of FAIL18."""
    win_by_idx = {
        w["idx"]: w for w in winners
        if w.get("group") == "fail"
    }
    rec_by_idx = {
        r["idx"]: r for r in recs
        if r.get("group") == "fail"
    }
    rows = []
    for idx in sorted(FAIL18):
        rec = rec_by_idx.get(idx)
        win = win_by_idx.get(idx, {})
        passed = rec["n_configs_passed"] if rec else 0
        rescued_prior = {
            "A4": idx in A4_RESCUE, "A5": idx in A5_RESCUE,
            "A6": idx in A6_RESCUE, "H3": idx in H3_FAIL18,
        }
        cheapest = win.get("winners", {}).get("cheapest")
        rows.append(dict(
            idx=idx,
            in_ceiling5=idx in CEILING_5,
            rescued_prior=rescued_prior,
            rescued_by_ss=passed > 0,
            n_correct_configs=passed,
            cheapest_config_id=(cheapest["config_id"] if cheapest else None),
            cheapest_config=(cheapest["config"] if cheapest else None),
        ))
    return rows


def dim_marginals(winners: list) -> dict:
    """Section 6: per-dim histogram among cheapest winners."""
    dims = ("block_length", "template_name", "template_position",
            "gen_length", "temperature")
    marg: dict[str, Counter] = {d: Counter() for d in dims}
    for w in winners:
        c = w.get("winners", {}).get("cheapest")
        if not c:
            continue
        for d in dims:
            marg[d][c["config"][d]] += 1
    return {
        d: sorted(
            [(v, cnt, cnt / sum(ct.values())) for v, cnt in ct.most_common()],
            key=lambda x: -x[1]
        )
        for d, ct in marg.items()
    }


def difficulty_hist(recs: list) -> dict:
    """Section 7: histogram of n_correct_configs for fail prompts."""
    fail_passed = [r["n_configs_passed"] for r in recs if r.get("group") == "fail"]
    if not fail_passed:
        return dict(n=0)
    return dict(
        n=len(fail_passed),
        min=min(fail_passed), max=max(fail_passed),
        median=statistics.median(fail_passed),
        mean=statistics.mean(fail_passed),
        zero=sum(1 for x in fail_passed if x == 0),
        one_to_five=sum(1 for x in fail_passed if 1 <= x <= 5),
        six_to_twenty=sum(1 for x in fail_passed if 6 <= x <= 20),
        twenty_one_plus=sum(1 for x in fail_passed if x > 20),
    )


# ── Report rendering ──────────────────────────────────────────────────────────

def render_report(run_dir: Path, h: dict, vs: dict, tpos: dict,
                  wk: dict, fail18: list, marg: dict, diff: dict,
                  summary: dict) -> str:
    lines: list[str] = []
    lines.append(f"# Strategy Search — Analysis Report\n")
    lines.append(f"**run_dir**: `{run_dir}`\n")
    lines.append(f"**configs/prompt**: {summary.get('n_configs_per_prompt')}\n")

    # Section 1
    lines.append("## 1. Headline\n")
    lines.append(f"- Oracle correct (all): **{h['oracle_all']}/{h['total']} "
                 f"= {h['oracle_rate_all']:.2%}**")
    lines.append(f"- Oracle correct (fail group only): **{h['oracle_fail']}/{h['n_fail']} "
                 f"= {h['oracle_rate_fail']:.2%}**")
    lines.append(f"- Oracle correct (ok group, sanity): {h['oracle_ok']}/{h['n_ok']} "
                 f"= {h['oracle_rate_ok']:.2%}")
    lines.append(f"- SS rescued fail set ({len(h['ss_fail_rescued'])} prompts): "
                 f"`{h['ss_fail_rescued']}`")
    lines.append(f"- **Capacity-ceiling check** (prior ceiling = `{sorted(CEILING_5)}`):")
    if h["ceiling_broken_by_ss"]:
        lines.append(f"    - ⚠ SS **broke** {len(h['ceiling_broken_by_ss'])} "
                     f"previously-ceiling prompts: `{h['ceiling_broken_by_ss']}` "
                     f"→ re-investigate ceiling hypothesis")
    if h["ceiling_held_under_ss"]:
        lines.append(f"    - Confirmed still-stuck: `{h['ceiling_held_under_ss']}`")
    lines.append("")

    # Section 2
    lines.append("## 2. vs Prior Baselines\n")
    lines.append(f"- A-union (A4∪A5∪A6) on n=60 fail: **{vs['a_union_size']}/60** "
                 f"`{vs['a_union']}`")
    lines.append(f"- Full-method union (A-union ∪ H3 on FAIL18): **{vs['full_method_union_size']}/18** "
                 f"`{vs['full_method_union']}`")
    lines.append(f"- SS rescue size: **{vs['ss_rescued_size']}/60**")
    lines.append(f"- SS-new over A-union: **+{vs['ss_new_over_a_union_size']}** "
                 f"prompts `{vs['ss_new_over_a_union']}`")
    lines.append(f"- SS-new over full-method-union: **+{vs['ss_new_over_full_union_size']}** "
                 f"prompts `{vs['ss_new_over_full_union']}`")
    if vs["regressed_size"] > 0:
        lines.append(f"- ⚠ **Regression** — A-union prompts NOT rescued by SS: "
                     f"`{vs['regressed']}` ({vs['regressed_size']}) → unexpected; check search space coverage")
    else:
        lines.append(f"- ✓ No regression: SS is a strict superset of A-union on fail group")
    lines.append("")

    # Section 3
    lines.append("## 3. template_position Novelty (paper key claim)\n")
    lines.append("Rescue set partitioned by `template_position`:")
    for p in ("prefix", "suffix_scaffold", "mid_anchor", "none"):
        lines.append(f"- `{p}`: **{tpos['sizes_by_position'][p]}** prompts, "
                     f"unique-to-this-position: **{tpos['unique_size_per_position'][p]}** "
                     f"`{tpos['unique_per_position'][p]}`")
    inpaint_novel = tpos["inpaint_novel_set"]
    lines.append(f"\n**Inpainting-novel** = (`suffix_scaffold` ∪ `mid_anchor`) \\ `prefix`: "
                 f"**{len(inpaint_novel)}** prompts `{inpaint_novel}`")
    if inpaint_novel:
        lines.append("→ Supports the diffusion-LM-unique claim: canvas-position-aware "
                     "scaffolding rescues prompts that prefix-only conditioning cannot.")
    else:
        lines.append("→ ⚠ No inpaint-unique rescue. The template_position dimension "
                     "provides diversity but no *net-new* prompt-level rescue over prefix.")
    lines.append("")

    # Section 4
    lines.append("## 4. Winner-kind Distribution (distill difficulty proxy)\n")
    lines.append("| Kind | N | Distinct configs | Entropy (bits) | Top-1 share |")
    lines.append("|---|---|---|---|---|")
    for k in ("cheapest", "shortest", "most_reliable", "deterministic"):
        s = wk.get(k, {})
        lines.append(f"| {k} | {s.get('n', 0)} | {s.get('distinct', 0)} | "
                     f"{s.get('entropy_bits', 0):.2f} | {s.get('top1_frac', 0):.2%} |")
    lines.append("\n**Top-5 cheapest configs** (if distillation targets a universal policy):")
    for c, n, frac in wk.get("cheapest", {}).get("top5", []):
        lines.append(f"- `{c}` — {n} prompts ({frac:.2%})")
    lines.append("")

    # Section 5
    lines.append("## 5. FAIL18 Per-prompt Breakdown\n")
    lines.append("| idx | ceiling5 | A4 | A5 | A6 | H3 | SS rescued? | n_correct_cfg | cheapest winner |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in fail18:
        prior = row["rescued_prior"]
        cw = row["cheapest_config"]
        cheap_desc = (
            f"bl{cw['block_length']}/"
            f"{cw['template_name']}/"
            f"{cw['template_position']}/"
            f"g{cw['gen_length']}/"
            f"T{cw['temperature']:g}"
            if cw else "—"
        )
        lines.append(
            f"| {row['idx']} "
            f"| {'⚠' if row['in_ceiling5'] else '—'} "
            f"| {'✓' if prior['A4'] else '·'} "
            f"| {'✓' if prior['A5'] else '·'} "
            f"| {'✓' if prior['A6'] else '·'} "
            f"| {'✓' if prior['H3'] else '·'} "
            f"| {'✅' if row['rescued_by_ss'] else '❌'} "
            f"| {row['n_correct_configs']} "
            f"| `{cheap_desc}` |"
        )
    lines.append("")

    # Section 6
    lines.append("## 6. Per-dim Marginals (among `cheapest` winners)\n")
    for d, rows in marg.items():
        lines.append(f"**{d}**:")
        for v, n, frac in rows:
            lines.append(f"  - `{v}` — {n} ({frac:.2%})")
    lines.append("")

    # Section 7
    lines.append("## 7. Difficulty distribution (fail group, n_correct_configs)\n")
    if diff.get("n", 0) > 0:
        lines.append(f"- n={diff['n']}  min={diff['min']}  max={diff['max']}  "
                     f"median={diff['median']}  mean={diff['mean']:.2f}")
        lines.append(f"- 0 correct (ceiling): {diff['zero']}")
        lines.append(f"- 1-5 correct (narrow): {diff['one_to_five']}")
        lines.append(f"- 6-20 correct: {diff['six_to_twenty']}")
        lines.append(f"- 21+ correct (easy): {diff['twenty_one_plus']}")
    lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True,
                    help="runs/validation/strategy_search_<ts>")
    ap.add_argument("--print", action="store_true",
                    help="also print report to stdout")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    summary, winners, recs = load_run(run_dir)

    h = headline_stats(summary, winners, recs)
    vs = vs_prior_baselines(set(h["ss_fail_rescued"]))
    tpos = template_position_novelty(recs)
    wk = winner_kind_stats(winners)
    fail18 = fail18_breakdown(recs, winners)
    marg = dim_marginals(winners)
    diff = difficulty_hist(recs)

    # Write machine-readable
    stats = dict(
        run_dir=str(run_dir),
        headline=h, vs_prior=vs, template_position=tpos,
        winner_kinds=wk, fail18_breakdown=fail18,
        dim_marginals=marg, difficulty=diff,
        baseline_invariants=dict(
            FAIL18=sorted(FAIL18), CEILING_5=sorted(CEILING_5),
            A_UNION=sorted(A_UNION), A4_RESCUE=sorted(A4_RESCUE),
            A5_RESCUE=sorted(A5_RESCUE), A6_RESCUE=sorted(A6_RESCUE),
            H3_FAIL18=sorted(H3_FAIL18),
        ),
    )
    stats_path = run_dir / "analysis_stats.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Write human-readable report
    report = render_report(run_dir, h, vs, tpos, wk, fail18, marg, diff, summary)
    report_path = run_dir / "analysis_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"[SSAN] stats  → {stats_path}")
    print(f"[SSAN] report → {report_path}")
    if args.print:
        print()
        print(report)


if __name__ == "__main__":
    main()
