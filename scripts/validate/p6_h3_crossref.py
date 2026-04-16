"""P6 —— H3 rescue × (A4 ∪ A5 ∪ A6) 交叉引用 (n=60 版)

对应 H3 的新 schema（`fail_XXXX.json` + `temps.T.pass@8`）修正 P5 bug，
同时把 A6 纳入 union，并计算 stuck set / true capacity ceiling。

H3 rescue 判据（与 hypotheses.md H3 阈值对齐）：
  h3_rescued(idx) = ∃ t ∈ {0.3, 0.7, 1.0}: temps[t].pass@8 == 1.0
  h3_stuck(idx)   = ∀ t ∈ {0.3, 0.7, 1.0}: temps[t].pass@8 == 0.0

A{4,5,6} rescue 判据（与各自 summary 对齐）：
  a4_rescued = per_layout.bl32 == False AND any(per_layout.values())
  a5_rescued = per_template.baseline == False AND any(per_template.values())
  a6_rescued = per_length.g128 == False AND any(per_length.values())

用法：
  python scripts/validate/p6_h3_crossref.py \\
      --h3_run runs/validation/h3_passN_20260415_133254 \\
      --a4_run runs/validation/a4_block_rerank_20260415_182338 \\
      --a5_run runs/validation/a5_prompt_template_20260415_191434 \\
      --a6_run runs/validation/a6_gen_length_20260416_012648
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_dir(run_dir: Path, pattern: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for p in sorted((run_dir / "per_prompt").glob(pattern)):
        rec = json.loads(p.read_text(encoding="utf-8"))
        out[int(rec["idx"])] = rec
    return out


def _h3_judge(rec: dict) -> tuple[bool, bool, dict]:
    """Return (rescued, stuck, best_t_per_k)."""
    temps = rec.get("temps", {})
    p8 = {t: float(d.get("pass@8", 0.0)) for t, d in temps.items()}
    rescued = any(v >= 1.0 for v in p8.values())
    stuck = len(p8) > 0 and all(v == 0.0 for v in p8.values())
    return rescued, stuck, p8


def _a_judge(rec: dict, field: str, baseline_key: str) -> bool:
    per = rec.get(field, {})
    if per.get(baseline_key, False):
        return False
    return any(per.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h3_run", type=str, required=True)
    ap.add_argument("--a4_run", type=str, required=True)
    ap.add_argument("--a5_run", type=str, required=True)
    ap.add_argument("--a6_run", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    h3_run = Path(args.h3_run)
    a4_run = Path(args.a4_run)
    a5_run = Path(args.a5_run)
    a6_run = Path(args.a6_run)

    h3 = _load_dir(h3_run, "fail_*.json")
    a4 = _load_dir(a4_run, "*.json")
    a5 = _load_dir(a5_run, "*.json")
    a6 = _load_dir(a6_run, "*.json")

    common = sorted(set(h3) & set(a4) & set(a5) & set(a6))
    if not common:
        raise SystemExit(
            "H3 / A4 / A5 / A6 idx 不相交，检查 run_dir 是否指向同一 fail set"
        )

    h3_rescue: set[int] = set()
    h3_stuck: set[int] = set()
    for i in common:
        rescued, stuck, _ = _h3_judge(h3[i])
        if rescued:
            h3_rescue.add(i)
        if stuck:
            h3_stuck.add(i)

    a4_rescue = {i for i in common if _a_judge(a4[i], "per_layout", "bl32")}
    a5_rescue = {i for i in common if _a_judge(a5[i], "per_template", "baseline")}
    a6_rescue = {i for i in common if _a_judge(a6[i], "per_length", "g128")}

    a_union = a4_rescue | a5_rescue | a6_rescue
    full_union = a_union | h3_rescue
    ceiling = set(common) - full_union

    n = len(common)
    result: dict = {
        "n_common": n,
        "runs": {
            "h3": str(h3_run),
            "a4": str(a4_run),
            "a5": str(a5_run),
            "a6": str(a6_run),
        },
        "rescue_counts": {
            "h3": len(h3_rescue),
            "a4": len(a4_rescue),
            "a5": len(a5_rescue),
            "a6": len(a6_rescue),
            "a_union": len(a_union),
            "full_union": len(full_union),
            "true_capacity_ceiling": len(ceiling),
            "h3_stuck_at_all_T": len(h3_stuck),
        },
        "rescue_rates": {
            "h3":      len(h3_rescue) / n,
            "a4":      len(a4_rescue) / n,
            "a5":      len(a5_rescue) / n,
            "a6":      len(a6_rescue) / n,
            "a_union": len(a_union) / n,
            "full_union": len(full_union) / n,
            "true_capacity_ceiling": len(ceiling) / n,
        },
        "rescue_sets": {
            "h3": sorted(h3_rescue),
            "a4": sorted(a4_rescue),
            "a5": sorted(a5_rescue),
            "a6": sorted(a6_rescue),
            "a_union": sorted(a_union),
            "full_union": sorted(full_union),
            "h3_stuck_at_all_T": sorted(h3_stuck),
            "true_capacity_ceiling": sorted(ceiling),
        },
        "h3_breakdown": {
            "in_a_union":      sorted(h3_rescue & a_union),
            "outside_a_union": sorted(h3_rescue - a_union),
            "fraction_explained_by_a_union": (
                len(h3_rescue & a_union) / len(h3_rescue) if h3_rescue else 0.0
            ),
        },
        "only_by_axis": {
            "h3_only": sorted(h3_rescue - a4_rescue - a5_rescue - a6_rescue),
            "a4_only": sorted(a4_rescue - a5_rescue - a6_rescue - h3_rescue),
            "a5_only": sorted(a5_rescue - a4_rescue - a6_rescue - h3_rescue),
            "a6_only": sorted(a6_rescue - a4_rescue - a5_rescue - h3_rescue),
        },
        "pairwise_intersections": {
            "h3_and_a4": sorted(h3_rescue & a4_rescue),
            "h3_and_a5": sorted(h3_rescue & a5_rescue),
            "h3_and_a6": sorted(h3_rescue & a6_rescue),
            "a4_and_a5": sorted(a4_rescue & a5_rescue),
            "a4_and_a6": sorted(a4_rescue & a6_rescue),
            "a5_and_a6": sorted(a5_rescue & a6_rescue),
        },
    }

    out_path = (
        Path(args.out) if args.out
        else h3_run.parent / f"p6_h3_crossref_{h3_run.name}.json"
    )
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("P6 — H3 × (A4 ∪ A5 ∪ A6) crossref")
    print("=" * 70)
    print(f"  n_common = {n}")
    print(f"  H3 rescue        = {len(h3_rescue):>3}/{n}  ({len(h3_rescue)/n:.2%})")
    print(f"  H3 stuck (all T) = {len(h3_stuck):>3}/{n}  ({len(h3_stuck)/n:.2%})")
    print(f"  A4 rescue        = {len(a4_rescue):>3}/{n}  ({len(a4_rescue)/n:.2%})")
    print(f"  A5 rescue        = {len(a5_rescue):>3}/{n}  ({len(a5_rescue)/n:.2%})")
    print(f"  A6 rescue        = {len(a6_rescue):>3}/{n}  ({len(a6_rescue)/n:.2%})")
    print(f"  A∪ (A4|A5|A6)    = {len(a_union):>3}/{n}  ({len(a_union)/n:.2%})")
    print(f"  Full union       = {len(full_union):>3}/{n}  ({len(full_union)/n:.2%})")
    print(f"  True capacity    = {len(ceiling):>3}/{n}  ({len(ceiling)/n:.2%})  idx={result['rescue_sets']['true_capacity_ceiling']}")
    print()
    print(f"  H3 ⊆ A∪           = {result['h3_breakdown']['fraction_explained_by_a_union']:.1%}")
    print(f"  H3 outside A∪     = {result['h3_breakdown']['outside_a_union']}")
    print()
    print("  Axis-only:")
    for k, v in result["only_by_axis"].items():
        print(f"    {k:<8} = {v}")
    print()
    print(f"  H3 stuck set      = {sorted(h3_stuck)}")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
