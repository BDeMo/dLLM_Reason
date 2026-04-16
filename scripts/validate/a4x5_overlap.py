"""A4 × A5 overlap analysis.

读 `runs/validation/a4_block_rerank_*/per_prompt/*.json`
和 `runs/validation/a5_prompt_template_*/per_prompt/*.json`，
按 `idx` 对齐，算：

  - base_correct (A5 baseline，等于 A4 bl32)
  - A4_rescue_set  = {idx : bl32=False AND any layout True}
  - A5_rescue_set  = {idx : baseline=False AND any template True}
  - overlap        = A4 ∩ A5
  - union          = A4 ∪ A5
  - joint_any_rescue_rate = |union| / N
  - independence_factor   = |union| / (|A4| + |A5|)  (越接近 1 越不相交)
  - hypothetical 20-cell ensemble ceiling = any of (5 layout × 4 template) on fail prompts

也给每条 rescue prompt 打一份详细报告：哪个 layout / template 救了它。

用法：
  python scripts/validate/a4x5_overlap.py
  # 或指定特定 run_dir：
  python scripts/validate/a4x5_overlap.py --a4_run <path> --a5_run <path>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

OUT_BASE = Path(__file__).parent.parent.parent / "runs" / "validation"


def latest_run(pattern: str) -> Path:
    matches = sorted(glob.glob(str(OUT_BASE / pattern)))
    if not matches:
        raise FileNotFoundError(f"No match for {pattern} under {OUT_BASE}")
    return Path(matches[-1])


def load_records(run_dir: Path) -> dict[int, dict]:
    recs = {}
    for p in sorted((run_dir / "per_prompt").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        recs[d["idx"]] = d
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a4_run", type=str, default=None,
                    help="A4 run dir (default: latest a4_block_rerank_*)")
    ap.add_argument("--a5_run", type=str, default=None,
                    help="A5 run dir (default: latest a5_prompt_template_*)")
    ap.add_argument("--out", type=str, default=None,
                    help="Write JSON report to this path")
    args = ap.parse_args()

    a4 = Path(args.a4_run) if args.a4_run else latest_run("a4_block_rerank_*")
    a5 = Path(args.a5_run) if args.a5_run else latest_run("a5_prompt_template_*")
    print(f"[A4] {a4.name}")
    print(f"[A5] {a5.name}")

    r4 = load_records(a4)
    r5 = load_records(a5)
    common = sorted(set(r4.keys()) & set(r5.keys()))
    print(f"[info] A4 N={len(r4)}  A5 N={len(r5)}  common N={len(common)}")

    layouts = list(next(iter(r4.values()))["per_layout"].keys())
    templates = list(next(iter(r5.values()))["per_template"].keys())
    print(f"[info] layouts   = {layouts}")
    print(f"[info] templates = {templates}")

    # Base = A5 baseline == A4 bl32 (should match — both are default)
    base5 = {i: r5[i]["per_template"]["baseline"] for i in common}
    base4 = {i: r4[i]["per_layout"]["bl32"] for i in common}
    mismatch = [i for i in common if base5[i] != base4[i]]
    if mismatch:
        print(f"[WARN] base mismatch between A4 bl32 and A5 baseline on {len(mismatch)} prompts:")
        for i in mismatch[:5]:
            print(f"       idx={i}  a4_bl32={base4[i]}  a5_baseline={base5[i]}")
    base_correct = sum(1 for i in common if base5[i])

    # Rescue sets (on common set)
    a4_any = {i: any(r4[i]["per_layout"].values()) for i in common}
    a5_any = {i: any(r5[i]["per_template"].values()) for i in common}
    a4_rescue = {i for i in common if not base5[i] and a4_any[i]}
    a5_rescue = {i for i in common if not base5[i] and a5_any[i]}
    both_rescue  = a4_rescue & a5_rescue
    only_a4      = a4_rescue - a5_rescue
    only_a5      = a5_rescue - a4_rescue
    union_rescue = a4_rescue | a5_rescue
    fail_count = sum(1 for i in common if not base5[i])

    N = len(common)

    # Joint 20-cell ensemble ceiling (A4 ∪ A5 as if you ran both):
    # a prompt is rescuable iff (any layout correct) OR (any template correct).
    joint_any = {i: a4_any[i] or a5_any[i] for i in common}
    joint_rate = sum(joint_any.values()) / N

    def pct(x, total):
        return f"{x/total*100:.2f}%" if total else "n/a"

    print()
    print("=" * 70)
    print("A4 × A5 OVERLAP REPORT")
    print("=" * 70)
    print(f"N (common)                  = {N}")
    print(f"baseline correct            = {base_correct}  ({pct(base_correct, N)})")
    print(f"baseline wrong (fail set)   = {fail_count}  ({pct(fail_count, N)})")
    print()
    print(f"A4 rescue set               = {len(a4_rescue)}  ({pct(len(a4_rescue), N)} of N, {pct(len(a4_rescue), fail_count)} of fails)")
    print(f"A5 rescue set               = {len(a5_rescue)}  ({pct(len(a5_rescue), N)} of N, {pct(len(a5_rescue), fail_count)} of fails)")
    print(f"  both (A4 ∩ A5)            = {len(both_rescue)}")
    print(f"  only A4                   = {len(only_a4)}")
    print(f"  only A5                   = {len(only_a5)}")
    print(f"  union (A4 ∪ A5)           = {len(union_rescue)}  ({pct(len(union_rescue), N)} of N, {pct(len(union_rescue), fail_count)} of fails)")
    print()
    denom = len(a4_rescue) + len(a5_rescue)
    indep = len(union_rescue) / denom if denom else float("nan")
    print(f"independence factor         = {indep:.3f}  (1.0 = disjoint, 0.5 = full overlap)")
    print(f"joint any-rescue rate (20-cell ceiling)")
    print(f"                            = {joint_rate*100:.2f}% of N  ({sum(joint_any.values())}/{N})")
    print(f"                            = {sum(joint_any.values())-base_correct}/{fail_count}"
          f" = {pct(sum(joint_any.values())-base_correct, fail_count)} of fails rescued")

    # Per-prompt rescue breakdown
    def winning_layouts(i):
        return [k for k, v in r4[i]["per_layout"].items() if v]
    def winning_templates(i):
        return [k for k, v in r5[i]["per_template"].items() if v]

    print()
    print("-" * 70)
    print("Rescued prompts detail:")
    print("-" * 70)
    for i in sorted(union_rescue):
        tags = []
        if i in both_rescue: tags.append("BOTH")
        elif i in only_a4:   tags.append("A4 only")
        elif i in only_a5:   tags.append("A5 only")
        gt = r4[i]["gt"]
        wl = winning_layouts(i)
        wt = winning_templates(i)
        print(f"idx={i:3d}  gt={gt:>8}  [{', '.join(tags):8}]  layouts={wl}  templates={wt}")

    print()
    print("-" * 70)
    print("Per-config rescue contribution (how many unique fails does each config rescue alone):")
    print("-" * 70)
    for layout in layouts:
        ids = {i for i in common if not base5[i] and r4[i]["per_layout"][layout]}
        print(f"  A4.{layout:20s} rescues {len(ids):2d} fails")
    for tpl in templates:
        ids = {i for i in common if not base5[i] and r5[i]["per_template"][tpl]}
        print(f"  A5.{tpl:20s} rescues {len(ids):2d} fails")

    report = {
        "a4_run": str(a4),
        "a5_run": str(a5),
        "n_common": N,
        "base_correct": base_correct,
        "fail_count": fail_count,
        "a4_rescue": sorted(a4_rescue),
        "a5_rescue": sorted(a5_rescue),
        "both_rescue": sorted(both_rescue),
        "only_a4": sorted(only_a4),
        "only_a5": sorted(only_a5),
        "union_rescue": sorted(union_rescue),
        "independence_factor": indep,
        "joint_any_rescue_rate": joint_rate,
        "joint_fail_rescue_rate": (sum(joint_any.values())-base_correct)/fail_count if fail_count else 0.0,
        "rescued_detail": [
            {
                "idx": i,
                "gt": r4[i]["gt"],
                "winning_layouts": winning_layouts(i),
                "winning_templates": winning_templates(i),
            }
            for i in sorted(union_rescue)
        ],
    }

    out_path = Path(args.out) if args.out else OUT_BASE / f"a4x5_overlap_{a4.name.split('_')[-1]}_{a5.name.split('_')[-1]}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"[OK] report written → {out_path}")


if __name__ == "__main__":
    main()
