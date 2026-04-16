"""P6 —— A4 独救条目的手工特征分析

目标: 看 A4 独救 (A4 rescue ∧ ¬A5 rescue) 的那几条在特征上跟其他 fail 有什么差异。
如果有显著信号 → 可以写 rule 直接当 layout 选择器，省 5× inference。

复用 p4 的 features()，按组比较:
  - A4-only-rescue   (A4 rescue ∧ ¬A5 rescue)
  - shared-rescue    (A4 rescue ∧ A5 rescue)
  - A5-only-rescue   (A5 rescue ∧ ¬A4 rescue)
  - fail-no-rescue   (baseline=F ∧ ¬A4 ∧ ¬A5)

用法:
  python scripts/validate/p6_a4_rescue_features.py \\
      --a4_run runs/validation/a4_block_rerank_20260415_182338 \\
      --a5_run runs/validation/a5_prompt_template_20260415_191434
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from p4_cot_broken_pattern import features, summary, mann_whitney_u

ROOT = Path(__file__).resolve().parents[2]
SCOPE = ROOT / "runs" / "validation" / "scope_fail_prompts.json"


def load_per_prompt(run_dir: Path) -> dict[int, dict]:
    out = {}
    for p in sorted((run_dir / "per_prompt").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out[d["idx"]] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a4_run", type=str, required=True)
    ap.add_argument("--a5_run", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    a4 = load_per_prompt(Path(args.a4_run))
    a5 = load_per_prompt(Path(args.a5_run))
    fails = json.loads(SCOPE.read_text(encoding="utf-8"))

    common = set(a4) & set(a5)
    a4_rescue = {i for i in common
                 if not a4[i]["per_layout"].get("bl32", False)
                 and any(a4[i]["per_layout"].values())}
    a5_rescue = {i for i in common
                 if not a5[i]["per_template"].get("baseline", False)
                 and any(a5[i]["per_template"].values())}
    base_fail = {i for i in common
                 if not a5[i]["per_template"].get("baseline", False)}

    groups = {
        "a4_only":     sorted(a4_rescue - a5_rescue),
        "shared":      sorted(a4_rescue & a5_rescue),
        "a5_only":     sorted(a5_rescue - a4_rescue),
        "no_rescue":   sorted(base_fail - a4_rescue - a5_rescue),
    }

    # feature extraction
    feat_keys = list(features("dummy").keys())
    group_feats: dict[str, list[dict]] = {}
    for g, idxs in groups.items():
        group_feats[g] = []
        for i in idxs:
            p = fails[i]["prompt"]
            f = features(p)
            group_feats[g].append({"idx": i, "prompt_head": p[:120], **f})

    stats = {
        g: {k: summary([x[k] for x in items]) for k in feat_keys}
        for g, items in group_feats.items()
    }

    # Pairwise: a4_only vs others
    sig = {}
    for other in ["shared", "a5_only", "no_rescue"]:
        pair = {}
        for k in feat_keys:
            a = [x[k] for x in group_feats["a4_only"]]
            b = [x[k] for x in group_feats[other]]
            U, z = mann_whitney_u(a, b)
            pair[k] = {"U": U, "z": z, "n_a4_only": len(a), "n_other": len(b)}
        sig[f"a4_only_vs_{other}"] = pair

    out = {
        "a4_run": str(args.a4_run),
        "a5_run": str(args.a5_run),
        "group_sizes": {g: len(v) for g, v in group_feats.items()},
        "group_feature_stats": stats,
        "significance": sig,
        "groups_detail": {g: items for g, items in group_feats.items()},
    }
    out_path = Path(args.out) if args.out else (
        ROOT / "runs" / "validation" / "p6_a4_rescue_features.json"
    )
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== P6 A4-rescue features ===")
    for g, v in group_feats.items():
        print(f"  {g}: {len(v)}")

    print("\n=== Feature means per group ===")
    header = "feature".ljust(18) + "".join(g.ljust(18) for g in groups)
    print(header)
    for k in feat_keys:
        row = k.ljust(18)
        for g in groups:
            s = stats[g].get(k, {})
            row += f"μ={s.get('mean', 0):.1f}".ljust(18)
        print(row)

    print("\n=== z-score (a4_only vs *) |z|>1.64 ~ p<0.1 ===")
    for pair, d in sig.items():
        print(f"  [{pair}]")
        for k, r in d.items():
            mark = " *" if abs(r["z"]) > 1.64 else ""
            print(f"    {k:<18} z={r['z']:+.2f}{mark}")

    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
