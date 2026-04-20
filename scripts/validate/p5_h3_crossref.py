"""P5 —— H3 rescue ∩ A4∪A5 union_rescue 交叉引用

读 H3 per_prompt 和 A4/A5 per_prompt，算:
  - H3 rescue set (pass@k=4 或 8 对 k=1 有增益)
  - A4 ∪ A5 union_rescue set
  - 交集 / 差集
  - 结论：H3 diversity 是否跟 layout/template 多样性是同一信号？

用法:
  python scripts/validate/p5_h3_crossref.py \\
      --h3_run runs/validation/h3_... \\
      --a4_run runs/validation/a4_block_rerank_20260415_182338 \\
      --a5_run runs/validation/a5_prompt_template_20260415_191434
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_per_prompt(run_dir: Path) -> dict[int, dict]:
    return {
        json.loads(p.read_text(encoding="utf-8"))["idx"]:
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((run_dir / "per_prompt").glob("*.json"))
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h3_run", type=str, required=True)
    ap.add_argument("--a4_run", type=str, required=True)
    ap.add_argument("--a5_run", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    h3 = load_per_prompt(Path(args.h3_run))
    a4 = load_per_prompt(Path(args.a4_run))
    a5 = load_per_prompt(Path(args.a5_run))

    common = set(h3) & set(a4) & set(a5)
    if not common:
        raise SystemExit("H3 / A4 / A5 idx 不相交，确认 run_dir")

    # A4 rescue: base=bl32 fail AND any layout correct
    a4_rescue = {
        i for i in common
        if not a4[i]["per_layout"].get("bl32", False)
        and any(a4[i]["per_layout"].values())
    }
    # A5 rescue: baseline fail AND any template correct
    a5_rescue = {
        i for i in common
        if not a5[i]["per_template"].get("baseline", False)
        and any(a5[i]["per_template"].values())
    }
    union_rescue = a4_rescue | a5_rescue

    # H3 rescue: base pass@1=0 AND pass@k>0 for some k
    # H3 record shape varies; try common keys
    h3_rescue = set()
    for i in common:
        r = h3[i]
        # adapt: try 'pass_at_k' dict, or 'base_correct' + 'any_correct'
        pass_k = r.get("pass_at_k") or {}
        base = pass_k.get("1") if pass_k else r.get("base_correct")
        anyk = any(v for k, v in pass_k.items() if k != "1") if pass_k else r.get("any_correct")
        if base is False and anyk:
            h3_rescue.add(i)

    result = {
        "n_common": len(common),
        "h3_rescue_count": len(h3_rescue),
        "a4_rescue_count": len(a4_rescue),
        "a5_rescue_count": len(a5_rescue),
        "union_rescue_count": len(union_rescue),
        "h3_in_union": sorted(h3_rescue & union_rescue),
        "h3_outside_union": sorted(h3_rescue - union_rescue),
        "h3_rescue_set": sorted(h3_rescue),
        "union_rescue_set": sorted(union_rescue),
    }
    n_in = len(h3_rescue & union_rescue)
    n_total = len(h3_rescue)
    result["fraction_h3_explained_by_union"] = n_in / n_total if n_total else 0.0

    out_path = Path(args.out) if args.out else (
        Path(args.h3_run).parent / f"p5_h3_crossref_{Path(args.h3_run).name}.json"
    )
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== P5 H3 × (A4 ∪ A5) crossref ===")
    for k in ["n_common", "h3_rescue_count", "a4_rescue_count", "a5_rescue_count",
              "union_rescue_count"]:
        print(f"  {k}: {result[k]}")
    print(f"  H3 ∩ union:          {result['h3_in_union']}")
    print(f"  H3 outside union:    {result['h3_outside_union']}")
    print(f"  H3 explained by union: {result['fraction_h3_explained_by_union']:.1%}")
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
