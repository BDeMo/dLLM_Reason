#!/usr/bin/env python
"""Aggregate per-shard summaries from orm_eval_bon.py into final summary.

After 8 shards finish, sum up their partial counts and produce
summary.json + summary.md identical to the single-GPU output.

Usage:
  python scripts/orm_eval_aggregate.py --eval_dir runs/validation/orm_eval_v1
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    shards = sorted(eval_dir.glob("summary_shard*.json"))
    if not shards:
        raise SystemExit(f"[ORM-AGG] no summary_shard*.json under {eval_dir}")
    print(f"[ORM-AGG] found {len(shards)} shard summaries")

    totals = {g: {k: 0 for k in ("n", "greedy", "SC@N", "BoN@N", "pass@N")}
              for g in ("fail", "ok")}
    config = None
    for sp in shards:
        s = json.loads(sp.read_text(encoding="utf-8"))
        config = config or s.get("config")
        for g in ("fail", "ok"):
            for k in totals[g]:
                totals[g][k] += s[g][k]

    summary = {"config": config, **totals}
    (eval_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    def pct(n, d): return f"{100*n/max(d,1):.1f}% ({n}/{d})"
    md = ["# ORM BoN evaluation",
          "",
          f"base: `{config.get('base_ckpt','?')}`",
          f"head: `{config.get('orm_head','?')}`",
          f"N samples: {config.get('n_samples','?')}, "
          f"T: {config.get('temperature','?')}",
          "",
          "| metric | fail rescue | ok retention |",
          "|---|---|---|"]
    for m in ("greedy", "SC@N", "BoN@N", "pass@N"):
        md.append(f"| {m} | {pct(totals['fail'][m], totals['fail']['n'])} "
                  f"| {pct(totals['ok'][m], totals['ok']['n'])} |")
    (eval_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n[ORM-AGG] → {eval_dir}/summary.{{json,md}}")


if __name__ == "__main__":
    main()
