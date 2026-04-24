#!/usr/bin/env python
"""Aggregate all runs under runs/validation/t6_passN/ into a summary.md.

Standalone version of the aggregator inside t6_passN.sh — use this to
re-aggregate after the C1 fix on already-completed runs, without
re-running the (expensive) eval step.

Reads every <dir>/summary.json and produces:
  - runs/validation/t6_passN/summary.md (one row per ckpt × temp)
  - runs/validation/t6_passN/summary_<ts>.md (timestamped copy)

Usage:
  python scripts/t6_passN_aggregate.py
  python scripts/t6_passN_aggregate.py --dir runs/validation/t6_passN
"""
from __future__ import annotations
import argparse, json
from datetime import datetime
from pathlib import Path


def fmt_pct(x):
    try:
        return f"{float(x):.1%}"
    except (TypeError, ValueError):
        return "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/validation/t6_passN",
                    help="root dir containing <ckpt_run>/summary.json")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        print(f"[AGG] {root} not found"); return

    rows = []
    for sj in sorted(root.glob("*/summary.json")):
        try:
            v = json.loads(sj.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[AGG] skip {sj}: {e}"); continue
        rows.append((sj.parent.name, v))

    if not rows:
        print("[AGG] no summary.json found"); return

    # Union of temps across all rows
    all_temps = set()
    for _, v in rows:
        all_temps.update(v.get("fail_stats", {}).keys())
    temps = sorted(all_temps, key=float)

    lines = [
        "# T6 pass@N eval — aggregated",
        "",
        "h3_passN reports pass@k for k ∈ {1, 4, 8} where pass@8 is really",
        "pass@N (= n_samples). Column names are hard-coded upstream regardless of N.",
        "",
        "| ckpt | temp | fail p@1 | fail p@4 | fail p@8 | ok p@1 | ok p@4 | ok p@8 | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, v in rows:
        fs = v.get("fail_stats", {})
        os_ = v.get("ok_stats", {})
        verdict = v.get("verdict", "?")
        for T in temps:
            f = fs.get(T, {})
            o = os_.get(T, {})
            if not f and not o:
                continue
            lines.append(
                f"| {name} | {T} "
                f"| {fmt_pct(f.get('pass@1'))} "
                f"| {fmt_pct(f.get('pass@4'))} "
                f"| {fmt_pct(f.get('pass@8'))} "
                f"| {fmt_pct(o.get('pass@1'))} "
                f"| {fmt_pct(o.get('pass@4'))} "
                f"| {fmt_pct(o.get('pass@8'))} "
                f"| {verdict if T == temps[-1] else ''} |"
            )
        lines.append(
            f"| {name} | (max-T) | — | — | "
            f"{fmt_pct(v.get('fail_pass@8_max'))} "
            f"| — | — | {fmt_pct(v.get('ok_pass@8_max'))} | — |"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    content = "\n".join(lines)
    (root / f"summary_{ts}.md").write_text(content, encoding="utf-8")
    (root / "summary.md").write_text(content, encoding="utf-8")
    print(content)
    print(f"\n[AGG] wrote {root / 'summary.md'}")


if __name__ == "__main__":
    main()
