#!/usr/bin/env python
"""Stitch t7's comparison.md with cached baseline + t6 rows.

Phase D.1 used to re-eval baseline + t6 every time even though those
results are deterministic for the same canonical config. This wrapper:

1. Reads the new run's per-ckpt summary.json (only t7 evaluated)
2. Finds the most recent prior comparison.md that includes both
   'baseline' and 't6' rows AND was run on the same base_ckpt
3. Copies baseline + t6 rows verbatim, appends new t7 row
4. Writes combined comparison.md to new_eval/comparison.md

Saves ~6-8 min of redundant eval per T7 run.

Usage:
    python scripts/t7_stitch_comparison.py \\
        --new_eval runs/validation/t7_eval_<TS> \\
        --base_ckpt runs/training/v161_t6_ablate/hf_step_336
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path


def find_prior_comparison(base_ckpt: str, exclude: Path):
    """Find the most recent t7_eval_* dir whose comparison.md has the
    given base_ckpt as 't6' AND has both baseline and t7 rows."""
    base_label = Path(base_ckpt).name  # e.g. hf_step_336
    eval_dirs = sorted(Path("runs/validation").glob("t7_eval_*"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    for d in eval_dirs:
        if d == exclude:
            continue
        cm = d / "comparison.md"
        if not cm.exists():
            continue
        # Read summary.json to verify t6 ckpt path
        sj = d / "summary.json"
        if not sj.exists():
            continue
        try:
            data = json.loads(sj.read_text(encoding="utf-8"))
            ckpts = data.get("ckpts", [])
            t6_entry = next((c for c in ckpts if c["label"] == "t6"), None)
            base_entry = next((c for c in ckpts if c["label"] == "baseline"), None)
            if t6_entry and base_entry and Path(t6_entry["ckpt"]).name == base_label:
                return d, ckpts
        except Exception:
            continue
    return None, None


def fmt_row(c: dict) -> str:
    n_fail = c["n_fail"]; n_ok = c["n_ok"]
    fc = c["fail_correct"]; ok = c["ok_correct"]
    f18 = c.get("fail18_rescued", [])
    f18n = c.get("fail18_rescued_count", 0)
    cb = c.get("ceiling_broken", [])
    cbn = c.get("ceiling_broken_count", 0)
    return (f"| {c['label']} | {fc/n_fail:.2%} ({fc}/{n_fail}) "
            f"| {ok/n_ok:.2%} ({ok}/{n_ok}) "
            f"| {f18n}/18  `{f18}` "
            f"| {cbn}/5  `{cb}` |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_eval", required=True,
                    help="dir containing the new run's summary.json (only t7)")
    ap.add_argument("--base_ckpt", required=True,
                    help="t6 base ckpt path — used to match prior comparison.md")
    args = ap.parse_args()

    new_eval = Path(args.new_eval)
    sj = new_eval / "summary.json"
    if not sj.exists():
        print(f"[STITCH] no {sj}"); return 1
    data = json.loads(sj.read_text(encoding="utf-8"))
    ckpts = data.get("ckpts", [])
    t7_entry = next((c for c in ckpts if c["label"] == "t7"), None)
    if t7_entry is None:
        print(f"[STITCH] no 't7' entry in {sj}"); return 1

    # Find baseline + t6 from prior run
    prior_dir, prior_ckpts = find_prior_comparison(args.base_ckpt, exclude=new_eval)
    if prior_ckpts is None:
        print("[STITCH] no compatible prior comparison.md found — leaving "
              "new_eval/comparison.md as-is (only t7 row)")
        return 0

    base_entry = next(c for c in prior_ckpts if c["label"] == "baseline")
    t6_entry = next(c for c in prior_ckpts if c["label"] == "t6")
    print(f"[STITCH] reusing baseline + t6 from {prior_dir}")

    # Build combined comparison.md
    lines = [
        "# v1.6 Eval Comparison (stitched: baseline+t6 cached)",
        "",
        f"_Source for baseline+t6 rows: `{prior_dir.name}` (deterministic re-use)._",
        "",
        "| Label | fail pass@1 | ok pass@1 | FAIL18 rescued | ceiling broken |",
        "|---|---|---|---|---|",
        fmt_row(base_entry),
        fmt_row(t6_entry),
        fmt_row(t7_entry),
    ]
    out = new_eval / "comparison.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[STITCH] → {out}")

    # Also rewrite summary.json to include all 3 ckpts (downstream may rely on it)
    full = {
        "ckpts": [base_entry, t6_entry, t7_entry],
        "config": data.get("config", {}),
        "stitched_from": str(prior_dir),
    }
    sj.write_text(json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
