#!/usr/bin/env python
"""Self-Consistency (SC) post-processor for T6 decoding outputs.

Reads per_prompt/*.json produced by h3_passN_at_temperature.py (which
now persists answer_list alongside correct_list) and computes:
  - SC@k = mode(answer_list[:k]) == gt  for k ∈ {1, 4, 8}
  - pass@k (unchanged, as reference)

Why SC:  pass@N is ORACLE (any correct → rescue). SC is DEPLOYABLE
(one deterministic answer per prompt = majority vote of the N samples).
For gsm8k (single-number answers) SC tracks pass@N closely.

Zero-GPU post-processing. Requires runs produced AFTER the h3_passN
patch that saves answer_list (commit with this file).

Usage:
  # aggregate SC across all runs in one ablate dir
  python scripts/t6_self_consistency.py \
      --dir runs/validation/t6_decode_ablate/<ckpt_label>

  # or a specific h3_passN run dir
  python scripts/t6_self_consistency.py \
      --run_dir runs/validation/t6_passN/<run>

  # or all h3_passN runs under t6_passN/
  python scripts/t6_self_consistency.py --dir runs/validation/t6_passN

Outputs:
  <dir>/sc_summary.md           — one row per (run, temp) with pass/SC
  <dir>/sc_summary_<ts>.md      — timestamped copy
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from datetime import datetime
from pathlib import Path


def mode_of(answers: list[float | None]) -> float | None:
    """Return the mode of non-None answers, or None if all None."""
    clean = [a for a in answers if a is not None]
    if not clean:
        return None
    c = Counter(clean)
    return c.most_common(1)[0][0]


def sc_at_k(answers: list[float | None], gt_num: float | None, k: int) -> int:
    """SC@k: is mode of first k samples' answers == gt? (1/0)"""
    if gt_num is None:
        return 0
    m = mode_of(answers[:k])
    if m is None:
        return 0
    return int(abs(m - gt_num) < 1e-4)


def pass_at_k(corrects: list[bool], k: int) -> int:
    return int(any(corrects[:k]))


def extract_num(s):
    import re
    nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", str(s or ""))
    if not nums: return None
    try: return float(nums[-1].replace(",", ""))
    except: return None


def analyze_run(run_dir: Path) -> dict:
    """For one h3_passN run dir: aggregate pass@k and SC@k per (group, temp)."""
    pp_dir = run_dir / "per_prompt"
    if not pp_dir.is_dir():
        return {}

    # per (temp, group): {"pass": {1:..., 4:..., 8:...}, "sc": {...}, "n": N}
    buckets: dict = {}

    for f in sorted(pp_dir.glob("*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        group = rec.get("group")
        gt_num = extract_num(rec.get("gt"))
        temps = rec.get("temps", {})
        for T, d in temps.items():
            corrects = d.get("correct_list", [])
            answers = d.get("answer_list")
            if answers is None:
                # pre-patch run: can't compute SC, only pass
                answers = []
            key = (group, T)
            b = buckets.setdefault(key, {"pass": Counter(), "sc": Counter(),
                                         "n": 0, "has_answers": 0})
            b["n"] += 1
            for k in (1, 4, 8):
                b["pass"][k] += pass_at_k(corrects, k)
            if answers:
                b["has_answers"] += 1
                for k in (1, 4, 8):
                    b["sc"][k] += sc_at_k(answers, gt_num, k)
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None,
                    help="scan all <sub>/per_prompt/ under this root")
    ap.add_argument("--run_dir", default=None,
                    help="a single h3_passN run dir (has per_prompt/ inside)")
    args = ap.parse_args()

    if args.run_dir:
        targets = [Path(args.run_dir)]
        root = Path(args.run_dir).parent
    elif args.dir:
        root = Path(args.dir)
        if not root.is_dir():
            print(f"[SC] {root} not found"); return
        targets = [d for d in root.iterdir()
                   if d.is_dir() and (d / "per_prompt").is_dir()]
    else:
        print("[SC] pass --run_dir or --dir"); return
    if not targets:
        print("[SC] no per_prompt dirs found"); return

    lines = [
        "# Self-Consistency post-processor",
        "",
        "SC@k = mode of first k samples' answers equals gt (deployable, no oracle).",
        "pass@k = any of first k is correct (oracle upper bound).",
        "SC ≤ pass always; the gap = cost of removing the oracle.",
        "",
        "| run | temp | group | n | pass@1 | pass@4 | pass@8 | SC@1 | SC@4 | SC@8 | has_ans |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run_dir in sorted(targets):
        buckets = analyze_run(run_dir)
        if not buckets:
            lines.append(f"| {run_dir.name} | (no data) | — | 0 | — | — | — | — | — | — | — |")
            continue
        # sort buckets (group, temp)
        for (group, T), b in sorted(buckets.items(),
                                     key=lambda kv: (kv[0][0], float(kv[0][1]))):
            n = b["n"]
            def pct(x): return f"{100*x/max(n,1):.1f}%"
            has = b["has_answers"]
            sc_cells = ("—", "—", "—") if has == 0 else tuple(pct(b["sc"][k]) for k in (1,4,8))
            lines.append(
                f"| {run_dir.name} | {T} | {group} | {n} "
                f"| {pct(b['pass'][1])} | {pct(b['pass'][4])} | {pct(b['pass'][8])} "
                f"| {sc_cells[0]} | {sc_cells[1]} | {sc_cells[2]} "
                f"| {has}/{n} |"
            )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_md = root / f"sc_summary_{ts}.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    (root / "sc_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[SC] summary → {root / 'sc_summary.md'}")


if __name__ == "__main__":
    main()
