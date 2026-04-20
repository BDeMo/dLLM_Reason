"""T7: self-distill data generator.

Generate (prompt, correct_output) pairs by sampling LLaDA at T>0 and
keeping samples where the extracted answer matches ground truth.

Usage:
    # gen from fail set (the main target for T7 self-distill)
    python scripts/validate/t7_gen_correct_samples.py \\
        --groups fail --n 60 --n_samples 8 --temperature 0.7

    # gen from both fail+ok with multiple temperatures
    python scripts/validate/t7_gen_correct_samples.py \\
        --groups fail,ok --n 60 --temperatures 0.3,0.7,1.0 --n_samples 8

    # select shortest correct sample per prompt as target
    python scripts/validate/t7_gen_correct_samples.py \\
        --groups fail --n 60 --pick shortest

Output JSONL (per line):
    {
      "group": "fail", "idx": 0, "gt": "70000",
      "question": "<prompt text>",
      "answer": "<selected correct output text>",
      "selection": "shortest",
      "temperature": 0.7,
      "n_candidates": 3
    }

This JSONL can be consumed directly by src/dllm_reason/data/jsonl_dataset.py
(thin wrapper around ReasoningDataset) for the SFT training loop.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from _runlib import RunDir, ProgressPrinter, add_common_args, resolve_run_dir
from _http_client import ValidationAPIClient, add_server_arg

sys.path.insert(0, str(Path(__file__).parent))
from h1_remask_rescue import is_correct

ROOT = Path(__file__).resolve().parents[2]
SCOPE_FAIL = ROOT / "runs" / "validation" / "scope_fail_prompts.json"
SCOPE_OK = ROOT / "runs" / "validation" / "scope_ok_prompts.json"
OUT_BASE = ROOT / "runs" / "validation"


# ── Sample selection policies ─────────────────────────────────────────────────

def pick_candidate(candidates: list[dict], pick: str) -> dict:
    """Pick one correct sample from candidates per the 'pick' policy.

    candidates: list of {"output": str, "temperature": float, ...}, all already correct
    pick: "shortest" | "longest" | "first" | "random"
    """
    if not candidates:
        raise ValueError("no candidates to pick from")
    if pick == "shortest":
        return min(candidates, key=lambda c: len(c["output"]))
    if pick == "longest":
        return max(candidates, key=lambda c: len(c["output"]))
    if pick == "first":
        return candidates[0]
    if pick == "random":
        import random
        return random.Random(42).choice(candidates)
    raise ValueError(f"unknown pick policy: {pick}")


# ── Main ──────────────────────────────────────────────────────────────────────

def load_prompts(groups: list[str], n: int) -> list[tuple[str, int, dict]]:
    out = []
    if "fail" in groups:
        fails = json.loads(SCOPE_FAIL.read_text(encoding="utf-8"))[:n]
        for i, r in enumerate(fails):
            out.append(("fail", i, r))
    if "ok" in groups:
        oks = json.loads(SCOPE_OK.read_text(encoding="utf-8"))[:n]
        for i, r in enumerate(oks):
            out.append(("ok", i, r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60,
                    help="top-N per group")
    ap.add_argument("--groups", type=str, default="fail",
                    help="comma-separated: fail / ok / fail,ok")
    ap.add_argument("--temperatures", type=str, default="0.7",
                    help="comma-separated temperatures to try, e.g. '0.3,0.7,1.0'")
    ap.add_argument("--n_samples", type=int, default=8,
                    help="samples per (prompt, temperature)")
    ap.add_argument("--gen_length", type=int, default=192,
                    help="generation length (use 160 for fast, 192 for broader coverage)")
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--num_steps", type=int, default=None,
                    help="default = gen_length (coupled per E1 finding)")
    ap.add_argument("--pick", type=str, default="shortest",
                    choices=["shortest", "longest", "first", "random"],
                    help="selection policy for per-prompt SFT target "
                         "from correct candidates")
    ap.add_argument("--out_jsonl", type=str, default=None,
                    help="output JSONL path (default: <run_dir>/t7_sft.jsonl)")
    add_common_args(ap)
    add_server_arg(ap)
    args = ap.parse_args()

    if args.num_steps is None:
        args.num_steps = args.gen_length

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    temps = [float(t.strip()) for t in args.temperatures.split(",") if t.strip()]
    prompts = load_prompts(groups, args.n)

    run_dir = resolve_run_dir(args, "t7_selfdistill", OUT_BASE)
    rd = RunDir(
        run_dir, "T7-SelfDistill",
        config={
            **vars(args),
            "groups": groups,
            "temperatures": temps,
            "n_prompts": len(prompts),
        },
        resume=args.resume,
    )
    print(f"[T7] run_dir = {rd.dir}")
    print(f"[T7] prompts: {len(prompts)}  "
          f"(total samples per prompt = {len(temps) * args.n_samples})")

    out_path = Path(args.out_jsonl) if args.out_jsonl else rd.dir / "t7_sft.jsonl"

    def is_done(key: str) -> bool:
        p = rd.per_prompt / f"{key}.json"
        return p.exists()

    def prompt_key(group: str, i: int) -> str:
        return f"{group}_{i:04d}"

    todo = [(g, i, r) for (g, i, r) in prompts if not is_done(prompt_key(g, i))]
    print(f"[T7] done={len(prompts) - len(todo)}  todo={len(todo)}")

    if args.dry_run:
        print(f"[T7] DRY RUN — would sample "
              f"{len(todo) * len(temps) * args.n_samples} total calls")
        return

    api = ValidationAPIClient(args.server_url)
    api.check_health()

    pp = ProgressPrinter(len(todo), tag="T7 ")
    for group, idx, rec in todo:
        prompt, gt = rec["prompt"], rec["ground_truth"]
        key = prompt_key(group, idx)

        candidates = []
        for T in temps:
            for s_i in range(args.n_samples):
                try:
                    out = api.generate(
                        prompt, strategy="confidence",
                        max_new_tokens=args.gen_length,
                        num_steps=args.num_steps,
                        block_length=args.block_length,
                        temperature=T,
                    )
                except Exception as e:
                    print(f"[T7] WARN: sample fail at {key} T={T} s={s_i}: {e}")
                    continue
                if is_correct(out, gt):
                    candidates.append({
                        "output": out,
                        "temperature": T,
                        "sample_idx": s_i,
                    })

        n_cand = len(candidates)
        if n_cand > 0:
            chosen = pick_candidate(candidates, args.pick)
            rec_out = {
                "group": group, "idx": idx, "gt": gt,
                "question": prompt,
                "answer": chosen["output"],
                "selection": args.pick,
                "temperature": chosen["temperature"],
                "sample_idx": chosen["sample_idx"],
                "n_candidates": n_cand,
                "n_total_samples": len(temps) * args.n_samples,
                "all_candidates": candidates,  # keep full list for re-picking later
            }
        else:
            rec_out = {
                "group": group, "idx": idx, "gt": gt,
                "question": prompt,
                "answer": None,
                "selection": args.pick,
                "temperature": None,
                "n_candidates": 0,
                "n_total_samples": len(temps) * args.n_samples,
                "all_candidates": [],
            }

        # Atomic write per-prompt record (same pattern as strategy_search.py)
        import os
        path = rd.per_prompt / f"{key}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec_out, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, path)
        pp.tick(f"{key}  {n_cand}/{len(temps) * args.n_samples}")

    # ── Build SFT JSONL from all per_prompt files ────────────────────────────
    all_recs = []
    for group, idx, rec in prompts:
        p = rd.per_prompt / f"{prompt_key(group, idx)}.json"
        if p.exists():
            all_recs.append(json.loads(p.read_text(encoding="utf-8")))

    n_with = sum(1 for r in all_recs if r["n_candidates"] > 0)
    n_without = len(all_recs) - n_with

    with out_path.open("w", encoding="utf-8") as f:
        for r in all_recs:
            if r["answer"] is None:
                continue  # skip prompts with no correct sample (T7 abstain)
            sft_pair = {
                "group": r["group"],
                "idx": r["idx"],
                "gt": r["gt"],
                "question": r["question"],
                "answer": r["answer"],
                "selection": r["selection"],
                "temperature": r["temperature"],
                "n_candidates": r["n_candidates"],
            }
            f.write(json.dumps(sft_pair, ensure_ascii=False) + "\n")

    # Summary
    summary = {
        "n_prompts": len(all_recs),
        "n_with_correct": n_with,
        "n_without_correct": n_without,
        "cover_rate": n_with / max(len(all_recs), 1),
        "sft_pairs_written": n_with,
        "out_jsonl": str(out_path),
    }
    rd.write_summary(summary)

    print()
    print("═" * 60)
    print(f"[T7] prompts with ≥1 correct sample: {n_with}/{len(all_recs)} "
          f"({n_with / max(len(all_recs), 1):.2%})")
    print(f"[T7] SFT pairs → {out_path}")
    print(f"[T7] summary   → {rd.summary_path}")


if __name__ == "__main__":
    main()
