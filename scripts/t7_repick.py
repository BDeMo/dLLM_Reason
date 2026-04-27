#!/usr/bin/env python
"""Re-aggregate t7_gen_correct_samples per_prompt files into a new jsonl
with a different `pick` policy — WITHOUT re-running inference.

Each per_prompt/<group>_<idx>.json stores `all_candidates`: every
correct sample from that prompt's (T × N) sweep. Saves the chosen one
under a specific pick policy. To switch policies (e.g. shortest →
first), just re-pick from the same candidates.

Saves ~1.5h vs re-running scripts/t7_pipeline.sh Phase A on 8 GPU.

Usage:
    python scripts/t7_repick.py \\
        --gen_dir runs/validation/t7_gen_20260426_192555 \\
        --pick first \\
        --out_jsonl t7_sft_first.jsonl

    # default --gen_dir = latest t7_gen_*/
    python scripts/t7_repick.py --pick first

    # also filter out short / truncated samples
    python scripts/t7_repick.py --pick first --min_len 100
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def pick_candidate(candidates: list[dict], pick: str) -> dict:
    if not candidates:
        raise ValueError("empty candidates")
    if pick == "shortest":
        return min(candidates, key=lambda c: len(c["output"]))
    if pick == "longest":
        return max(candidates, key=lambda c: len(c["output"]))
    if pick == "first":
        return candidates[0]
    if pick == "random":
        import random
        return random.Random(42).choice(candidates)
    raise ValueError(f"unknown pick: {pick}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", default=None,
                    help="directory containing per_prompt/*.json. Default: "
                         "latest runs/validation/t7_gen_*")
    ap.add_argument("--pick", default="first",
                    choices=["shortest", "longest", "first", "random"])
    ap.add_argument("--min_len", type=int, default=0,
                    help="filter out candidates with answer length < this "
                         "(chars). Default 0 = no filter.")
    ap.add_argument("--out_jsonl", default=None,
                    help="output path (default: <gen_dir>/t7_sft_<pick>.jsonl)")
    args = ap.parse_args()

    if args.gen_dir is None:
        cands = sorted(Path("runs/validation").glob("t7_gen_*"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            print("[REPICK] no t7_gen_* dirs found"); return
        args.gen_dir = str(cands[0])

    gen = Path(args.gen_dir)
    pp_dir = gen / "per_prompt"
    if not pp_dir.is_dir():
        print(f"[REPICK] no per_prompt under {gen}"); return

    out_path = Path(args.out_jsonl) if args.out_jsonl else \
        gen / f"t7_sft_{args.pick}{f'_minlen{args.min_len}' if args.min_len else ''}.jsonl"

    n_total = 0
    n_with = 0
    n_filtered_out = 0
    with out_path.open("w", encoding="utf-8") as f:
        for p in sorted(pp_dir.glob("*.json")):
            n_total += 1
            r = json.loads(p.read_text(encoding="utf-8"))
            cands = r.get("all_candidates", [])
            if args.min_len > 0:
                cands = [c for c in cands if len(c["output"]) >= args.min_len]
            if not cands:
                if r.get("n_candidates", 0) > 0:
                    n_filtered_out += 1
                continue
            chosen = pick_candidate(cands, args.pick)
            f.write(json.dumps({
                "group": r["group"], "idx": r["idx"], "gt": r["gt"],
                "question": r["question"], "answer": chosen["output"],
                "selection": args.pick,
                "min_len": args.min_len,
                "temperature": chosen["temperature"],
                "n_candidates": len(cands),
            }, ensure_ascii=False) + "\n")
            n_with += 1

    print(f"[REPICK] gen_dir       = {gen}")
    print(f"[REPICK] pick          = {args.pick}  min_len={args.min_len}")
    print(f"[REPICK] {n_with}/{n_total} prompts kept "
          f"({n_with / max(n_total, 1):.1%})")
    if n_filtered_out:
        print(f"[REPICK]   {n_filtered_out} prompts had candidates but "
              f"all < min_len → dropped")
    print(f"[REPICK] → {out_path}")


if __name__ == "__main__":
    main()
