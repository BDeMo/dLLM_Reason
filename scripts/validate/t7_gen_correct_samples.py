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

def load_prompts(
    groups: list[str], n: int,
    scope_path: str | None = None,
    scope_group: str = "gsm8k",
) -> list[tuple[str, int, dict]]:
    """Load prompts.

    If ``scope_path`` is given, read the whole file as a single group labelled
    ``scope_group`` (e.g. 'gsm8k' for the full gsm8k train set loader output).
    This is the 'Full' track in the v1.6 plan.

    Otherwise, read default scope_fail_prompts.json / scope_ok_prompts.json
    per ``groups`` (e.g. "fail" or "fail,ok"). This is the 'Fast' track.
    """
    out = []
    if scope_path:
        data = json.loads(Path(scope_path).read_text(encoding="utf-8"))
        if n:
            data = data[:n]
        for i, r in enumerate(data):
            out.append((scope_group, i, r))
        return out

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
    ap.add_argument("--model", type=str, default="checkpoints/llada-instruct",
                    help="HF-format model path. T7 typically generates from "
                         "the BASE model (or a T6 ckpt) on gsm8k train fail "
                         "prompts to collect correct trajectories. Direct "
                         "in-process load — no HTTP / no serve.py required.")
    ap.add_argument("--n", type=int, default=60,
                    help="top-N per group (0 = all in scope)")
    ap.add_argument("--groups", type=str, default="fail",
                    help="comma-separated: fail / ok / fail,ok (only if "
                         "--scope_path not set)")
    ap.add_argument("--temperatures", type=str, default="0.7",
                    help="comma-separated temperatures, e.g. '0.3,0.7,1.0'")
    ap.add_argument("--n_samples", type=int, default=8,
                    help="samples per (prompt, temperature)")
    ap.add_argument("--gen_length", type=int, default=192)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--num_steps", type=int, default=None,
                    help="default = gen_length (coupled per E1 finding)")
    ap.add_argument("--pick", type=str, default="shortest",
                    choices=["shortest", "longest", "first", "random"])
    ap.add_argument("--out_jsonl", type=str, default=None)
    ap.add_argument("--scope_path", type=str, default=None,
                    help="custom scope JSON (e.g. gsm8k_train_prompts.json).")
    ap.add_argument("--scope_group", type=str, default="gsm8k")
    ap.add_argument("--prompt_batch", default="auto",
                    help="cross-prompt batching (P): 'auto' probes max P "
                         "that fits memory; explicit int forces; 1 disables")
    ap.add_argument("--prompt_shard", default="0/1",
                    help="<idx>/<total>: round-robin partition prompts across "
                         "shards (multi-GPU). Each shard processes "
                         "i %% total == idx.")
    add_common_args(ap)
    args = ap.parse_args()
    try:
        SHARD_IDX, SHARD_TOTAL = (int(x) for x in args.prompt_shard.split("/"))
        assert 0 <= SHARD_IDX < SHARD_TOTAL
    except Exception:
        raise SystemExit(f"--prompt_shard must be 'idx/total': got {args.prompt_shard!r}")

    if args.num_steps is None:
        args.num_steps = args.gen_length

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    temps = [float(t.strip()) for t in args.temperatures.split(",") if t.strip()]
    prompts = load_prompts(groups, args.n,
                           scope_path=args.scope_path,
                           scope_group=args.scope_group)

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

    # Apply prompt-shard filter (multi-GPU)
    todo = [(g, i, r) for k, (g, i, r) in enumerate(todo)
            if k % SHARD_TOTAL == SHARD_IDX]
    print(f"[T7] shard {SHARD_IDX}/{SHARD_TOTAL}: todo after shard = {len(todo)}")

    if args.dry_run:
        print(f"[T7] DRY RUN — would sample "
              f"{len(todo) * len(temps) * args.n_samples} total calls")
        return

    # ── Direct in-process model load (no HTTP / no serve.py) ─────────────
    import torch
    from transformers import AutoModel, AutoTokenizer
    sys.path.insert(0, str(Path(__file__).parent))
    from h1_remask_rescue import (generate_batched_multi, _get_mask_token_id,
                                   is_correct, extract_answer)

    print(f"[T7] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).cuda().eval()
    mask_id = _get_mask_token_id(model, tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"[T7] mask_id = {mask_id}  pad_id = {pad_id}")

    # Probe attn_mask once
    try:
        with torch.no_grad():
            _x = torch.full((1, 16), mask_id, dtype=torch.long, device="cuda")
            _a = torch.ones((1, 16), dtype=torch.long, device="cuda")
            _ = model(_x, attention_mask=_a).logits
            del _x, _a, _
        ATTN_OK = True
    except TypeError:
        ATTN_OK = False
    torch.cuda.empty_cache()
    print(f"[T7] attention_mask supported: {ATTN_OK}")

    # Resolve P_BATCH (autotune or explicit)
    pb_arg = str(args.prompt_batch).strip().lower()
    if pb_arg in ("auto", "0"):
        # Probe P at worst-case length
        worst = max(len(tok(tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            add_generation_prompt=True, tokenize=False))["input_ids"])
            for _, _, r in todo[:50] or todo)
        L = worst + args.gen_length
        for P in [16, 12, 8, 4, 2, 1]:
            try:
                B = P * args.n_samples
                torch.cuda.empty_cache()
                _x = torch.full((B, L), mask_id, dtype=torch.long, device="cuda")
                _a = torch.ones((B, L), dtype=torch.long, device="cuda")
                with torch.no_grad():
                    if ATTN_OK:
                        _ = model(_x, attention_mask=_a).logits
                    else:
                        _ = model(_x).logits
                del _x, _a, _
                torch.cuda.empty_cache()
                P_BATCH = P
                print(f"[T7] autotune: P_BATCH={P} (B={B}) fits at L={L}")
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
        else:
            P_BATCH = 1
    else:
        P_BATCH = max(1, int(pb_arg))

    # Sort todo by prompt length (descending) to minimize in-chunk padding
    def _plen(rec):
        text = tok.apply_chat_template(
            [{"role": "user", "content": rec["prompt"]}],
            add_generation_prompt=True, tokenize=False)
        return len(tok(text)["input_ids"])
    todo = sorted(todo, key=lambda gir: -_plen(gir[2]))

    def chunked(lst, k):
        for i in range(0, len(lst), k):
            yield lst[i:i+k]

    pp = ProgressPrinter(len(todo), tag="T7 ")
    import os
    for chunk in chunked(todo, P_BATCH):
        prompts_text = [r["prompt"] for _, _, r in chunk]

        # candidates_per_prompt[i] = list of {"output", "temperature", "sample_idx"}
        candidates_per_prompt = [[] for _ in chunk]

        for T in temps:
            outs_per_prompt = generate_batched_multi(
                model, tok, prompts_text,
                n_samples=args.n_samples,
                gen_length=args.gen_length,
                steps=args.num_steps,
                block_length=args.block_length,
                temperature=T,
                mask_id=mask_id,
                pad_token_id=pad_id,
                _attn_mask_supported=ATTN_OK,
            )
            for i, (group, idx, rec) in enumerate(chunk):
                gt = rec["ground_truth"]
                for s_i, out in enumerate(outs_per_prompt[i]):
                    if is_correct(out, gt):
                        candidates_per_prompt[i].append({
                            "output": out,
                            "temperature": T,
                            "sample_idx": s_i,
                        })

        # Write per-prompt records
        for (group, idx, rec), candidates in zip(chunk, candidates_per_prompt):
            key = prompt_key(group, idx)
            n_cand = len(candidates)
            if n_cand > 0:
                chosen = pick_candidate(candidates, args.pick)
                rec_out = {
                    "group": group, "idx": idx, "gt": rec["ground_truth"],
                    "question": rec["prompt"],
                    "answer": chosen["output"],
                    "selection": args.pick,
                    "temperature": chosen["temperature"],
                    "sample_idx": chosen["sample_idx"],
                    "n_candidates": n_cand,
                    "n_total_samples": len(temps) * args.n_samples,
                    "all_candidates": candidates,
                }
            else:
                rec_out = {
                    "group": group, "idx": idx, "gt": rec["ground_truth"],
                    "question": rec["prompt"],
                    "answer": None,
                    "selection": args.pick,
                    "temperature": None,
                    "n_candidates": 0,
                    "n_total_samples": len(temps) * args.n_samples,
                    "all_candidates": [],
                }
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
