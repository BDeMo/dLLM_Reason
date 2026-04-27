#!/usr/bin/env python
"""Collect ORM training data: (prompt, output, label) triples.

Fresh sampling on gsm8k train using a frozen base model (T6 ckpt).
For each prompt, generate N samples at T>0, save the FULL output text
plus the auto-derived label = is_correct(extracted_answer, gt).

Reference:
  Cobbe et al. 2021 (arXiv:2110.14168)  — original gsm8k ORM data
  V-STaR  (arXiv:2402.06457)              — exact pos+neg pipeline

Usage:
  torchrun --standalone --nproc_per_node=8 scripts/orm_collect_data.py \\
      --model runs/training/v161_t6_ablate/hf_step_336 \\
      --scope_path runs/validation/gsm8k_train_prompts.json \\
      --n_samples 8 \\
      --temperature 0.7 \\
      --gen_length 192 \\
      --out_jsonl runs/validation/orm_data/orm_train.jsonl \\
      --prompt_shard $RANK/$WORLD

Or via wrapper that handles the shard split: scripts/orm_collect_data.sh
"""
from __future__ import annotations
import argparse, json, os, sys, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "validate"))


def extract_answer(s):
    nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", str(s or ""))
    if not nums: return None
    try: return float(nums[-1].replace(",", ""))
    except: return None


def is_correct(out: str, gt: str) -> bool:
    p = extract_answer(out); g = extract_answer(gt)
    if p is None or g is None: return False
    return abs(p - g) < 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model ckpt (e.g. T6 step_336)")
    ap.add_argument("--scope_path", required=True,
                    help="JSON file: list of {prompt, ground_truth}")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--n", type=int, default=0,
                    help="cap prompts (0 = all)")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="sampling temperature; should be >0 for diversity")
    ap.add_argument("--gen_length", type=int, default=192)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--steps", type=int, default=192)
    ap.add_argument("--prompt_batch", default="auto",
                    help="'auto' (probe) or int (force)")
    ap.add_argument("--prompt_shard", default="0/1",
                    help="<idx>/<total> for multi-GPU sharding")
    ap.add_argument("--require_both", action="store_true", default=True,
                    help="only emit prompts that have ≥1 pos AND ≥1 neg")
    args = ap.parse_args()

    SHARD_IDX, SHARD_TOTAL = (int(x) for x in args.prompt_shard.split("/"))

    scope = json.loads(Path(args.scope_path).read_text(encoding="utf-8"))
    if args.n > 0:
        scope = scope[: args.n]
    todo = [(i, r) for i, r in enumerate(scope)
            if i % SHARD_TOTAL == SHARD_IDX]
    print(f"[ORM] shard {SHARD_IDX}/{SHARD_TOTAL}: {len(todo)}/{len(scope)} prompts")

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Append per-shard, dedupe by (idx, sample_idx)

    import torch
    from transformers import AutoModel, AutoTokenizer
    from h1_remask_rescue import generate_batched_multi, _get_mask_token_id

    print(f"[ORM] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).cuda().eval()
    mask_id = _get_mask_token_id(model, tok)

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

    # Resolve P_BATCH
    pb = str(args.prompt_batch).strip().lower()
    if pb in ("auto", "0"):
        # tokenize a sample to estimate worst seq len
        worst = max(len(tok(tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            add_generation_prompt=True, tokenize=False))["input_ids"])
            for _, r in todo[: min(50, len(todo))])
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
                print(f"[ORM] autotune: P_BATCH={P} (B={B}) fits at L={L}")
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
        else:
            P_BATCH = 1
    else:
        P_BATCH = max(1, int(pb))

    # Sort todo by prompt length (descending) — same trick as h3_passN
    def _plen(r):
        text = tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            add_generation_prompt=True, tokenize=False)
        return len(tok(text)["input_ids"])
    todo.sort(key=lambda ir: -_plen(ir[1]))

    def chunked(lst, k):
        for i in range(0, len(lst), k):
            yield lst[i:i+k]

    # Open per-shard tmp file (avoid concurrent write race)
    shard_path = out_path.with_suffix(f".shard{SHARD_IDX}.jsonl")
    n_kept = 0
    n_pos = 0
    n_neg = 0
    n_dropped_no_both = 0

    with shard_path.open("w", encoding="utf-8") as f:
        for chunk in chunked(todo, P_BATCH):
            prompts_text = [r["prompt"] for _, r in chunk]
            outs_per_prompt = generate_batched_multi(
                model, tok, prompts_text,
                n_samples=args.n_samples,
                gen_length=args.gen_length, steps=args.steps,
                block_length=args.block_length,
                temperature=args.temperature,
                mask_id=mask_id,
                _attn_mask_supported=ATTN_OK,
            )
            for (idx, rec), outs in zip(chunk, outs_per_prompt):
                gt = rec["ground_truth"]
                pos = []; neg = []
                for s_i, out_text in enumerate(outs):
                    label = 1 if is_correct(out_text, gt) else 0
                    (pos if label else neg).append((s_i, out_text))
                if args.require_both and (not pos or not neg):
                    n_dropped_no_both += 1
                    continue
                n_kept += 1
                for s_i, out_text in pos:
                    f.write(json.dumps({
                        "idx": idx, "gt": gt,
                        "question": rec["prompt"],
                        "output": out_text,
                        "label": 1, "sample_idx": s_i,
                        "temperature": args.temperature,
                    }, ensure_ascii=False) + "\n")
                    n_pos += 1
                for s_i, out_text in neg:
                    f.write(json.dumps({
                        "idx": idx, "gt": gt,
                        "question": rec["prompt"],
                        "output": out_text,
                        "label": 0, "sample_idx": s_i,
                        "temperature": args.temperature,
                    }, ensure_ascii=False) + "\n")
                    n_neg += 1
            f.flush()

    print(f"[ORM] shard {SHARD_IDX}/{SHARD_TOTAL} done:")
    print(f"      kept {n_kept} prompts, pos={n_pos}, neg={n_neg}, "
          f"dropped(no-both)={n_dropped_no_both}")
    print(f"      → {shard_path}")
    print(f"[ORM] after all shards finish, concatenate to {out_path}:")
    print(f"      cat {out_path.with_suffix('.shard*.jsonl')} > {out_path}")


if __name__ == "__main__":
    main()
