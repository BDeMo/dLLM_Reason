#!/usr/bin/env python
"""BoN evaluation using a trained ORM head.

Two paths:
  (a) Live BoN: re-sample N at T>0, score each with ORM, pick argmax.
  (b) Post-hoc BoN on existing decode_ablate per_prompt — but those
      don't store full output text, so this won't work well unless the
      output region is short.

For now we implement (a). Reuse generate_batched_multi for fast sampling,
then run the ORM forward on each (prompt, sample) pair.

Output: comparison table of greedy / SC@N / BoN@N / pass@N (oracle).

Usage:
  python scripts/orm_eval_bon.py \\
      --base_ckpt runs/training/v161_t6_ablate/hf_step_336 \\
      --orm_head runs/training/orm_v1/head_final.pt \\
      --n_samples 8 --temperature 0.7 \\
      --out_dir runs/validation/orm_eval_v1
"""
from __future__ import annotations
import argparse, json, sys, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "validate"))

import torch
from transformers import AutoModel, AutoTokenizer

from dllm_reason.models.orm_head import ORMWrapper, ORMHead
from h1_remask_rescue import generate_batched_multi, _get_mask_token_id


def extract_answer(s):
    nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", str(s or ""))
    if not nums: return None
    try: return float(nums[-1].replace(",", ""))
    except: return None


def is_correct(out, gt):
    p = extract_answer(out); g = extract_answer(gt)
    if p is None or g is None: return False
    return abs(p - g) < 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--orm_head", required=True,
                    help="head_final.pt from orm_train.py")
    ap.add_argument("--scope_fail", default="runs/validation/scope_fail_prompts.json")
    ap.add_argument("--scope_ok", default="runs/validation/scope_ok_prompts.json")
    ap.add_argument("--n_fail", type=int, default=331)
    ap.add_argument("--n_ok", type=int, default=200)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--gen_length", type=int, default=128)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--prompt_batch", default="auto")
    ap.add_argument("--max_seq_len", type=int, default=768)
    ap.add_argument("--pooling", default="last", choices=["last", "mean"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prompt_shard", default="0/1",
                    help="<idx>/<total> for multi-GPU sharding "
                         "(prompts where i%total==idx are processed)")
    args = ap.parse_args()
    SHARD_IDX, SHARD_TOTAL = (int(x) for x in args.prompt_shard.split("/"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ORM-EVAL] loading base {args.base_ckpt} ...")
    tok = AutoTokenizer.from_pretrained(args.base_ckpt, trust_remote_code=True)
    base = AutoModel.from_pretrained(args.base_ckpt, trust_remote_code=True,
                                     dtype=torch.bfloat16).cuda().eval()
    hidden_size = base.config.hidden_size
    mask_id = _get_mask_token_id(base, tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    print(f"[ORM-EVAL] loading ORM head {args.orm_head} ...")
    head = ORMHead(hidden_size, pooling=args.pooling)
    head.load_state_dict(torch.load(args.orm_head, map_location="cuda"))
    head.cuda().eval()

    # Probe attn_mask
    try:
        with torch.no_grad():
            _x = torch.full((1, 16), mask_id, dtype=torch.long, device="cuda")
            _a = torch.ones((1, 16), dtype=torch.long, device="cuda")
            _ = base(_x, attention_mask=_a).logits
            del _x, _a, _
        ATTN_OK = True
    except TypeError:
        ATTN_OK = False
    torch.cuda.empty_cache()

    # Score function: encode (prompt + output), forward through base,
    # pool last hidden, head.
    @torch.no_grad()
    def orm_score_batch(prompts: list[str], outputs: list[str]) -> list[float]:
        # Build full text and remember prompt boundary per (prompt, output)
        texts = []
        prompt_lens = []
        for p, o in zip(prompts, outputs):
            chat = tok.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True, tokenize=False)
            prompt_lens.append(len(tok(chat, add_special_tokens=False)["input_ids"]))
            texts.append(chat + o)
        enc = tok(texts, padding=True, truncation=True,
                  max_length=args.max_seq_len, return_tensors="pt").to("cuda")
        # build output_mask: 1 in answer region (and non-pad)
        L = enc["input_ids"].shape[1]
        out_mask = torch.zeros_like(enc["attention_mask"])
        for i, pl in enumerate(prompt_lens):
            if pl < L:
                out_mask[i, pl:] = 1
        out_mask = out_mask * enc["attention_mask"]
        if ATTN_OK:
            out = base(enc["input_ids"], attention_mask=enc["attention_mask"],
                       output_hidden_states=True)
        else:
            out = base(enc["input_ids"], output_hidden_states=True)
        hidden = out.hidden_states[-1] if hasattr(out, "hidden_states") and out.hidden_states else out.last_hidden_state
        logits = head(hidden, attention_mask=enc["attention_mask"],
                      output_mask=out_mask).float()
        return logits.tolist()

    # Load scopes
    fail_all = json.loads(Path(args.scope_fail).read_text(encoding="utf-8"))
    ok_all   = json.loads(Path(args.scope_ok).read_text(encoding="utf-8"))
    fail = fail_all[: args.n_fail] if args.n_fail > 0 else fail_all
    ok   = ok_all[: args.n_ok]   if args.n_ok   > 0 else ok_all
    todo_all = [("fail", i, r) for i, r in enumerate(fail)] + \
               [("ok",   i, r) for i, r in enumerate(ok)]
    # shard by global index
    todo = [t for k, t in enumerate(todo_all)
            if k % SHARD_TOTAL == SHARD_IDX]
    print(f"[ORM-EVAL] shard {SHARD_IDX}/{SHARD_TOTAL}: "
          f"fail={len(fail)}  ok={len(ok)}  shard_n={len(todo)}/{len(todo_all)}")

    P_BATCH = 4 if args.prompt_batch == "auto" else int(args.prompt_batch)

    n_correct_greedy = {"fail": 0, "ok": 0}
    n_correct_sc     = {"fail": 0, "ok": 0}
    n_correct_bon    = {"fail": 0, "ok": 0}
    n_correct_pass   = {"fail": 0, "ok": 0}
    n_total          = {"fail": 0, "ok": 0}
    per_prompt_dir = out_dir / "per_prompt"
    per_prompt_dir.mkdir(exist_ok=True)

    for chunk_start in range(0, len(todo), P_BATCH):
        chunk = todo[chunk_start: chunk_start + P_BATCH]
        prompts_text = [r["prompt"] for _, _, r in chunk]
        gts = [r["ground_truth"] for _, _, r in chunk]

        # Sample N outputs per prompt
        outs_per_prompt = generate_batched_multi(
            base, tok, prompts_text,
            n_samples=args.n_samples,
            gen_length=args.gen_length, steps=args.steps,
            block_length=args.block_length, temperature=args.temperature,
            mask_id=mask_id,
            _attn_mask_supported=ATTN_OK,
        )

        # Score each (prompt, output) — flatten and re-batch
        flat_prompts = []
        flat_outputs = []
        for ptext, outs in zip(prompts_text, outs_per_prompt):
            for o in outs:
                flat_prompts.append(ptext)
                flat_outputs.append(o)
        # Score in chunks of 16 to control memory
        scores_flat = []
        for i in range(0, len(flat_outputs), 16):
            sb = orm_score_batch(flat_prompts[i:i+16], flat_outputs[i:i+16])
            scores_flat.extend(sb)
        # Reshape back
        N = args.n_samples
        scores_per_prompt = [scores_flat[i*N:(i+1)*N] for i in range(len(prompts_text))]

        # Per-prompt metrics
        for (group, idx, rec), outs, scores, gt in zip(
                chunk, outs_per_prompt, scores_per_prompt, gts):
            corrects = [is_correct(o, gt) for o in outs]
            answers = [extract_answer(o) for o in outs]
            answers_clean = [a for a in answers if a is not None]
            gt_num = extract_answer(gt)

            # Greedy = sample 0
            n_correct_greedy[group] += int(corrects[0])
            # SC = mode
            if answers_clean:
                mode_a = Counter(answers_clean).most_common(1)[0][0]
                n_correct_sc[group] += int(gt_num is not None
                                           and abs(mode_a - gt_num) < 1e-4)
            # BoN = argmax score
            best_i = max(range(N), key=lambda j: scores[j])
            n_correct_bon[group] += int(corrects[best_i])
            # pass@N = any correct
            n_correct_pass[group] += int(any(corrects))

            n_total[group] += 1

            # Persist
            (per_prompt_dir / f"{group}_{idx:04d}.json").write_text(
                json.dumps({
                    "group": group, "idx": idx, "gt": gt,
                    "outputs": outs,
                    "answers": answers,
                    "corrects": [bool(c) for c in corrects],
                    "scores": scores,
                    "best_i_bon": best_i,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(f"[ORM-EVAL] chunk {chunk_start//P_BATCH+1}/"
              f"{(len(todo)+P_BATCH-1)//P_BATCH} done")

    # Per-shard partial summary (final aggregation happens in pipeline)
    shard_summary = {
        "config": vars(args),
        "shard": f"{SHARD_IDX}/{SHARD_TOTAL}",
        "fail": {
            "n":      n_total["fail"],
            "greedy": n_correct_greedy["fail"],
            "SC@N":   n_correct_sc["fail"],
            "BoN@N":  n_correct_bon["fail"],
            "pass@N": n_correct_pass["fail"],
        },
        "ok": {
            "n":      n_total["ok"],
            "greedy": n_correct_greedy["ok"],
            "SC@N":   n_correct_sc["ok"],
            "BoN@N":  n_correct_bon["ok"],
            "pass@N": n_correct_pass["ok"],
        },
    }
    (out_dir / f"summary_shard{SHARD_IDX}.json").write_text(
        json.dumps(shard_summary, indent=2, ensure_ascii=False))
    print(f"[ORM-EVAL] shard {SHARD_IDX}/{SHARD_TOTAL} done → "
          f"{out_dir}/summary_shard{SHARD_IDX}.json")


if __name__ == "__main__":
    main()
