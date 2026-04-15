"""H1：Commit-once-never-revise 是主要瓶颈

对比：
  A. baseline  — 原生 LLaDA 采样 (T=0, low_confidence remasking)
  B. revise    — 原生 + 每 revise_every 步对已 commit 的 conf<revise_thresh token 置回 mask

在 scope_fail_prompts.json 上跑两者，统计 rescue_rate = (B对 ∧ A错) / N

Verdict 阈值：
  rescue ≥ 5%  → SUPPORTED
  rescue ≤ 1%  → REJECTED
  否则          → INCONCLUSIVE

Usage:
    python scripts/validate/h1_remask_rescue.py --n 50 --revise_every 8 --revise_thresh 0.3
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SCOPE = ROOT / "runs" / "validation" / "scope_fail_prompts.json"
OUT_DIR = ROOT / "runs" / "validation"


# ── 采样核心（从 scripts/infer_llada.py 复用 + 加 revise hook） ────────────────

def _get_mask_token_id(model, tokenizer) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "mask_token_id", None) is not None:
        return cfg.mask_token_id
    for c in ("<|mdm_mask|>", "[MASK]", "<mask>"):
        tid = tokenizer.convert_tokens_to_ids(c)
        if tid is not None and tid != tokenizer.unk_token_id:
            return tid
    return tokenizer.mask_token_id


def _add_gumbel(logits, t):
    if t == 0:
        return logits
    return logits + t * torch.distributions.Gumbel(0, 1).sample(logits.shape).to(logits.device)


def _num_transfer(mask_index, steps):
    B = mask_index.shape[0]
    n_mask = mask_index.sum(dim=1)
    base = n_mask // steps
    extra = n_mask % steps
    sched = base.unsqueeze(1).expand(B, steps).clone()
    for b in range(B):
        sched[b, : extra[b]] += 1
    return sched


@torch.no_grad()
def generate(model, tokenizer, prompt, gen_length=128, steps=128, block_length=32,
             temperature=0.0, revise_every=0, revise_thresh=0.3,
             mask_id=None) -> str:
    """若 revise_every > 0 开启 revise hook（把已 commit 的 conf<thresh token 置回 mask）。"""
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    device = next(model.parameters()).device
    mask_id = mask_id if mask_id is not None else _get_mask_token_id(model, tokenizer)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    prompt_len = input_ids.shape[1]
    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    # 记录已 commit 的 confidence（用于 revise hook 判断）
    committed_conf = torch.full_like(x, float("inf"), dtype=torch.float)
    committed_conf[:, :prompt_len] = float("inf")  # prompt 永不 revise

    global_step = 0
    for block_idx in range(num_blocks):
        b_start = prompt_len + block_idx * block_length
        b_end = prompt_len + (block_idx + 1) * block_length
        block_mask = torch.zeros_like(x, dtype=torch.bool)
        block_mask[:, b_start:b_end] = True
        block_masked = x[:, b_start:b_end] == mask_id
        n_xfer = _num_transfer(block_masked, steps_per_block)

        for step in range(steps_per_block):
            mask_index = (x == mask_id)
            logits = model(x).logits
            x0 = _add_gumbel(logits, temperature).argmax(dim=-1)
            p = F.softmax(logits.double(), dim=-1)
            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1).float()
            x0_p = x0_p.masked_fill(~block_mask, -float("inf"))
            x0 = torch.where(mask_index, x0, x)
            conf = torch.where(mask_index, x0_p,
                               torch.full_like(x0_p, -float("inf")))
            n = int(n_xfer[0, step].item())
            if n > 0:
                _, top_idx = torch.topk(conf[0], k=n)
                transfer = torch.zeros_like(x, dtype=torch.bool)
                transfer[0, top_idx] = True
                x[transfer] = x0[transfer]
                # 记录 commit confidence（用真实 prob 不是 gumbel 后的）
                committed_conf[transfer] = torch.gather(
                    F.softmax(logits.float(), -1)[0], -1, x0[0].unsqueeze(-1)
                ).squeeze(-1)[transfer[0]]

            # ── Revise hook（H1 的关键改动）──────────────────────────
            global_step += 1
            if revise_every > 0 and global_step % revise_every == 0:
                # 只在 gen 区域 + 当前或过去 block 找低置信 token
                gen_slice = slice(prompt_len, b_end)  # 不超前
                committed = (x != mask_id)
                committed[:, :prompt_len] = False
                committed[:, b_end:] = False
                low_conf = (committed_conf < revise_thresh) & committed
                if low_conf.any():
                    x[low_conf] = mask_id
                    committed_conf[low_conf] = float("inf")

    return tokenizer.decode(x[0, prompt_len:], skip_special_tokens=True)


# ── 正确性判定 ────────────────────────────────────────────────────────────────

def extract_answer(s: str) -> float | None:
    nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", s or "")
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))  # gsm8k 惯例取最后一个数字
    except Exception:
        return None


def is_correct(output: str, gt: str) -> bool:
    pred = extract_answer(output)
    truth = extract_answer(gt)
    if pred is None or truth is None:
        return False
    return abs(pred - truth) < 1e-4


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/llada-instruct")
    ap.add_argument("--n", type=int, default=50, help="fail 样本数（控时间）")
    ap.add_argument("--revise_every", type=int, default=8)
    ap.add_argument("--revise_thresh", type=float, default=0.3)
    ap.add_argument("--gen_length", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--block_length", type=int, default=32)
    args = ap.parse_args()

    assert SCOPE.exists(), f"先跑 h0_forensics.py 生成 {SCOPE}"
    fails = json.loads(SCOPE.read_text(encoding="utf-8"))[: args.n]
    print(f"[H1] 使用 {len(fails)} 条 fail prompt")

    print(f"[H1] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).cuda().eval()
    mask_id = _get_mask_token_id(model, tok)
    print(f"[H1] mask_id = {mask_id}")

    results = []
    t0 = time.time()
    for i, rec in enumerate(fails):
        prompt = rec["prompt"]
        gt = rec["ground_truth"]

        # A. baseline
        out_base = generate(model, tok, prompt,
                            gen_length=args.gen_length, steps=args.steps,
                            block_length=args.block_length, temperature=0.0,
                            revise_every=0, mask_id=mask_id)
        base_ok = is_correct(out_base, gt)

        # B. with revise hook
        out_rev = generate(model, tok, prompt,
                           gen_length=args.gen_length, steps=args.steps,
                           block_length=args.block_length, temperature=0.0,
                           revise_every=args.revise_every,
                           revise_thresh=args.revise_thresh, mask_id=mask_id)
        rev_ok = is_correct(out_rev, gt)

        results.append({
            "idx": i, "gt": gt, "base_correct": base_ok, "revise_correct": rev_ok,
            "base_out_tail": out_base[-160:], "revise_out_tail": out_rev[-160:],
        })
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(fails)}]  base={int(base_ok)}  revise={int(rev_ok)}   "
              f"elapsed={elapsed:.1f}s  eta={elapsed/(i+1)*(len(fails)-i-1):.0f}s")

    # ── Verdict ───────────────────────────────────────────────────────────────
    N = len(results)
    base_ok = sum(r["base_correct"] for r in results)
    rev_ok = sum(r["revise_correct"] for r in results)
    rescued = sum(1 for r in results if r["revise_correct"] and not r["base_correct"])
    broken = sum(1 for r in results if r["base_correct"] and not r["revise_correct"])
    rescue_rate = rescued / max(N, 1)

    if rescue_rate >= 0.05:
        verdict = "SUPPORTED"
    elif rescue_rate <= 0.01:
        verdict = "REJECTED"
    else:
        verdict = "INCONCLUSIVE"

    summary = {
        "hypothesis": "H1",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args),
        "n": N,
        "base_correct": base_ok,
        "revise_correct": rev_ok,
        "rescued": rescued,
        "broken": broken,
        "rescue_rate": rescue_rate,
        "verdict": verdict,
        "records": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"h1_remask_rescue_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("═" * 60)
    print(f"[H1] N={N}  base={base_ok}  revise={rev_ok}")
    print(f"     rescued (revise✓ ∧ base✗) = {rescued}  → rescue_rate={rescue_rate:.3%}")
    print(f"     broken  (base✓ ∧ revise✗) = {broken}")
    print(f"[H1] Verdict: {verdict}")
    print(f"[H1] saved → {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
