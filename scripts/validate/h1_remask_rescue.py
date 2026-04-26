"""H1：Commit-once-never-revise 是主要瓶颈

对比：
  A. baseline  — 原生 LLaDA 采样 (T=0, low_confidence remasking)
  B. revise    — 原生 + 每 revise_every 步对已 commit 的 conf<revise_thresh token 置回 mask

在 scope_fail_prompts.json 上跑两者，统计 rescue_rate = (B对 ∧ A错) / N

Verdict 阈值：
  rescue ≥ 5%  → SUPPORTED
  rescue ≤ 1%  → REJECTED
  否则          → INCONCLUSIVE

Run dir 结构（见 _runlib.py）：
    runs/validation/h1_remask/{config,per_prompt/,progress.jsonl,summary}.json

Usage:
    # 本地 dry-run
    python scripts/validate/h1_remask_rescue.py --n 2 --dry_run

    # 服务器首次
    python scripts/validate/h1_remask_rescue.py --n 137

    # 断点 resume
    python scripts/validate/h1_remask_rescue.py --n 137 --resume \\
        --run_dir runs/validation/h1_remask_20260415_083000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from _runlib import RunDir, ProgressPrinter, add_common_args, resolve_run_dir

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[2]
SCOPE = ROOT / "runs" / "validation" / "scope_fail_prompts.json"
OUT_BASE = ROOT / "runs" / "validation"


# ── Lazy 加载（dry_run 时不引入 torch）───────────────────────────────────────

def _load_model(model_id: str):
    import torch
    from transformers import AutoModel, AutoTokenizer
    print(f"[H1] loading {model_id} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).cuda().eval()
    return model, tok


def _get_mask_token_id(model, tokenizer) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "mask_token_id", None) is not None:
        return cfg.mask_token_id
    for c in ("<|mdm_mask|>", "[MASK]", "<mask>"):
        tid = tokenizer.convert_tokens_to_ids(c)
        if tid is not None and tid != tokenizer.unk_token_id:
            return tid
    return tokenizer.mask_token_id


# ── 采样核心（带 revise hook）─────────────────────────────────────────────────

def generate(model, tokenizer, prompt, gen_length=128, steps=128, block_length=32,
             temperature=0.0, revise_every=0, revise_thresh=0.3, mask_id=None) -> str:
    """若 revise_every > 0 启用 revise hook（把已 commit 的 conf<thresh token 置回 mask）。"""
    import torch
    import torch.nn.functional as F

    def _add_gumbel(logits, t):
        if t == 0:
            return logits
        return logits + t * torch.distributions.Gumbel(0, 1).sample(logits.shape).to(logits.device)

    def _num_transfer(mask_index, steps_):
        B = mask_index.shape[0]
        n_mask = mask_index.sum(dim=1)
        base = n_mask // steps_
        extra = n_mask % steps_
        sched = base.unsqueeze(1).expand(B, steps_).clone()
        for b in range(B):
            sched[b, : extra[b]] += 1
        return sched

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
    with torch.no_grad():
        x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
        x[:, :prompt_len] = input_ids
        committed_conf = torch.full_like(x, float("inf"), dtype=torch.float)
        committed_conf[:, :prompt_len] = float("inf")

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
                    committed_conf[transfer] = torch.gather(
                        F.softmax(logits.float(), -1)[0], -1, x0[0].unsqueeze(-1)
                    ).squeeze(-1)[transfer[0]]

                global_step += 1
                if revise_every > 0 and global_step % revise_every == 0:
                    committed = (x != mask_id)
                    committed[:, :prompt_len] = False
                    committed[:, b_end:] = False
                    low_conf = (committed_conf < revise_thresh) & committed
                    if low_conf.any():
                        x[low_conf] = mask_id
                        committed_conf[low_conf] = float("inf")

        return tokenizer.decode(x[0, prompt_len:], skip_special_tokens=True)


# ── Batched N-sample variant (8B bf16 underutilizes 1 sample on A100) ───────
# Same algorithm as generate() above, but B=N independent samples for one
# prompt. All N rows share the prompt; per-row Gumbel + per-row topk make
# each row a distinct sample path. Returns list[str] of N decoded outputs.
#
# Expected speedup vs N serial generate() calls: 5-8× on A100-80GB at N=8
# (limited by memory + matmul scaling). For N=16 may need to stage in
# halves if seq_len > 320.
def generate_batched(model, tokenizer, prompt, *, n_samples: int,
                     gen_length=128, steps=128, block_length=32,
                     temperature=0.0, mask_id=None) -> list[str]:
    import torch
    import torch.nn.functional as F

    def _add_gumbel(logits, t):
        if t == 0:
            return logits
        return logits + t * torch.distributions.Gumbel(0, 1).sample(logits.shape).to(logits.device)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    device = next(model.parameters()).device
    mask_id = mask_id if mask_id is not None else _get_mask_token_id(model, tokenizer)
    B = n_samples

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    prompt_len = input_ids.shape[1]
    L = prompt_len + gen_length

    with torch.no_grad():
        x = torch.full((B, L), mask_id, dtype=torch.long, device=device)
        x[:, :prompt_len] = input_ids                          # broadcast same prompt to all B

        for block_idx in range(num_blocks):
            b_start = prompt_len + block_idx * block_length
            b_end = prompt_len + (block_idx + 1) * block_length
            block_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
            block_mask[:, b_start:b_end] = True
            # All rows have same masked count in this block (same prompt)
            n_per_step = block_length // steps_per_block
            extra = block_length % steps_per_block

            for step in range(steps_per_block):
                mask_index = (x == mask_id)                    # (B, L)
                logits = model(x).logits                       # (B, L, V) — the bottleneck step
                x0 = _add_gumbel(logits, temperature).argmax(dim=-1)  # (B, L)
                p = F.softmax(logits.double(), dim=-1)
                x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1).float()
                x0_p = x0_p.masked_fill(~block_mask, -float("inf"))
                x0 = torch.where(mask_index, x0, x)
                conf = torch.where(mask_index, x0_p,
                                   torch.full_like(x0_p, -float("inf")))
                # Per-row topk: each sample diverges based on its own Gumbel
                n = n_per_step + (1 if step < extra else 0)
                if n > 0:
                    _, top_idx = torch.topk(conf, k=n, dim=-1)  # (B, n)
                    transfer = torch.zeros((B, L), dtype=torch.bool, device=device)
                    transfer.scatter_(1, top_idx, True)
                    x = torch.where(transfer, x0, x)

        outs = [tokenizer.decode(x[i, prompt_len:], skip_special_tokens=True)
                for i in range(B)]
        return outs


# ── 正确性判定 ────────────────────────────────────────────────────────────────

def extract_answer(s: str) -> float | None:
    nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", s or "")
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except Exception:
        return None


def is_correct(output: str, gt: str) -> bool:
    pred = extract_answer(output)
    truth = extract_answer(gt)
    if pred is None or truth is None:
        return False
    return abs(pred - truth) < 1e-4


# ── Verdict 计算（可独立调用于 resume-incomplete 的 run）───────────────────────

def compute_verdict(records: list[dict]) -> dict:
    N = len(records)
    base_ok = sum(r["base_correct"] for r in records)
    rev_ok = sum(r["revise_correct"] for r in records)
    rescued = sum(1 for r in records if r["revise_correct"] and not r["base_correct"])
    broken = sum(1 for r in records if r["base_correct"] and not r["revise_correct"])
    rescue_rate = rescued / max(N, 1)

    if rescue_rate >= 0.05:
        verdict = "SUPPORTED"
    elif rescue_rate <= 0.01:
        verdict = "REJECTED"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "n": N,
        "base_correct": base_ok,
        "revise_correct": rev_ok,
        "rescued": rescued,
        "broken": broken,
        "rescue_rate": rescue_rate,
        "verdict": verdict,
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/llada-instruct")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--revise_every", type=int, default=8)
    ap.add_argument("--revise_thresh", type=float, default=0.3)
    ap.add_argument("--gen_length", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--block_length", type=int, default=32)
    add_common_args(ap)
    args = ap.parse_args()

    # ── Scope ─────────────────────────────────────────────────────────────────
    assert SCOPE.exists(), f"先跑 h0_forensics.py 生成 {SCOPE}"
    fails = json.loads(SCOPE.read_text(encoding="utf-8"))[: args.n]
    print(f"[H1] 使用 {len(fails)} 条 fail prompt")

    # ── Run dir ───────────────────────────────────────────────────────────────
    run_dir = resolve_run_dir(args, "h1_remask", OUT_BASE)
    rd = RunDir(run_dir, "H1", config=vars(args), resume=args.resume)
    print(f"[H1] run_dir = {rd.dir}")

    done_before = sum(1 for i in range(len(fails)) if rd.has_prompt(i))
    todo = [i for i in range(len(fails)) if not rd.has_prompt(i)]
    print(f"[H1] done_before={done_before}  todo={len(todo)}")

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print("[H1] DRY RUN — 不加载模型")
        print(f"     会跑 {len(todo)} 条，结果保存到 {rd.dir}/per_prompt/XXXX.json")
        print(f"     config 已写入 {rd.config_path}")
        return

    # ── Load model（耗时）─────────────────────────────────────────────────────
    model, tok = _load_model(args.model)
    mask_id = _get_mask_token_id(model, tok)
    print(f"[H1] mask_id = {mask_id}")

    # ── 主循环 ────────────────────────────────────────────────────────────────
    pp = ProgressPrinter(len(todo), tag="H1 ")
    for i in todo:
        rec = fails[i]
        prompt, gt = rec["prompt"], rec["ground_truth"]

        out_base = generate(model, tok, prompt,
                            gen_length=args.gen_length, steps=args.steps,
                            block_length=args.block_length, temperature=0.0,
                            revise_every=0, mask_id=mask_id)
        base_ok = is_correct(out_base, gt)

        out_rev = generate(model, tok, prompt,
                           gen_length=args.gen_length, steps=args.steps,
                           block_length=args.block_length, temperature=0.0,
                           revise_every=args.revise_every,
                           revise_thresh=args.revise_thresh, mask_id=mask_id)
        rev_ok = is_correct(out_rev, gt)

        record = {
            "idx": i, "gt": gt,
            "base_correct": bool(base_ok), "revise_correct": bool(rev_ok),
            "base_out_tail": out_base[-200:],
            "revise_out_tail": out_rev[-200:],
        }
        rd.save_prompt(i, record)
        pp.tick(f"base={int(base_ok)} rev={int(rev_ok)}")

    # ── Aggregate + summary ───────────────────────────────────────────────────
    all_recs = rd.load_all_prompts()
    verdict = compute_verdict(all_recs)
    rd.write_summary({**verdict, "config": rd.config})

    print()
    print("═" * 60)
    print(f"[H1] N={verdict['n']}  base={verdict['base_correct']}  "
          f"revise={verdict['revise_correct']}")
    print(f"     rescued={verdict['rescued']}  broken={verdict['broken']}")
    print(f"     rescue_rate={verdict['rescue_rate']:.3%}")
    print(f"[H1] Verdict: {verdict['verdict']}")
    print(f"[H1] summary → {rd.summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
