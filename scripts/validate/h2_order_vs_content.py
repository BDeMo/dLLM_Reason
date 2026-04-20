"""H2：T=0 + 双向 attention 让 unmask order 近乎无关

断言：固定 prompt 下，改顺序产生的 output 方差 ≪ 改内容采样产生的方差。

两组采样（每条 prompt 都跑）：
  content axis: T ∈ {0.0, 0.3, 0.7}, 每 T 采 3 次  → 9 outputs
  order   axis: T=0.0 + block_length ∈ {16, 32, 64}       → 3 outputs

方差度量：
  pairwise normalized edit distance (char-level) 的 mean
  ratio = mean_dist(order) / mean_dist(content)

Verdict 阈值：
  ratio < 0.3  → SUPPORTED（order axis 信号弱，H2 成立）
  ratio > 0.7  → REJECTED
  否则          → INCONCLUSIVE

Usage:
    python scripts/validate/h2_order_vs_content.py --n 2 --dry_run
    python scripts/validate/h2_order_vs_content.py --n 20
    python scripts/validate/h2_order_vs_content.py --n 20 --resume \\
        --run_dir runs/validation/h2_order_content_20260415_083000
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

from _runlib import RunDir, ProgressPrinter, add_common_args, resolve_run_dir

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[2]
SCOPE = ROOT / "runs" / "validation" / "scope_fail_prompts.json"
OUT_BASE = ROOT / "runs" / "validation"


def norm_edit(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def mean_pairwise(outs: list[str]) -> float:
    if len(outs) < 2:
        return 0.0
    ds = [norm_edit(a, b) for a, b in combinations(outs, 2)]
    return sum(ds) / len(ds)


def compute_verdict(records: list[dict]) -> dict:
    ratios = [r["ratio"] for r in records if r["ratio"] == r["ratio"]]
    mean_ratio = stats.mean(ratios) if ratios else float("nan")
    mean_cvar = stats.mean(r["content_var"] for r in records) if records else 0.0
    mean_ovar = stats.mean(r["order_var"] for r in records) if records else 0.0

    if ratios and mean_ratio < 0.3:
        verdict = "SUPPORTED"
    elif ratios and mean_ratio > 0.7:
        verdict = "REJECTED"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "n": len(records),
        "mean_content_var": mean_cvar,
        "mean_order_var": mean_ovar,
        "mean_ratio": mean_ratio,
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/llada-instruct")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--gen_length", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--content_temps", type=float, nargs="+", default=[0.0, 0.3, 0.7])
    ap.add_argument("--content_reps", type=int, default=3)
    ap.add_argument("--order_blocks", type=int, nargs="+", default=[16, 32, 64])
    add_common_args(ap)
    args = ap.parse_args()

    assert SCOPE.exists(), f"先跑 h0_forensics.py 生成 {SCOPE}"
    prompts = [r["prompt"] for r in json.loads(SCOPE.read_text(encoding="utf-8"))[: args.n]]
    print(f"[H2] 使用 {len(prompts)} 条 prompt")

    run_dir = resolve_run_dir(args, "h2_order_content", OUT_BASE)
    rd = RunDir(run_dir, "H2", config=vars(args), resume=args.resume)
    print(f"[H2] run_dir = {rd.dir}")

    done_before = sum(1 for i in range(len(prompts)) if rd.has_prompt(i))
    todo = [i for i in range(len(prompts)) if not rd.has_prompt(i)]
    print(f"[H2] done_before={done_before}  todo={len(todo)}")

    if args.dry_run:
        n_content = len(args.content_temps) * args.content_reps
        n_order = len(args.order_blocks)
        print("[H2] DRY RUN — 不加载模型")
        print(f"     每条 prompt: content={n_content} outputs, order={n_order} outputs")
        print(f"     会跑 {len(todo)} 条，共 {len(todo)*(n_content+n_order)} 次 generate")
        print(f"     config 已写入 {rd.config_path}")
        return

    # Lazy import
    import torch
    from transformers import AutoModel, AutoTokenizer
    from h1_remask_rescue import generate, _get_mask_token_id

    print(f"[H2] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).cuda().eval()
    mask_id = _get_mask_token_id(model, tok)
    print(f"[H2] mask_id = {mask_id}")

    pp = ProgressPrinter(len(todo), tag="H2 ")
    for i in todo:
        prompt = prompts[i]
        content_outs = []
        for T in args.content_temps:
            for _ in range(args.content_reps):
                out = generate(model, tok, prompt,
                               gen_length=args.gen_length, steps=args.steps,
                               block_length=32, temperature=T,
                               revise_every=0, mask_id=mask_id)
                content_outs.append(out[-300:])
        order_outs = []
        for blk in args.order_blocks:
            out = generate(model, tok, prompt,
                           gen_length=args.gen_length, steps=args.steps,
                           block_length=blk, temperature=0.0,
                           revise_every=0, mask_id=mask_id)
            order_outs.append(out[-300:])

        cvar = mean_pairwise(content_outs)
        ovar = mean_pairwise(order_outs)
        ratio = ovar / cvar if cvar > 1e-6 else float("nan")

        record = {
            "idx": i,
            "content_var": cvar,
            "order_var": ovar,
            "ratio": ratio,
            "content_outs_tail": [o[-120:] for o in content_outs],
            "order_outs_tail": [o[-120:] for o in order_outs],
        }
        rd.save_prompt(i, record)
        pp.tick(f"cvar={cvar:.3f} ovar={ovar:.3f} r={ratio:.3f}")

    all_recs = rd.load_all_prompts()
    verdict = compute_verdict(all_recs)
    rd.write_summary({**verdict, "config": rd.config})

    print()
    print("═" * 60)
    print(f"[H2] N={verdict['n']}")
    print(f"     mean content_var = {verdict['mean_content_var']:.3f}")
    print(f"     mean order_var   = {verdict['mean_order_var']:.3f}")
    print(f"     ratio            = {verdict['mean_ratio']:.3f}")
    print(f"[H2] Verdict: {verdict['verdict']}")
    print(f"[H2] summary → {rd.summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
