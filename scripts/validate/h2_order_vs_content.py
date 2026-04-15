"""H2：T=0 + 双向 attention 让 unmask order 近乎无关

断言：固定 prompt 下，改顺序产生的 output 方差 ≪ 改内容采样产生的方差。

两组采样（每条 prompt 都跑）：
  content axis: remasking="low_confidence"  + T ∈ {0.0, 0.3, 0.7}, 每 T 采 3 次  → 9 outputs
  order   axis: T=0.0                        + remasking ∈ {low_confidence, random}
                                              + block_length ∈ {16, 32, 64}       → 6 outputs

方差度量：
  pairwise normalized edit distance (char-level) 的 mean
  ratio = mean_dist(order) / mean_dist(content)

Verdict 阈值：
  ratio < 0.3  → SUPPORTED（order axis 信号弱，H2 成立）
  ratio > 0.7  → REJECTED
  否则          → INCONCLUSIVE

Usage:
    python scripts/validate/h2_order_vs_content.py --n 20
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

# 复用 h1 的 generate（它本身支持 T + block_length；remasking 策略通过显式传参扩展）
import sys
sys.path.insert(0, str(Path(__file__).parent))
from h1_remask_rescue import generate, _get_mask_token_id

ROOT = Path(__file__).resolve().parents[2]
SCOPE = ROOT / "runs" / "validation" / "scope_fail_prompts.json"
OUT_DIR = ROOT / "runs" / "validation"


def norm_edit(a: str, b: str) -> float:
    """0 相同，1 完全不同。用 SequenceMatcher 的 ratio 转换。"""
    if not a and not b:
        return 0.0
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def mean_pairwise(outs: list[str]) -> float:
    if len(outs) < 2:
        return 0.0
    ds = [norm_edit(a, b) for a, b in combinations(outs, 2)]
    return sum(ds) / len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/llada-instruct")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--gen_length", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    args = ap.parse_args()

    assert SCOPE.exists(), "先跑 h0_forensics.py"
    prompts = [r["prompt"] for r in json.loads(SCOPE.read_text(encoding="utf-8"))[: args.n]]

    print(f"[H2] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).cuda().eval()
    mask_id = _get_mask_token_id(model, tok)

    # content axis：3 个温度 × 3 次重采样
    content_temps = [0.0, 0.3, 0.7]
    content_reps = 3
    # order axis：T=0，不同 block_length（不同的"顺序"）
    order_blocks = [16, 32, 64]

    per_prompt = []
    t0 = time.time()
    for pi, prompt in enumerate(prompts):
        content_outs = []
        for T in content_temps:
            for _ in range(content_reps):
                out = generate(model, tok, prompt,
                               gen_length=args.gen_length, steps=args.steps,
                               block_length=32, temperature=T,
                               revise_every=0, mask_id=mask_id)
                content_outs.append(out[-300:])  # 取末尾 300 字符做 diff，避免对齐噪声
        order_outs = []
        for blk in order_blocks:
            out = generate(model, tok, prompt,
                           gen_length=args.gen_length, steps=args.steps,
                           block_length=blk, temperature=0.0,
                           revise_every=0, mask_id=mask_id)
            order_outs.append(out[-300:])

        cvar = mean_pairwise(content_outs)
        ovar = mean_pairwise(order_outs)
        ratio = ovar / cvar if cvar > 1e-6 else float("nan")
        per_prompt.append({
            "idx": pi, "content_var": cvar, "order_var": ovar, "ratio": ratio
        })
        elapsed = time.time() - t0
        print(f"  [{pi+1}/{len(prompts)}]  content_var={cvar:.3f}  order_var={ovar:.3f}  "
              f"ratio={ratio:.3f}  eta={elapsed/(pi+1)*(len(prompts)-pi-1):.0f}s")

    # 汇总
    import statistics as stats
    ratios = [r["ratio"] for r in per_prompt if r["ratio"] == r["ratio"]]
    mean_ratio = stats.mean(ratios) if ratios else float("nan")
    mean_cvar = stats.mean(r["content_var"] for r in per_prompt)
    mean_ovar = stats.mean(r["order_var"] for r in per_prompt)

    if mean_ratio < 0.3:
        verdict = "SUPPORTED"
    elif mean_ratio > 0.7:
        verdict = "REJECTED"
    else:
        verdict = "INCONCLUSIVE"

    summary = {
        "hypothesis": "H2",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args),
        "n_prompts": len(prompts),
        "mean_content_var": mean_cvar,
        "mean_order_var": mean_ovar,
        "mean_ratio": mean_ratio,
        "verdict": verdict,
        "per_prompt": per_prompt,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"h2_order_vs_content_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("═" * 60)
    print(f"[H2] mean content_var = {mean_cvar:.3f}")
    print(f"     mean order_var   = {mean_ovar:.3f}")
    print(f"     ratio            = {mean_ratio:.3f}")
    print(f"[H2] Verdict: {verdict}")
    print(f"[H2] saved → {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
