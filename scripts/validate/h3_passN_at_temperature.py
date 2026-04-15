"""H3：LLaDA-instruct 在 gsm8k 的 137 条 init_fail 上达到能力上限

断言：即使加温度 + 多次重采样，这些 prompt 的 pass@N 依然 ≈ 0。

做法：
  fail 集取 K 条 (默认 30) + 对照组 K 条 (init_ok)
  每条 × T ∈ {0.3, 0.7, 1.0} × N=8 次
  算 pass@1 / pass@4 / pass@8（at least one correct）

Verdict 阈值：
  fail_pass@8 < 5% 且 ok_pass@8 > 90%  → SUPPORTED（能力上限）
  fail_pass@8 > 20%                      → REJECTED（采样 diversity 能救）
  否则                                    → INCONCLUSIVE

Usage:
    python scripts/validate/h3_passN_at_temperature.py --n_fail 30 --n_ok 30 --n_samples 8
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent))
from h1_remask_rescue import generate, _get_mask_token_id, is_correct

ROOT = Path(__file__).resolve().parents[2]
SCOPE = ROOT / "runs" / "validation" / "scope_fail_prompts.json"
DB = ROOT / "runs" / "research_20260411_030422" / "stage2_discovery" / "episodes.db"
OUT_DIR = ROOT / "runs" / "validation"


def load_init_ok(k: int):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT prompt, ground_truth FROM episodes WHERE correct=1 LIMIT ?", (k,))
    rows = cur.fetchall()
    conn.close()
    return [{"prompt": p, "ground_truth": gt} for p, gt in rows]


def pass_at_k(corrects: list[bool], k: int) -> float:
    return 1.0 if any(corrects[:k]) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/llada-instruct")
    ap.add_argument("--n_fail", type=int, default=30)
    ap.add_argument("--n_ok", type=int, default=30, help="对照组 prompt 数（init_ok）")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--gen_length", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--temps", type=float, nargs="+", default=[0.3, 0.7, 1.0])
    args = ap.parse_args()

    assert SCOPE.exists(), "先跑 h0_forensics.py"
    fail_prompts = json.loads(SCOPE.read_text(encoding="utf-8"))[: args.n_fail]
    ok_prompts = load_init_ok(args.n_ok)
    print(f"[H3] fail={len(fail_prompts)}  ok={len(ok_prompts)}  temps={args.temps}  N={args.n_samples}")

    print(f"[H3] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).cuda().eval()
    mask_id = _get_mask_token_id(model, tok)

    def eval_group(prompts, tag):
        """对一个 prompt 组跑全部温度 × N 次，返回 per-temp pass@k 汇总。"""
        stats = {T: {"pass@1": 0, "pass@4": 0, "pass@8": 0, "n": 0} for T in args.temps}
        per_prompt = []
        t0 = time.time()
        total = len(prompts) * len(args.temps) * args.n_samples
        done = 0
        for pi, rec in enumerate(prompts):
            prompt, gt = rec["prompt"], rec["ground_truth"]
            row = {"idx": pi, "gt": gt, "temps": {}}
            for T in args.temps:
                corrects = []
                for _ in range(args.n_samples):
                    out = generate(model, tok, prompt,
                                   gen_length=args.gen_length, steps=args.steps,
                                   block_length=args.block_length, temperature=T,
                                   revise_every=0, mask_id=mask_id)
                    corrects.append(is_correct(out, gt))
                    done += 1
                p1 = pass_at_k(corrects, 1)
                p4 = pass_at_k(corrects, min(4, args.n_samples))
                p8 = pass_at_k(corrects, args.n_samples)
                stats[T]["pass@1"] += p1
                stats[T]["pass@4"] += p4
                stats[T]["pass@8"] += p8
                stats[T]["n"] += 1
                row["temps"][str(T)] = {"correct_list": corrects,
                                         "pass@1": p1, "pass@4": p4, "pass@8": p8}
                elapsed = time.time() - t0
                eta = elapsed / max(done, 1) * (total - done)
                print(f"  [{tag} {pi+1}/{len(prompts)} T={T}] "
                      f"p@1={p1} p@4={p4} p@8={p8}  eta={eta:.0f}s")
            per_prompt.append(row)

        # 标准化
        for T in stats:
            n = max(stats[T]["n"], 1)
            for k in ("pass@1", "pass@4", "pass@8"):
                stats[T][k] /= n
        return stats, per_prompt

    fail_stats, fail_detail = eval_group(fail_prompts, "FAIL")
    ok_stats, ok_detail = eval_group(ok_prompts, "OK  ")

    # ── Verdict ───────────────────────────────────────────────────────────────
    fail_p8 = max(fail_stats[T]["pass@8"] for T in args.temps)  # 任意 T 下最高
    ok_p8 = max(ok_stats[T]["pass@8"] for T in args.temps)

    if fail_p8 < 0.05 and ok_p8 > 0.90:
        verdict = "SUPPORTED"
    elif fail_p8 > 0.20:
        verdict = "REJECTED"
    else:
        verdict = "INCONCLUSIVE"

    summary = {
        "hypothesis": "H3",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args),
        "fail_stats": fail_stats,
        "ok_stats": ok_stats,
        "fail_pass@8_max": fail_p8,
        "ok_pass@8_max": ok_p8,
        "verdict": verdict,
        "fail_detail": fail_detail,
        "ok_detail": ok_detail,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"h3_passN_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("═" * 60)
    print("[H3] fail 集 pass 汇总:")
    for T, s in fail_stats.items():
        print(f"     T={T}  p@1={s['pass@1']:.2%}  p@4={s['pass@4']:.2%}  p@8={s['pass@8']:.2%}")
    print("[H3] ok   集 pass 汇总:")
    for T, s in ok_stats.items():
        print(f"     T={T}  p@1={s['pass@1']:.2%}  p@4={s['pass@4']:.2%}  p@8={s['pass@8']:.2%}")
    print(f"[H3] fail_p@8_max = {fail_p8:.2%}   ok_p@8_max = {ok_p8:.2%}")
    print(f"[H3] Verdict: {verdict}")
    print(f"[H3] saved → {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
