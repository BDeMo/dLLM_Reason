"""A3：Span-level revise hook — 比 H1 的 single-token revise 更粗粒度

H1 REJECTED 的诊断显示 122/137 的 prompt，token-level conf 全 ≥ 0.3，revise hook 从未触发。
A3 假设：错误更可能出现在**连续几个 token 组成的 span**（比如一个算式片段），
单 token conf 可能高，但 span 平均 conf 低 — 用 window 检测更敏感。

做法：
  revise 判据改为"以位置 i 为中心的 window_size 窗口内已 commit token 的平均 conf < thresh"
  触发后把整个 window 置回 mask（不只是中心 token）

Verdict 阈值：
  rescue_rate ≥ 5%  → SUPPORTED
  rescue_rate ≤ 1%  → REJECTED
  否则              → INCONCLUSIVE

Inference via FastAPI server (scripts/serve.py, 默认 http://localhost:8000):
  baseline = POST /generate                 strategy=confidence
  revise   = POST /generate_span_revise

Usage:
    # 确保 server 已启动:
    # python scripts/serve.py --model_id checkpoints/llada-instruct

    python scripts/validate/a3_span_revise.py --n 2 --dry_run
    python scripts/validate/a3_span_revise.py --n 137
    python scripts/validate/a3_span_revise.py --n 137 --server_url http://1.2.3.4:8000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _runlib import RunDir, ProgressPrinter, add_common_args, resolve_run_dir
from _http_client import ValidationAPIClient, add_server_arg

sys.path.insert(0, str(Path(__file__).parent))
from h1_remask_rescue import is_correct, compute_verdict

ROOT = Path(__file__).resolve().parents[2]
SCOPE = ROOT / "runs" / "validation" / "scope_fail_prompts.json"
OUT_BASE = ROOT / "runs" / "validation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--revise_every", type=int, default=8)
    ap.add_argument("--revise_thresh", type=float, default=0.4)
    ap.add_argument("--window_size", type=int, default=4)
    ap.add_argument("--gen_length", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--block_length", type=int, default=32)
    add_common_args(ap)
    add_server_arg(ap)
    args = ap.parse_args()

    assert SCOPE.exists(), f"先跑 h0_forensics.py 生成 {SCOPE}"
    fails = json.loads(SCOPE.read_text(encoding="utf-8"))[: args.n]
    print(f"[A3] 使用 {len(fails)} 条 fail prompt")

    run_dir = resolve_run_dir(args, "a3_span_revise", OUT_BASE)
    rd = RunDir(run_dir, "A3", config=vars(args), resume=args.resume)
    print(f"[A3] run_dir = {rd.dir}")

    done_before = sum(1 for i in range(len(fails)) if rd.has_prompt(i))
    todo = [i for i in range(len(fails)) if not rd.has_prompt(i)]
    print(f"[A3] done_before={done_before}  todo={len(todo)}")

    if args.dry_run:
        print("[A3] DRY RUN — 不连 server")
        print(f"     {len(todo)} 条 × 2 gen/条 = {len(todo)*2} 次 HTTP call")
        print(f"     server_url = {args.server_url}")
        print(f"     window_size={args.window_size}  thresh={args.revise_thresh}  "
              f"every={args.revise_every}")
        return

    api = ValidationAPIClient(args.server_url)
    api.check_health()
    print(f"[A3] window_size={args.window_size}  thresh={args.revise_thresh}  "
          f"every={args.revise_every}")

    pp = ProgressPrinter(len(todo), tag="A3 ")
    for i in todo:
        rec = fails[i]
        prompt, gt = rec["prompt"], rec["ground_truth"]

        out_base = api.generate(
            prompt, strategy="confidence",
            max_new_tokens=args.gen_length, num_steps=args.steps,
            block_length=args.block_length, temperature=0.0,
        )
        base_ok = is_correct(out_base, gt)

        out_span = api.generate_span_revise(
            prompt,
            gen_length=args.gen_length, steps=args.steps,
            block_length=args.block_length, temperature=0.0,
            revise_every=args.revise_every, revise_thresh=args.revise_thresh,
            window_size=args.window_size,
        )
        span_ok = is_correct(out_span, gt)

        record = {
            "idx": i, "gt": gt,
            "base_correct": bool(base_ok), "revise_correct": bool(span_ok),
            "base_out_tail": out_base[-200:],
            "revise_out_tail": out_span[-200:],
        }
        rd.save_prompt(i, record)
        pp.tick(f"base={int(base_ok)} span={int(span_ok)}")

    all_recs = rd.load_all_prompts()
    verdict = compute_verdict(all_recs)
    rd.write_summary({**verdict, "config": rd.config})

    print()
    print("═" * 60)
    print(f"[A3] N={verdict['n']}  base={verdict['base_correct']}  "
          f"span={verdict['revise_correct']}")
    print(f"     rescued={verdict['rescued']}  broken={verdict['broken']}")
    print(f"     rescue_rate={verdict['rescue_rate']:.3%}")
    print(f"[A3] Verdict: {verdict['verdict']}")
    print(f"[A3] summary → {rd.summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
