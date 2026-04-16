"""P2.1.e —— Dump full A5 outputs (not just tail[-200:]).

动机:
  A5 只存 `out[-200:]`（`a5_prompt_template.py:137`），
  所以 `finding_broken_by_answer_is_spurious.zh.md` 看到的"Kylar 买眼镜"可能
  只是 baseline 完整输出的**末尾续写垃圾**，前半段模型可能真解对了 prompt 本身。

  本脚本对指定 idx 列表 **重跑 A5 的 4 个 template，保存完整输出**，
  让我们能分段看 head (解题) vs tail (续写)。

用法:
  # 默认：broken-by-answer 5 条
  bash scripts/validate/run_p21e.sh

  # 手动：
  python scripts/validate/p21e_dump_full.py \\
      --only_idx 2,17,22,24,57 \\
      --gen_length 256

  # 强制改长度：如果 gen_length=128 但输出被截断到 Kylar 说明 128 不够。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _http_client import ValidationAPIClient, add_server_arg
from h1_remask_rescue import is_correct
from a5_prompt_template import DEFAULT_TEMPLATES

ROOT = Path(__file__).resolve().parents[2]
SCOPE = ROOT / "runs" / "validation" / "scope_fail_prompts.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only_idx", type=str, default="2,17,22,24,57",
                    help="Comma-separated fail-prompt idx list (from scope_fail_prompts.json order)")
    ap.add_argument("--gen_length", type=int, default=128,
                    help="Match A5 default (128). Increase to 256 if truncation suspected.")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--out", type=str, default=None)
    add_server_arg(ap)
    args = ap.parse_args()

    idxs = [int(x) for x in args.only_idx.split(",") if x.strip()]
    fails = json.loads(SCOPE.read_text(encoding="utf-8"))

    out_path = Path(args.out) if args.out else (
        ROOT / "runs" / "validation" /
        f"p21e_full_outputs_idx{'_'.join(str(i) for i in idxs)}_g{args.gen_length}.json"
    )

    api = ValidationAPIClient(args.server_url)
    api.check_health()

    records = []
    for i in idxs:
        rec = fails[i]
        prompt, gt = rec["prompt"], rec["ground_truth"]
        print(f"\n=== idx={i}  gt={gt} ===")
        print(f"  prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        templates = {}
        for name, suffix in DEFAULT_TEMPLATES:
            p = prompt + suffix
            out = api.generate(
                p, strategy="confidence",
                max_new_tokens=args.gen_length, num_steps=args.steps,
                block_length=args.block_length, temperature=0.0,
            )
            ok = bool(is_correct(out, gt))
            templates[name] = {
                "correct": ok,
                "full_output": out,
                "len": len(out),
                "head_200": out[:200],
                "tail_200": out[-200:],
            }
            print(f"  {name:<10} correct={ok} len={len(out):>4}  head=\"{out[:80]}...\"")
        records.append({
            "idx": i,
            "gt": gt,
            "prompt": prompt,
            "templates": templates,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {
            "config": {
                "gen_length": args.gen_length,
                "steps": args.steps,
                "block_length": args.block_length,
                "only_idx": idxs,
            },
            "records": records,
        },
        ensure_ascii=False,
        indent=2,
    ), encoding="utf-8")
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
