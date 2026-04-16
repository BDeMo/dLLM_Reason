"""A4：Block-layout rerank — 换 block_length / 非均匀 layout，看 pass@any_layout

A1/A2 证明 edge-level / single-token 级干预无效；A3 在测 span-level。
A4 再粗一级：**换整个 block 切分方式**，看是否有 layout 能救当前 fail。

对每条 fail prompt 跑多个 layout：
  - uniform block_length ∈ {8, 16, 32, 64}  (走 /generate, strategy=confidence)
  - 1 个 non-uniform layout：前 64 tokens 用 block=16，后 64 tokens 用 block=64
    （走 /generate_block_schedule）

Verdict 阈值（同 H1）：
  rescue_rate ≥ 5%  → SUPPORTED
  rescue_rate ≤ 1%  → REJECTED
  否则              → INCONCLUSIVE

Usage:
    python scripts/validate/a4_block_rerank.py --n 2 --dry_run
    python scripts/validate/a4_block_rerank.py --n 137
    python scripts/validate/a4_block_rerank.py --n 137 --server_url http://1.2.3.4:8000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _runlib import RunDir, ProgressPrinter, add_common_args, resolve_run_dir
from _http_client import ValidationAPIClient, add_server_arg

sys.path.insert(0, str(Path(__file__).parent))
from h1_remask_rescue import is_correct

ROOT = Path(__file__).resolve().parents[2]
SCOPE = ROOT / "runs" / "validation" / "scope_fail_prompts.json"
OUT_BASE = ROOT / "runs" / "validation"

# (name, kind, arg)
#   kind="uniform"    arg=block_length
#   kind="nonuniform" arg=(block_sizes, steps_per_block)
DEFAULT_LAYOUTS = [
    ("bl8",  "uniform", 8),
    ("bl16", "uniform", 16),
    ("bl32", "uniform", 32),   # baseline
    ("bl64", "uniform", 64),
    ("short_then_long", "nonuniform",
     ([16, 16, 16, 16, 64], [16, 16, 16, 16, 64])),  # sum=128, steps=128
]


def compute_verdict(records: list[dict], layout_names: list[str]) -> dict:
    N = len(records)
    base_ok = sum(1 for r in records if r["per_layout"].get("bl32", False))
    any_ok = sum(1 for r in records if any(r["per_layout"].values()))
    rescued = sum(
        1 for r in records
        if any(r["per_layout"].values()) and not r["per_layout"].get("bl32", False)
    )
    broken = sum(
        1 for r in records
        if r["per_layout"].get("bl32", False) and not any(r["per_layout"].values())
    )
    rescue_rate = rescued / max(N, 1)
    any_rate = any_ok / max(N, 1)

    if rescue_rate >= 0.05:
        verdict = "SUPPORTED"
    elif rescue_rate <= 0.01:
        verdict = "REJECTED"
    else:
        verdict = "INCONCLUSIVE"

    per_layout_ok = {
        name: sum(1 for r in records if r["per_layout"].get(name, False))
        for name in layout_names
    }
    return {
        "n": N,
        "base_correct": base_ok,
        "any_layout_correct": any_ok,
        "rescued": rescued,
        "broken": broken,
        "rescue_rate": rescue_rate,
        "any_layout_rate": any_rate,
        "per_layout_correct": per_layout_ok,
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--gen_length", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    add_common_args(ap)
    add_server_arg(ap)
    args = ap.parse_args()

    assert SCOPE.exists(), f"先跑 h0_forensics.py 生成 {SCOPE}"
    fails = json.loads(SCOPE.read_text(encoding="utf-8"))[: args.n]
    print(f"[A4] 使用 {len(fails)} 条 fail prompt")

    layout_names = [name for name, *_ in DEFAULT_LAYOUTS]
    print(f"[A4] layouts = {layout_names}")

    run_dir = resolve_run_dir(args, "a4_block_rerank", OUT_BASE)
    rd = RunDir(run_dir, "A4", config={**vars(args), "layouts": layout_names},
                resume=args.resume)
    print(f"[A4] run_dir = {rd.dir}")

    done_before = sum(1 for i in range(len(fails)) if rd.has_prompt(i))
    todo = [i for i in range(len(fails)) if not rd.has_prompt(i)]
    print(f"[A4] done_before={done_before}  todo={len(todo)}")

    if args.dry_run:
        print("[A4] DRY RUN — 不连 server")
        print(f"     {len(todo)} 条 × {len(layout_names)} layout "
              f"= {len(todo)*len(layout_names)} 次 HTTP call")
        print(f"     server_url = {args.server_url}")
        return

    api = ValidationAPIClient(args.server_url)
    api.check_health()

    pp = ProgressPrinter(len(todo), tag="A4 ")
    for i in todo:
        rec = fails[i]
        prompt, gt = rec["prompt"], rec["ground_truth"]

        per_layout = {}
        tails = {}
        for name, kind, arg in DEFAULT_LAYOUTS:
            if kind == "uniform":
                bl = arg
                if args.gen_length % bl != 0:
                    per_layout[name] = False
                    tails[name] = f"<skipped: gen_length={args.gen_length} not divisible by {bl}>"
                    continue
                num_blocks = args.gen_length // bl
                steps_ = max(args.steps, num_blocks)
                while steps_ % num_blocks != 0:
                    steps_ += 1
                out = api.generate(
                    prompt, strategy="confidence",
                    max_new_tokens=args.gen_length, num_steps=steps_,
                    block_length=bl, temperature=0.0,
                )
            else:  # nonuniform
                block_sizes, steps_per_block = arg
                out = api.generate_block_schedule(
                    prompt,
                    block_sizes=block_sizes,
                    steps_per_block=steps_per_block,
                    temperature=0.0,
                )
            per_layout[name] = bool(is_correct(out, gt))
            tails[name] = out[-200:]

        record = {
            "idx": i, "gt": gt,
            "per_layout": per_layout,
            "tails": tails,
        }
        rd.save_prompt(i, record)
        ok_str = "/".join(f"{name}={int(per_layout[name])}" for name in layout_names)
        pp.tick(ok_str)

    all_recs = rd.load_all_prompts()
    verdict = compute_verdict(all_recs, layout_names)
    rd.write_summary({**verdict, "config": rd.config})

    print()
    print("═" * 60)
    print(f"[A4] N={verdict['n']}  base(bl32)={verdict['base_correct']}  "
          f"any_layout={verdict['any_layout_correct']}")
    for name, ok in verdict["per_layout_correct"].items():
        print(f"     {name}: {ok}/{verdict['n']}")
    print(f"     rescued={verdict['rescued']}  broken={verdict['broken']}")
    print(f"     rescue_rate={verdict['rescue_rate']:.3%}  "
          f"any_rate={verdict['any_layout_rate']:.3%}")
    print(f"[A4] Verdict: {verdict['verdict']}")
    print(f"[A4] summary → {rd.summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
