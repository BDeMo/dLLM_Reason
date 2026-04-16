"""A4 × A5 joint 6-cell ensemble —— P3 from closure_a_axis.zh.md

动机:
  overlap 分析预测 `{baseline, answer} × {bl8, bl32, bl64}` 6 格在 60 条上覆盖
  10/18 ≈ 55.56% of fails（跟 20 格上限一致）。实跑验证预测。

做法:
  每条 fail prompt 跑 3 block_length × 2 template = 6 个 (layout, tpl) 组合。
  any-correct 算 rescue。对比：
    - 单 config 正确率（6 个都列）
    - A4-only any（只变 layout）
    - A5-only any（只变 template）
    - joint any（6 格全算）
    - rescue_rate = |joint any ∧ ¬baseline| / N

用法:
  python scripts/validate/a4x5_joint.py --n 2 --dry_run
  python scripts/validate/a4x5_joint.py --n 60
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

BLOCK_LENGTHS = [8, 32, 64]
TEMPLATES = [("baseline", ""), ("answer", "\nAnswer:")]


def cell_name(bl: int, tpl: str) -> str:
    return f"bl{bl}_{tpl}"


def compute_verdict(records: list[dict]) -> dict:
    N = len(records)
    base_key = cell_name(32, "baseline")  # 原 A5/A4 baseline
    base_ok = sum(1 for r in records if r["per_cell"].get(base_key, False))

    joint_any = sum(1 for r in records if any(r["per_cell"].values()))
    # A4-only (baseline tpl, vary bl)
    a4_only_any = sum(
        1 for r in records
        if any(r["per_cell"].get(cell_name(bl, "baseline"), False) for bl in BLOCK_LENGTHS)
    )
    # A5-only (bl=32, vary tpl)
    a5_only_any = sum(
        1 for r in records
        if any(r["per_cell"].get(cell_name(32, tpl), False) for tpl, _ in TEMPLATES)
    )

    rescued = sum(
        1 for r in records
        if any(r["per_cell"].values()) and not r["per_cell"].get(base_key, False)
    )
    rescue_rate = rescued / max(N, 1)

    per_cell_ok = {}
    for bl in BLOCK_LENGTHS:
        for tpl, _ in TEMPLATES:
            key = cell_name(bl, tpl)
            per_cell_ok[key] = sum(1 for r in records if r["per_cell"].get(key, False))

    return {
        "n": N,
        "base_correct": base_ok,
        "joint_any_correct": joint_any,
        "a4_only_any": a4_only_any,
        "a5_only_any": a5_only_any,
        "rescued": rescued,
        "rescue_rate": rescue_rate,
        "joint_any_rate": joint_any / max(N, 1),
        "per_cell_correct": per_cell_ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--gen_length", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    add_common_args(ap)
    add_server_arg(ap)
    args = ap.parse_args()

    fails = json.loads(SCOPE.read_text(encoding="utf-8"))[: args.n]
    cells = [(bl, tpl, suffix) for bl in BLOCK_LENGTHS for tpl, suffix in TEMPLATES]
    cell_names = [cell_name(bl, tpl) for bl, tpl, _ in cells]
    print(f"[A4x5] {len(fails)} prompts × {len(cells)} cells = {len(fails) * len(cells)} calls")

    run_dir = resolve_run_dir(args, "a4x5_joint", OUT_BASE)
    rd = RunDir(run_dir, "A4x5", config={**vars(args), "cells": cell_names},
                resume=args.resume)
    todo = [i for i in range(len(fails)) if not rd.has_prompt(i)]
    print(f"[A4x5] run_dir = {rd.dir}  todo={len(todo)}")

    if args.dry_run:
        print(f"[A4x5] DRY RUN — {len(todo) * len(cells)} 次 HTTP call")
        return

    api = ValidationAPIClient(args.server_url)
    api.check_health()

    pp = ProgressPrinter(len(todo), tag="A4x5 ")
    for i in todo:
        rec = fails[i]
        prompt, gt = rec["prompt"], rec["ground_truth"]
        per_cell = {}
        tails = {}
        for bl, tpl, suffix in cells:
            p = prompt + suffix
            num_blocks = args.gen_length // bl
            steps_ = max(args.steps, num_blocks)
            while steps_ % num_blocks != 0:
                steps_ += 1
            out = api.generate(
                p, strategy="confidence",
                max_new_tokens=args.gen_length, num_steps=steps_,
                block_length=bl, temperature=0.0,
            )
            key = cell_name(bl, tpl)
            per_cell[key] = bool(is_correct(out, gt))
            tails[key] = out[-200:]
        rd.save_prompt(i, {"idx": i, "gt": gt, "per_cell": per_cell, "tails": tails})
        pp.tick(f"any={int(any(per_cell.values()))}")

    all_recs = rd.load_all_prompts()
    v = compute_verdict(all_recs)
    rd.write_summary({**v, "config": rd.config})

    print()
    print("═" * 60)
    print(f"[A4x5] N={v['n']}  base={v['base_correct']}  joint_any={v['joint_any_correct']}")
    print(f"       A4-only any={v['a4_only_any']}  A5-only any={v['a5_only_any']}")
    for k, ok in v["per_cell_correct"].items():
        print(f"       {k}: {ok}/{v['n']}")
    print(f"       rescued={v['rescued']}  rescue_rate={v['rescue_rate']:.3%}")


if __name__ == "__main__":
    main()
