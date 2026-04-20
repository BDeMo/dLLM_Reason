"""A6：gen_length rerank — P7 from closure_a_axis.zh.md

动机:
  P2.1.e 意外发现 g128 → g256 下 5/5 fail prompt 里 4 条 baseline 从正确翻转成错 (T=0)。
  total length 是 A4 没扫过的 layout 维度。A6 把 gen_length 当 A 轴第 5 把旋钮。

做法:
  每条 fail prompt 用 {64, 96, 128, 160, 192, 256} 跑 baseline（block_length=32 固定），
  算 per-length 正确率、any_length rescue rate。

Verdict（同 A4/A5）:
  rescue_rate ≥ 5%  → SUPPORTED
  rescue_rate ≤ 1%  → REJECTED
  否则              → INCONCLUSIVE

用法:
  python scripts/validate/a6_gen_length.py --n 2 --dry_run
  python scripts/validate/a6_gen_length.py --n 60
  python scripts/validate/a6_gen_length.py --n 137 --resume
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

DEFAULT_LENGTHS = [64, 96, 128, 160, 192, 256]  # 全部被 block_length=32 整除
BASELINE_LEN = 128  # A5/A4 baseline


def compute_verdict(records: list[dict], names: list[str]) -> dict:
    N = len(records)
    base_key = f"g{BASELINE_LEN}"
    base_ok = sum(1 for r in records if r["per_length"].get(base_key, False))
    any_ok = sum(1 for r in records if any(r["per_length"].values()))
    rescued = sum(
        1 for r in records
        if any(r["per_length"].values()) and not r["per_length"].get(base_key, False)
    )
    broken = sum(
        1 for r in records
        if r["per_length"].get(base_key, False) and not any(r["per_length"].values())
    )
    rescue_rate = rescued / max(N, 1)
    any_rate = any_ok / max(N, 1)
    verdict = "SUPPORTED" if rescue_rate >= 0.05 else (
        "REJECTED" if rescue_rate <= 0.01 else "INCONCLUSIVE"
    )
    per_len_ok = {
        name: sum(1 for r in records if r["per_length"].get(name, False))
        for name in names
    }
    return {
        "n": N,
        "base_correct": base_ok,
        "any_length_correct": any_ok,
        "rescued": rescued,
        "broken": broken,
        "rescue_rate": rescue_rate,
        "any_length_rate": any_rate,
        "per_length_correct": per_len_ok,
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--lengths", type=str, default=",".join(str(x) for x in DEFAULT_LENGTHS))
    add_common_args(ap)
    add_server_arg(ap)
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    assert SCOPE.exists(), f"先跑 h0_forensics.py 生成 {SCOPE}"
    fails = json.loads(SCOPE.read_text(encoding="utf-8"))[: args.n]
    names = [f"g{L}" for L in lengths]
    print(f"[A6] 使用 {len(fails)} 条 fail prompt · lengths = {lengths}")

    run_dir = resolve_run_dir(args, "a6_gen_length", OUT_BASE)
    rd = RunDir(run_dir, "A6", config={**vars(args), "lengths": lengths},
                resume=args.resume)
    print(f"[A6] run_dir = {rd.dir}")

    todo = [i for i in range(len(fails)) if not rd.has_prompt(i)]
    done_before = len(fails) - len(todo)
    print(f"[A6] done_before={done_before}  todo={len(todo)}")

    if args.dry_run:
        print(f"[A6] DRY RUN — {len(todo) * len(lengths)} 次 HTTP call")
        return

    api = ValidationAPIClient(args.server_url)
    api.check_health()

    pp = ProgressPrinter(len(todo), tag="A6 ")
    for i in todo:
        rec = fails[i]
        prompt, gt = rec["prompt"], rec["ground_truth"]
        per_length = {}
        tails = {}
        for L, name in zip(lengths, names):
            if L % args.block_length != 0:
                per_length[name] = False
                tails[name] = f"<skipped: {L} not divisible by bl={args.block_length}>"
                continue
            out = api.generate(
                prompt, strategy="confidence",
                max_new_tokens=L, num_steps=L,
                block_length=args.block_length, temperature=0.0,
            )
            per_length[name] = bool(is_correct(out, gt))
            tails[name] = out[-200:]
        rd.save_prompt(i, {"idx": i, "gt": gt,
                           "per_length": per_length, "tails": tails})
        ok_str = "/".join(f"{n}={int(per_length[n])}" for n in names)
        pp.tick(ok_str)

    all_recs = rd.load_all_prompts()
    verdict = compute_verdict(all_recs, names)
    rd.write_summary({**verdict, "config": rd.config})

    print()
    print("═" * 60)
    print(f"[A6] N={verdict['n']}  base(g{BASELINE_LEN})={verdict['base_correct']}  "
          f"any_length={verdict['any_length_correct']}")
    for name, ok in verdict["per_length_correct"].items():
        print(f"     {name}: {ok}/{verdict['n']}")
    print(f"     rescued={verdict['rescued']}  broken={verdict['broken']}")
    print(f"     rescue_rate={verdict['rescue_rate']:.3%}  "
          f"any_rate={verdict['any_length_rate']:.3%}")
    print(f"[A6] Verdict: {verdict['verdict']}")


if __name__ == "__main__":
    main()
