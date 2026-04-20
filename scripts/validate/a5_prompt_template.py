"""A5：Prompt-template rerank — 换 CoT 前缀 / 答案前缀，看 pass@any_template

A3/A4 在 sampler-side 换干预粒度；A5 更粗一级 —— 直接换 prompt 形状，不动 sampler。

对每条 fail prompt 跑 K 个 template：
  - "baseline" : 原 prompt 原样
  - "cot_plain": prompt + "\\nLet's solve this step by step."
  - "cot_step" : prompt + "\\nStep 1:" （强制 CoT 结构）
  - "answer"  : prompt + "\\nAnswer:"（直接要数字）

Verdict 阈值（同 H1）：
  rescue_rate ≥ 5%  → SUPPORTED
  rescue_rate ≤ 1%  → REJECTED
  否则              → INCONCLUSIVE

Usage:
    python scripts/validate/a5_prompt_template.py --n 2 --dry_run
    python scripts/validate/a5_prompt_template.py --n 137
    python scripts/validate/a5_prompt_template.py --n 137 --server_url http://1.2.3.4:8000
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

# (name, suffix_appended_to_prompt)
DEFAULT_TEMPLATES = [
    ("baseline",  ""),
    ("cot_plain", "\nLet's solve this step by step."),
    ("cot_step",  "\nStep 1:"),
    ("answer",    "\nAnswer:"),
]


def compute_verdict(records: list[dict], tpl_names: list[str]) -> dict:
    N = len(records)
    base_ok = sum(1 for r in records if r["per_template"].get("baseline", False))
    any_ok = sum(1 for r in records if any(r["per_template"].values()))
    rescued = sum(
        1 for r in records
        if any(r["per_template"].values()) and not r["per_template"].get("baseline", False)
    )
    broken = sum(
        1 for r in records
        if r["per_template"].get("baseline", False) and not any(r["per_template"].values())
    )
    rescue_rate = rescued / max(N, 1)
    any_rate = any_ok / max(N, 1)

    if rescue_rate >= 0.05:
        verdict = "SUPPORTED"
    elif rescue_rate <= 0.01:
        verdict = "REJECTED"
    else:
        verdict = "INCONCLUSIVE"

    per_tpl_ok = {
        name: sum(1 for r in records if r["per_template"].get(name, False))
        for name in tpl_names
    }
    return {
        "n": N,
        "base_correct": base_ok,
        "any_template_correct": any_ok,
        "rescued": rescued,
        "broken": broken,
        "rescue_rate": rescue_rate,
        "any_template_rate": any_rate,
        "per_template_correct": per_tpl_ok,
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--gen_length", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--block_length", type=int, default=32)
    add_common_args(ap)
    add_server_arg(ap)
    args = ap.parse_args()

    assert SCOPE.exists(), f"先跑 h0_forensics.py 生成 {SCOPE}"
    fails = json.loads(SCOPE.read_text(encoding="utf-8"))[: args.n]
    print(f"[A5] 使用 {len(fails)} 条 fail prompt")

    tpl_names = [name for name, _ in DEFAULT_TEMPLATES]
    print(f"[A5] templates = {tpl_names}")

    run_dir = resolve_run_dir(args, "a5_prompt_template", OUT_BASE)
    rd = RunDir(run_dir, "A5", config={**vars(args), "templates": tpl_names},
                resume=args.resume)
    print(f"[A5] run_dir = {rd.dir}")

    done_before = sum(1 for i in range(len(fails)) if rd.has_prompt(i))
    todo = [i for i in range(len(fails)) if not rd.has_prompt(i)]
    print(f"[A5] done_before={done_before}  todo={len(todo)}")

    if args.dry_run:
        print("[A5] DRY RUN — 不连 server")
        print(f"     {len(todo)} 条 × {len(tpl_names)} template "
              f"= {len(todo)*len(tpl_names)} 次 HTTP call")
        print(f"     server_url = {args.server_url}")
        return

    api = ValidationAPIClient(args.server_url)
    api.check_health()

    pp = ProgressPrinter(len(todo), tag="A5 ")
    for i in todo:
        rec = fails[i]
        prompt, gt = rec["prompt"], rec["ground_truth"]

        per_template = {}
        tails = {}
        for name, suffix in DEFAULT_TEMPLATES:
            p = prompt + suffix
            out = api.generate(
                p, strategy="confidence",
                max_new_tokens=args.gen_length, num_steps=args.steps,
                block_length=args.block_length, temperature=0.0,
            )
            per_template[name] = bool(is_correct(out, gt))
            tails[name] = out[-200:]

        record = {
            "idx": i, "gt": gt,
            "per_template": per_template,
            "tails": tails,
        }
        rd.save_prompt(i, record)
        ok_str = "/".join(f"{name}={int(per_template[name])}" for name in tpl_names)
        pp.tick(ok_str)

    all_recs = rd.load_all_prompts()
    verdict = compute_verdict(all_recs, tpl_names)
    rd.write_summary({**verdict, "config": rd.config})

    print()
    print("═" * 60)
    print(f"[A5] N={verdict['n']}  base={verdict['base_correct']}  "
          f"any_template={verdict['any_template_correct']}")
    for name, ok in verdict["per_template_correct"].items():
        print(f"     {name}: {ok}/{verdict['n']}")
    print(f"     rescued={verdict['rescued']}  broken={verdict['broken']}")
    print(f"     rescue_rate={verdict['rescue_rate']:.3%}  "
          f"any_rate={verdict['any_template_rate']:.3%}")
    print(f"[A5] Verdict: {verdict['verdict']}")
    print(f"[A5] summary → {rd.summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
