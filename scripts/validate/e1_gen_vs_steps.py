"""E1：gen_length vs num_steps 解耦 — 判定 A6 g160 增益来自"空间"还是"计算"

动机（`docs/archive/discussion_latent_space_reasoning.zh.md`）:
  A6 的设计 num_steps = gen_length，两者同时变。
  g160 相对 g128 同时多了：
    - 32 个 token 位置 (空间)
    - 32 步 diffusion (计算)
  必须拆开才能判定 latent reasoning 假说。

三配置对比:
  C = baseline:       gen=128, steps=128, bl=32 → 4 blocks × 32 steps/blk
  A = longA (=A6 g160): gen=160, steps=160, bl=32 → 5 blocks × 32 steps/blk
  B = stepsB:         gen=128, steps=160, bl=32 → 4 blocks × 40 steps/blk
                                                    ↑ 空间不变，每 block 多 8 步 diffusion

判定逻辑:
  记 R_A = (A 独救 C 救不了的 prompt 数) / N
  记 R_B = (B 独救 C 救不了的 prompt 数) / N
  记 R_AB_overlap = (A,B 都能救但 C 救不了的)

  主要指标 R_B:
    R_B ≥ 5%  → **SUPPORTED**  latent reasoning (计算是关键)
    R_B ≤ 1%  → **REJECTED**   latent reasoning (空间是关键)
    otherwise → INCONCLUSIVE

  次要比较:
    R_A > R_B 显著     → 空间增益 > 计算增益 (混合, 不纯 latent)
    R_A ≈ R_B          → 两者等价 (弱支持 latent)
    R_B > R_A          → 计算占主导 (强支持 latent)

重点关注 A6 独救的 3 条 (idx=0, 19, 51):
  如果 B 能救其中 ≥2 条 → A6 独救由 num_steps 驱动 → 强证据支持 latent reasoning
  如果 B 一条都救不了   → A6 独救由 gen_length 驱动 → 不支持 latent reasoning

用法:
  python scripts/validate/e1_gen_vs_steps.py --n 2 --dry_run
  python scripts/validate/e1_gen_vs_steps.py --n 60
  python scripts/validate/e1_gen_vs_steps.py --n 60 --resume
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

# 三配置，命名 "<label>_g<gen>_s<steps>"
CONFIGS = [
    # label,     gen_length, num_steps, 语义
    ("C_g128_s128", 128, 128),  # baseline: 空间 128, 计算 128
    ("A_g160_s160", 160, 160),  # A6 g160:  空间 160, 计算 160  (= 空间↑ + 计算↑)
    ("B_g128_s160", 128, 160),  # stepsB:   空间 128, 计算 160  (= 空间不变, 计算↑)
]

# A6 独救的 3 条（from empirical_rescue_per_prompt.zh.md）
A6_ONLY_IDX = {0, 19, 51}


def compute_verdict(records: list[dict]) -> dict:
    N = len(records)
    names = [cfg[0] for cfg in CONFIGS]
    base_name = CONFIGS[0][0]  # "C_g128_s128"
    longA_name = CONFIGS[1][0]  # "A_g160_s160"
    stepsB_name = CONFIGS[2][0]  # "B_g128_s160"

    per_cfg_ok = {
        name: sum(1 for r in records if r["per_config"].get(name, False))
        for name in names
    }

    base_ok = per_cfg_ok[base_name]
    # "any of {A,B} 能救 ∧ C 救不了"
    any_rescued = sum(
        1 for r in records
        if (r["per_config"].get(longA_name) or r["per_config"].get(stepsB_name))
        and not r["per_config"].get(base_name)
    )
    longA_rescued = sum(
        1 for r in records
        if r["per_config"].get(longA_name) and not r["per_config"].get(base_name)
    )
    stepsB_rescued = sum(
        1 for r in records
        if r["per_config"].get(stepsB_name) and not r["per_config"].get(base_name)
    )
    # overlap / exclusive
    both_rescued = sum(
        1 for r in records
        if r["per_config"].get(longA_name)
        and r["per_config"].get(stepsB_name)
        and not r["per_config"].get(base_name)
    )
    longA_only = sum(
        1 for r in records
        if r["per_config"].get(longA_name)
        and not r["per_config"].get(stepsB_name)
        and not r["per_config"].get(base_name)
    )
    stepsB_only = sum(
        1 for r in records
        if r["per_config"].get(stepsB_name)
        and not r["per_config"].get(longA_name)
        and not r["per_config"].get(base_name)
    )

    # 反向：baseline 对但 A/B 错（退化数）
    broken_by_longA = sum(
        1 for r in records
        if r["per_config"].get(base_name) and not r["per_config"].get(longA_name)
    )
    broken_by_stepsB = sum(
        1 for r in records
        if r["per_config"].get(base_name) and not r["per_config"].get(stepsB_name)
    )

    r_a = longA_rescued / max(N, 1)
    r_b = stepsB_rescued / max(N, 1)

    # 主判定：rescue_rate 三档（与 A4/A5/A6/H1 一致），
    # interpretation 字段给 latent reasoning 的具体含义
    if r_b >= 0.05:
        verdict = "SUPPORTED"
        interpretation = "latent_reasoning_supported (stepsB rescues without extra space)"
    elif r_b <= 0.01:
        verdict = "REJECTED"
        interpretation = "latent_reasoning_rejected (extra steps alone doesn't rescue)"
    else:
        verdict = "INCONCLUSIVE"
        interpretation = f"R_B={r_b:.2%} between 1% and 5%"

    # A6 独救 3 条的细粒度检查
    a6_only_outcome = {}
    for r in records:
        if r["idx"] in A6_ONLY_IDX:
            a6_only_outcome[r["idx"]] = {
                base_name: r["per_config"].get(base_name, False),
                longA_name: r["per_config"].get(longA_name, False),
                stepsB_name: r["per_config"].get(stepsB_name, False),
            }
    a6_only_by_stepsB = sum(
        1 for v in a6_only_outcome.values() if v[stepsB_name] and not v[base_name]
    )
    a6_only_by_longA = sum(
        1 for v in a6_only_outcome.values() if v[longA_name] and not v[base_name]
    )

    return {
        "n": N,
        "per_config_correct": per_cfg_ok,
        "base_correct": base_ok,
        "any_rescued": any_rescued,
        "longA_rescued": longA_rescued,
        "stepsB_rescued": stepsB_rescued,
        "overlap_both_rescued": both_rescued,
        "longA_only_rescued": longA_only,
        "stepsB_only_rescued": stepsB_only,
        "broken_by_longA": broken_by_longA,
        "broken_by_stepsB": broken_by_stepsB,
        "rescue_rate_longA": r_a,
        "rescue_rate_stepsB": r_b,
        "verdict": verdict,
        "interpretation": interpretation,
        "a6_only_idx": sorted(A6_ONLY_IDX),
        "a6_only_n_total": len(a6_only_outcome),
        "a6_only_rescued_by_stepsB": a6_only_by_stepsB,
        "a6_only_rescued_by_longA": a6_only_by_longA,
        "a6_only_outcome": a6_only_outcome,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--block_length", type=int, default=32)
    add_common_args(ap)
    add_server_arg(ap)
    args = ap.parse_args()

    assert SCOPE.exists(), f"先跑 h0_forensics.py 生成 {SCOPE}"
    fails = json.loads(SCOPE.read_text(encoding="utf-8"))[: args.n]
    print(f"[E1] 使用 {len(fails)} 条 fail prompt")
    max_a6_idx = max(A6_ONLY_IDX)
    if args.n <= max_a6_idx:
        print(f"[E1] ⚠ --n={args.n} 会漏掉 A6 独救 idx "
              f"{sorted(i for i in A6_ONLY_IDX if i >= args.n)}. "
              f"建议 --n >= {max_a6_idx + 1}")
    print(f"[E1] 三配置 (block_length={args.block_length}):")
    for label, gen_len, steps in CONFIGS:
        nb = gen_len // args.block_length
        spb = steps // nb if steps % nb == 0 else f"{steps}/{nb} (auto-adjust)"
        print(f"       {label}: gen={gen_len}, steps={steps}, "
              f"num_blocks={nb}, steps/blk={spb}")

    run_dir = resolve_run_dir(args, "e1_gen_vs_steps", OUT_BASE)
    rd = RunDir(run_dir, "E1",
                config={**vars(args), "configs": [list(c) for c in CONFIGS]},
                resume=args.resume)
    print(f"[E1] run_dir = {rd.dir}")

    todo = [i for i in range(len(fails)) if not rd.has_prompt(i)]
    done_before = len(fails) - len(todo)
    print(f"[E1] done_before={done_before}  todo={len(todo)}")

    if args.dry_run:
        print(f"[E1] DRY RUN — {len(todo) * len(CONFIGS)} 次 HTTP call "
              f"(3 configs × {len(todo)} prompts)")
        return

    api = ValidationAPIClient(args.server_url)
    api.check_health()

    pp = ProgressPrinter(len(todo), tag="E1 ")
    for i in todo:
        rec = fails[i]
        prompt, gt = rec["prompt"], rec["ground_truth"]
        per_config = {}
        tails = {}
        for label, gen_len, steps in CONFIGS:
            # pre-check 整除性（auto-adjust 会发生但我们想知道）
            if gen_len % args.block_length != 0:
                per_config[label] = False
                tails[label] = f"<skipped: gen={gen_len} not divisible by bl={args.block_length}>"
                continue
            out = api.generate(
                prompt, strategy="confidence",
                max_new_tokens=gen_len, num_steps=steps,
                block_length=args.block_length, temperature=0.0,
            )
            per_config[label] = bool(is_correct(out, gt))
            tails[label] = out[-200:]
        a6_tag = " [A6_ONLY]" if i in A6_ONLY_IDX else ""
        rd.save_prompt(i, {
            "idx": i, "gt": gt,
            "is_a6_only": i in A6_ONLY_IDX,
            "per_config": per_config,
            "tails": tails,
        })
        ok_str = "/".join(f"{lab.split('_')[0]}={int(per_config[lab])}"
                          for lab, _, _ in CONFIGS)
        pp.tick(ok_str + a6_tag)

    all_recs = rd.load_all_prompts()
    verdict = compute_verdict(all_recs)
    rd.write_summary({**verdict, "config": rd.config})

    print()
    print("═" * 72)
    print(f"[E1] N={verdict['n']}")
    print(f"     per_config correct:")
    for name, ok in verdict["per_config_correct"].items():
        pct = ok / max(verdict["n"], 1) * 100
        print(f"       {name}: {ok}/{verdict['n']}  ({pct:.1f}%)")
    print()
    print(f"     rescue (相对 baseline C_g128_s128):")
    print(f"       longA  (空间↑+计算↑, = A6 g160): {verdict['longA_rescued']}/{verdict['n']}"
          f"  ({verdict['rescue_rate_longA']:.2%})")
    print(f"       stepsB (空间不变+计算↑):        {verdict['stepsB_rescued']}/{verdict['n']}"
          f"  ({verdict['rescue_rate_stepsB']:.2%})")
    print(f"       overlap (both):      {verdict['overlap_both_rescued']}")
    print(f"       longA_only:          {verdict['longA_only_rescued']}")
    print(f"       stepsB_only:         {verdict['stepsB_only_rescued']}")
    print(f"     broken (baseline 对 ∧ 新配置错):")
    print(f"       longA:   {verdict['broken_by_longA']}")
    print(f"       stepsB:  {verdict['broken_by_stepsB']}")
    print()
    print(f"     A6 独救 3 条 (idx={verdict['a6_only_idx']}) 细粒度:")
    print(f"       covered in this run: {verdict['a6_only_n_total']}/3")
    print(f"       rescued by longA:    {verdict['a6_only_rescued_by_longA']}")
    print(f"       rescued by stepsB:   {verdict['a6_only_rescued_by_stepsB']}  ← 关键信号")
    for idx, outcome in sorted(verdict["a6_only_outcome"].items()):
        flags = "  ".join(f"{k}={'✓' if v else '✗'}" for k, v in outcome.items())
        print(f"       idx={idx}: {flags}")
    print()
    print(f"[E1] Verdict: {verdict['verdict']}")
    print(f"     interpretation: {verdict['interpretation']}")
    print(f"     R_B (stepsB rescue_rate) = {verdict['rescue_rate_stepsB']:.2%}")
    print(f"     R_A (longA rescue_rate)  = {verdict['rescue_rate_longA']:.2%}")
    if verdict["rescue_rate_stepsB"] >= verdict["rescue_rate_longA"] * 0.7 and \
       verdict["rescue_rate_stepsB"] >= 0.05:
        print(f"     → stepsB ≳ longA → 计算占主导 → 强支持 latent reasoning")
    elif verdict["rescue_rate_stepsB"] < verdict["rescue_rate_longA"] * 0.3:
        print(f"     → longA >> stepsB → 空间占主导 → 不支持 latent reasoning")
    else:
        print(f"     → longA > stepsB 但 stepsB 不可忽略 → 混合效应")


if __name__ == "__main__":
    main()
