"""E5：g128 截断检查 — 排除 A6 增益的 trivial 解释

动机（`docs/archive/discussion_latent_space_reasoning.zh.md` §3.5）:
  最无聊的反方解释是 "g128=128 tokens 对某些 prompt 物理上写不完"。
  如果 g128 答案被截断（答案没生成完就 budget 用光），g160 的增益就只是
  "多给点 token 位置让答案写完"，不需要任何 latent reasoning 来解释。

做法:
  对 A6 run_dir 的每条 prompt × 每个 gen_length，offline 分析 tail：
    - 是否以句号/问号/感叹号结尾（完整句收尾）
    - 是否含 "answer is XXX" / "= XXX" 等 answer marker
    - tail 长度是否接近 200 chars（A6 存 out[-200:]）
  综合给出 is_likely_truncated 判定。

判定:
  对 A6 独救 3 条 (idx=0, 19, 51) 的 g128 tail:
    - 若 ≥2 条 likely_truncated → A6 独救增益是 trivial 截断效应 → latent reasoning 不成立
    - 若 ≤1 条 likely_truncated → 增益不能被截断解释 → 需要继续做 E1/E2

输出:
  runs/validation/e5_truncation_<ts>/
    per_prompt_truncation.json   # 每条 prompt × gen_length 的判定
    summary.json                 # truncation_rate per gen_length + a6_only 聚焦

用法:
  python scripts/validate/e5_truncation_check.py  \\
      --a6_run_dir runs/validation/a6_gen_length_20260416_012648
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_BASE = ROOT / "runs" / "validation"

A6_ONLY_IDX = {0, 19, 51}

# A6 的 tails 是 out[-200:]，所以 tail 长度 ≤ 200。若 ≥195 说明原输出被裁剪到 200
# 字符边界以下，或者输出在 200 chars 内没收尾 —— 前者靠不住判定，后者是截断信号。
TAIL_STORAGE_LIMIT = 200

# "答案 marker" 启发式：常见 CoT 收尾 pattern
ANSWER_MARKERS = [
    re.compile(r"answer\s+is\s+[\$\-]?\d", re.IGNORECASE),
    re.compile(r"=\s*[\$\-]?\d[\d,\.]*\s*[\.\s]*$"),  # "= 42." or "= $42"
    re.compile(r"####\s*[\$\-]?\d"),  # gsm8k style "#### 42"
    re.compile(r"\bfinal\s+answer\b", re.IGNORECASE),
    re.compile(r"therefore[,\s]+.*?\d", re.IGNORECASE),
]

# 收尾符号（"正常结束"的标志）
SENTENCE_END = re.compile(r"[.!?]\s*$")
# 数字 + 可选 period（"The profit is 50000" 或 "...is 50000."）
ENDS_WITH_DIGIT = re.compile(r"\d[\d,]*\.?\s*$")
# 明显未完成（以 CJK/letter/连接符结尾，没有终止符）
ENDS_MID_WORD = re.compile(r"[A-Za-z\u4e00-\u9fff\-_/]\s*$")


def analyze_tail(tail: str) -> dict:
    """分析单个 tail，返回判定字典。"""
    stripped = tail.rstrip()
    has_marker = any(pat.search(tail) for pat in ANSWER_MARKERS)
    ends_sentence = bool(SENTENCE_END.search(stripped))
    ends_digit = bool(ENDS_WITH_DIGIT.search(stripped))
    ends_mid_word = bool(ENDS_MID_WORD.search(stripped)) and not ends_sentence

    # 综合判定：
    #   - 有 answer marker 且以句号/数字收尾 → 明确未截断
    #   - 无 answer marker 且 mid-word 结尾 → 明确截断
    #   - 其它 → ambiguous
    if has_marker and (ends_sentence or ends_digit):
        verdict = "complete"
    elif ends_mid_word and not has_marker:
        verdict = "truncated"
    elif ends_sentence and not has_marker:
        # 句号结尾但没有 answer marker —— 可能是推理链中段的句号
        # 保守判 "maybe_truncated"
        verdict = "maybe_truncated"
    elif has_marker and not (ends_sentence or ends_digit):
        # 有 marker 但结尾奇怪 —— 少见
        verdict = "ambiguous"
    elif ends_digit:
        # 数字结尾但没句号/marker —— 可能是答案数字被 budget 抢没了
        verdict = "maybe_truncated"
    else:
        verdict = "ambiguous"

    return {
        "has_answer_marker": has_marker,
        "ends_sentence": ends_sentence,
        "ends_digit": ends_digit,
        "ends_mid_word": ends_mid_word,
        "tail_len_chars": len(tail),
        "verdict": verdict,
        "last_40": tail[-40:],
    }


def analyze_run(a6_run_dir: Path) -> tuple[list[dict], dict]:
    """分析整个 A6 run，返回 (per_prompt_detail, summary)"""
    per_prompt_files = sorted((a6_run_dir / "per_prompt").glob("????.json"))
    assert per_prompt_files, f"no per_prompt/*.json under {a6_run_dir}"

    per_prompt_detail = []
    for p in per_prompt_files:
        rec = json.loads(p.read_text(encoding="utf-8"))
        idx = rec["idx"]
        per_len_analysis = {}
        for length_name, tail in rec["tails"].items():
            per_len_analysis[length_name] = {
                **analyze_tail(tail),
                "correct": bool(rec["per_length"].get(length_name, False)),
            }
        per_prompt_detail.append({
            "idx": idx,
            "gt": rec["gt"],
            "is_a6_only": idx in A6_ONLY_IDX,
            "per_length": per_len_analysis,
        })

    # Aggregate: truncation rate per gen_length
    lengths = sorted(per_prompt_detail[0]["per_length"].keys(),
                     key=lambda x: int(x.lstrip("g")))
    trunc_stats = {}
    for L in lengths:
        verdicts = [r["per_length"][L]["verdict"] for r in per_prompt_detail]
        trunc_stats[L] = {
            "complete": verdicts.count("complete"),
            "truncated": verdicts.count("truncated"),
            "maybe_truncated": verdicts.count("maybe_truncated"),
            "ambiguous": verdicts.count("ambiguous"),
            "n": len(verdicts),
            "truncated_rate": verdicts.count("truncated") / max(len(verdicts), 1),
            "any_trunc_rate": (verdicts.count("truncated") + verdicts.count("maybe_truncated")) / max(len(verdicts), 1),
        }

    # A6 独救 3 条 focus
    a6_only_focus = {}
    for r in per_prompt_detail:
        if r["is_a6_only"]:
            a6_only_focus[r["idx"]] = {
                "gt": r["gt"],
                "g128": r["per_length"].get("g128", {}),
                "g160": r["per_length"].get("g160", {}),
                "g192": r["per_length"].get("g192", {}),
            }

    # Final verdict for E5
    a6_only_g128_trunc = sum(
        1 for v in a6_only_focus.values()
        if v.get("g128", {}).get("verdict") in ("truncated", "maybe_truncated")
    )
    if a6_only_g128_trunc >= 2:
        e5_verdict = "TRIVIAL_TRUNCATION"
        interp = f"{a6_only_g128_trunc}/3 A6-only prompts have truncated g128 → A6 gain likely just budget"
    elif a6_only_g128_trunc <= 1:
        e5_verdict = "NOT_TRUNCATION"
        interp = f"only {a6_only_g128_trunc}/3 truncated → A6 gain cannot be explained by truncation"
    else:
        e5_verdict = "AMBIGUOUS"
        interp = f"{a6_only_g128_trunc}/3 truncated"

    summary = {
        "a6_run_dir": str(a6_run_dir),
        "n_prompts": len(per_prompt_detail),
        "per_length_truncation": trunc_stats,
        "a6_only_focus": a6_only_focus,
        "a6_only_g128_truncated_count": a6_only_g128_trunc,
        "verdict": e5_verdict,
        "interpretation": interp,
    }
    return per_prompt_detail, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a6_run_dir", type=str, required=True,
                    help="A6 run_dir containing per_prompt/*.json")
    ap.add_argument("--run_dir", type=str, default=None,
                    help="output dir; default runs/validation/e5_truncation_<ts>/")
    args = ap.parse_args()

    a6_rd = Path(args.a6_run_dir).resolve()
    assert a6_rd.exists(), f"{a6_rd} doesn't exist"

    out_rd = Path(args.run_dir) if args.run_dir else (
        OUT_BASE / f"e5_truncation_{datetime.now():%Y%m%d_%H%M%S}"
    )
    out_rd.mkdir(parents=True, exist_ok=True)

    per_prompt, summary = analyze_run(a6_rd)

    (out_rd / "per_prompt_truncation.json").write_text(
        json.dumps(per_prompt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary["timestamp"] = datetime.now().isoformat(timespec="seconds")
    (out_rd / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[E5] analyzed {summary['n_prompts']} prompts from {a6_rd.name}")
    print(f"[E5] output → {out_rd}")
    print()
    print("per-length truncation breakdown:")
    print(f"  {'len':>5} {'comp':>5} {'trun':>5} {'maybe':>6} {'ambig':>6}  "
          f"trunc%  any_trunc%")
    for L, s in summary["per_length_truncation"].items():
        print(f"  {L:>5} {s['complete']:>5} {s['truncated']:>5} "
              f"{s['maybe_truncated']:>6} {s['ambiguous']:>6}  "
              f"{s['truncated_rate']:>6.1%}  {s['any_trunc_rate']:>6.1%}")
    print()
    print(f"A6-only focus (idx={sorted(A6_ONLY_IDX)}):")
    for idx, focus in summary["a6_only_focus"].items():
        print(f"  idx={idx}  gt={focus['gt']}")
        for L in ("g128", "g160", "g192"):
            info = focus.get(L, {})
            if not info:
                print(f"    {L}: <missing>")
                continue
            correct_mark = "✓" if info.get("correct") else "✗"
            print(f"    {L}: {info['verdict']:>16} "
                  f"correct={correct_mark}  "
                  f"last_40={info.get('last_40', '')!r}")
    print()
    print(f"[E5] A6-only g128 truncated count: "
          f"{summary['a6_only_g128_truncated_count']}/3")
    print(f"[E5] Verdict: {summary['verdict']}")
    print(f"     {summary['interpretation']}")


if __name__ == "__main__":
    main()
