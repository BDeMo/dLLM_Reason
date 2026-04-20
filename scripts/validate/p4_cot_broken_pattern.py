"""P4 —— cot_step / cot_plain 砸 baseline-对的那些 prompt 有什么共性？

从 A5 run 读 per_prompt/*.json，对每条 prompt 算特征：
  - char_len / word_len / sentence_count
  - digit_count / arithmetic_hint_count (+, -, ×, ÷, times, sum, etc.)
  - question_head_hint：first sentence 前缀词 (John/Kylar/... human 名，或 how much/many/...)

然后分 4 组：
  - cot_step_broken  : baseline=T AND cot_step=F
  - cot_step_ok      : baseline=T AND cot_step=T
  - cot_plain_broken : baseline=T AND cot_plain=F
  - cot_plain_ok     : baseline=T AND cot_plain=T

对每个特征：broken vs ok 的均值/分位数差异，用 Mann-Whitney U 估计显著性（不依赖 scipy —— 手写 ranking）。

输出:
  - 每个 feature 每组的 mean / median / min / max
  - 显著差异（U-stat p<0.1，粗筛）
  - 按 broken_score 排序的 prompt 列表（辅助人眼看）

用法:
  python scripts/validate/p4_cot_broken_pattern.py \\
      --a5_run runs/validation/a5_prompt_template_20260415_191434
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOPE = ROOT / "runs" / "validation" / "scope_fail_prompts.json"


_DIGIT_RE = re.compile(r"\d+(?:\.\d+)?")
_ARITH_WORDS = {"plus", "minus", "times", "divided", "sum", "total", "each", "per",
                "twice", "half", "double", "more", "less", "fewer"}
_ARITH_SYMS = set("+-*×÷/%$")
_HOW_RE = re.compile(r"\b(how\s+(many|much|long|old|far))\b", re.I)


def features(prompt: str) -> dict:
    words = prompt.split()
    sents = [s for s in re.split(r"[.!?]\s+", prompt.strip()) if s]
    digits = _DIGIT_RE.findall(prompt)
    arith_word_hits = sum(1 for w in prompt.lower().split()
                          if w.strip(".,!?;:") in _ARITH_WORDS)
    arith_sym_hits = sum(prompt.count(c) for c in _ARITH_SYMS)
    how_hits = len(_HOW_RE.findall(prompt))
    return {
        "char_len": len(prompt),
        "word_len": len(words),
        "sentence_count": len(sents),
        "digit_count": len(digits),
        "arith_word": arith_word_hits,
        "arith_sym": arith_sym_hits,
        "how_hits": how_hits,
    }


def summary(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "min": min(xs),
        "max": max(xs),
        "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
    }


def mann_whitney_u(a: list[float], b: list[float]) -> tuple[float, float]:
    """Approximate Mann-Whitney U via normal approximation. Returns (U, z)."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0.0, 0.0
    combined = [(x, 0) for x in a] + [(x, 1) for x in b]
    combined.sort(key=lambda t: t[0])
    # rank with tie-averaging
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    sum_a = sum(r for r, (_, g) in zip(ranks, combined) if g == 0)
    U_a = sum_a - na * (na + 1) / 2
    mean_U = na * nb / 2
    std_U = (na * nb * (na + nb + 1) / 12) ** 0.5
    z = (U_a - mean_U) / std_U if std_U > 0 else 0.0
    return U_a, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a5_run", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    a5_dir = Path(args.a5_run)
    assert (a5_dir / "per_prompt").is_dir(), f"bad a5_run: {a5_dir}"
    fails = json.loads(SCOPE.read_text(encoding="utf-8"))

    out_path = Path(args.out) if args.out else (
        ROOT / "runs" / "validation" / f"p4_cot_broken_pattern_{a5_dir.name}.json"
    )

    groups: dict[str, list[dict]] = {
        "cot_step_broken": [], "cot_step_ok": [],
        "cot_plain_broken": [], "cot_plain_ok": [],
    }
    per_prompt_info = []

    for fp in sorted((a5_dir / "per_prompt").glob("*.json")):
        rec = json.loads(fp.read_text(encoding="utf-8"))
        idx = rec["idx"]
        pt = rec.get("per_template", {})
        if not pt.get("baseline", False):
            continue
        prompt = fails[idx]["prompt"]
        feats = features(prompt)
        info = {"idx": idx, "gt": rec.get("gt"), "prompt_head": prompt[:120],
                "per_template": pt, **feats}
        per_prompt_info.append(info)
        if pt.get("cot_step", False):
            groups["cot_step_ok"].append(info)
        else:
            groups["cot_step_broken"].append(info)
        if pt.get("cot_plain", False):
            groups["cot_plain_ok"].append(info)
        else:
            groups["cot_plain_broken"].append(info)

    feat_keys = list(features("dummy").keys())
    group_stats = {
        g: {k: summary([x[k] for x in items]) for k in feat_keys}
        for g, items in groups.items()
    }

    # significance tests
    sig = {}
    for broken_key, ok_key in [("cot_step_broken", "cot_step_ok"),
                                ("cot_plain_broken", "cot_plain_ok")]:
        pair = {}
        for k in feat_keys:
            a = [x[k] for x in groups[broken_key]]
            b = [x[k] for x in groups[ok_key]]
            U, z = mann_whitney_u(a, b)
            pair[k] = {"U": U, "z": z, "n_broken": len(a), "n_ok": len(b)}
        sig[f"{broken_key}_vs_{ok_key}"] = pair

    out = {
        "a5_run": str(a5_dir),
        "n_baseline_correct_total": len(per_prompt_info),
        "group_sizes": {g: len(items) for g, items in groups.items()},
        "group_feature_stats": group_stats,
        "significance": sig,
        "broken_lists": {
            "cot_step_broken": [
                {"idx": x["idx"], "gt": x["gt"], "prompt_head": x["prompt_head"],
                 **{k: x[k] for k in feat_keys}}
                for x in groups["cot_step_broken"]
            ],
            "cot_plain_broken": [
                {"idx": x["idx"], "gt": x["gt"], "prompt_head": x["prompt_head"],
                 **{k: x[k] for k in feat_keys}}
                for x in groups["cot_plain_broken"]
            ],
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== P4 cot-broken pattern ===")
    print(f"a5_run: {a5_dir.name}")
    print(f"baseline-correct total: {len(per_prompt_info)}")
    for g, items in groups.items():
        print(f"  {g}: {len(items)}")

    print("\n=== Feature stats (mean) ===")
    header = "feature".ljust(20) + "".join(g.ljust(22) for g in groups)
    print(header)
    for k in feat_keys:
        row = k.ljust(20)
        for g in groups:
            s = group_stats[g][k]
            row += f"μ={s.get('mean', 0):.1f} σ={s.get('stdev', 0):.1f}  ".ljust(22)
        print(row)

    print("\n=== Mann-Whitney z (broken vs ok; |z|>1.64 ≈ p<0.1 two-sided) ===")
    for pair, d in sig.items():
        print(f"  [{pair}]  n_broken={list(d.values())[0]['n_broken']}  "
              f"n_ok={list(d.values())[0]['n_ok']}")
        for k, r in d.items():
            mark = " *" if abs(r["z"]) > 1.64 else ""
            print(f"    {k:<20} z={r['z']:+.2f}{mark}")

    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
