"""扫 runs/validation/h{1,2,3}_*/summary.json，更新 hypotheses{.md, .zh.md} 结论板。

策略：对每个 hypothesis 取**最新时间戳**的 run_dir 作为权威结果。
idempotent — 重复跑只会用最新 summary 覆盖对应行。

Usage:
    python scripts/validate/aggregate_verdicts.py
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "runs" / "validation"
HYP_MD_EN = ROOT / "docs" / "archive" / "hypotheses.md"
HYP_MD_ZH = ROOT / "docs" / "archive" / "hypotheses.zh.md"


def latest_summary(pattern: str) -> dict | None:
    runs = sorted(VAL.glob(pattern))
    for rd in reversed(runs):
        sp = rd / "summary.json"
        if sp.exists():
            d = json.loads(sp.read_text(encoding="utf-8"))
            d["_run_dir"] = str(rd.relative_to(ROOT))
            d["_summary_path"] = str(sp.relative_to(ROOT))
            return d
    return None


def fmt_h1(s: dict) -> tuple[str, str]:
    nums = (f"N={s.get('n','?')}  base={s.get('base_correct','?')}  "
            f"revise={s.get('revise_correct','?')}  "
            f"rescued={s.get('rescued','?')}  broken={s.get('broken','?')}  "
            f"rescue_rate={s.get('rescue_rate',0):.2%}")
    return s.get("verdict", "—"), nums


def fmt_h2(s: dict) -> tuple[str, str]:
    nums = (f"N={s.get('n','?')}  "
            f"content_var={s.get('mean_content_var',0):.3f}  "
            f"order_var={s.get('mean_order_var',0):.3f}  "
            f"ratio={s.get('mean_ratio',float('nan')):.3f}")
    return s.get("verdict", "—"), nums


def fmt_h3(s: dict) -> tuple[str, str]:
    nums = (f"n_fail={s.get('n_fail','?')}  n_ok={s.get('n_ok','?')}  "
            f"fail_p@8={s.get('fail_pass@8_max',0):.2%}  "
            f"ok_p@8={s.get('ok_pass@8_max',0):.2%}")
    return s.get("verdict", "—"), nums


def fmt_a3(s: dict) -> tuple[str, str]:
    """A3 summary shape 同 H1 (复用 h1.compute_verdict)。"""
    nums = (f"N={s.get('n','?')}  base={s.get('base_correct','?')}  "
            f"revise={s.get('revise_correct','?')}  "
            f"rescued={s.get('rescued','?')}  broken={s.get('broken','?')}  "
            f"rescue_rate={s.get('rescue_rate',0):.2%}")
    return s.get("verdict", "—"), nums


def fmt_a4(s: dict) -> tuple[str, str]:
    per = s.get("per_layout_correct", {})
    per_str = " ".join(f"{k}={v}" for k, v in per.items())
    nums = (f"N={s.get('n','?')}  base(bl32)={s.get('base_correct','?')}  "
            f"any={s.get('any_layout_correct','?')}  "
            f"rescue_rate={s.get('rescue_rate',0):.2%}  "
            f"[{per_str}]")
    return s.get("verdict", "—"), nums


def fmt_a5(s: dict) -> tuple[str, str]:
    per = s.get("per_template_correct", {})
    per_str = " ".join(f"{k}={v}" for k, v in per.items())
    nums = (f"N={s.get('n','?')}  base={s.get('base_correct','?')}  "
            f"any={s.get('any_template_correct','?')}  "
            f"rescue_rate={s.get('rescue_rate',0):.2%}  "
            f"[{per_str}]")
    return s.get("verdict", "—"), nums


def fmt_a6(s: dict) -> tuple[str, str]:
    per = s.get("per_length_correct", {})
    per_str = " ".join(f"{k}={v}" for k, v in per.items())
    nums = (f"N={s.get('n','?')}  base(g128)={s.get('base_correct','?')}  "
            f"any={s.get('any_length_correct','?')}  "
            f"rescue_rate={s.get('rescue_rate',0):.2%}  "
            f"[{per_str}]")
    return s.get("verdict", "—"), nums


def fmt_a4x5(s: dict) -> tuple[str, str]:
    per = s.get("per_cell_correct", {})
    per_str = " ".join(f"{k}={v}" for k, v in per.items())
    nums = (f"N={s.get('n','?')}  base={s.get('base_correct','?')}  "
            f"joint_any={s.get('joint_any_correct','?')}  "
            f"rescue_rate={s.get('rescue_rate',0):.2%}  "
            f"[{per_str}]")
    return "—", nums  # no verdict threshold defined for joint ensemble


def fmt_e1(s: dict) -> tuple[str, str]:
    """E1: gen_length vs num_steps 解耦。See e1_gen_vs_steps.summary.json schema."""
    per = s.get("per_config_correct", {})
    a6_only = s.get("a6_only_rescued_by_longA", "?")
    a6_only_b = s.get("a6_only_rescued_by_stepsB", "?")
    a6_only_n = s.get("a6_only_n_total", "?")
    labels = list(per.keys())
    per_str = "  ".join(f"{lbl}={per[lbl]}" for lbl in labels)
    nums = (f"N={s.get('n','?')}  {per_str}  "
            f"rescue_longA={s.get('rescue_rate_longA',0):.2%}  "
            f"rescue_stepsB={s.get('rescue_rate_stepsB',0):.2%}  "
            f"a6_only_longA={a6_only}/{a6_only_n}  "
            f"a6_only_stepsB={a6_only_b}/{a6_only_n}")
    return s.get("verdict", "—"), nums


def fmt_e5(s: dict) -> tuple[str, str]:
    """E5: A6 tail 截断 offline 检查。"""
    trunc = s.get("a6_only_g128_truncated_count", "?")
    focus = s.get("a6_only_focus", {})
    parts = []
    for idx, per_len in focus.items():
        g128 = per_len.get("g128", {})
        v = g128.get("verdict", "?")
        c = "✓" if g128.get("correct") else "✗"
        parts.append(f"idx{idx}:{v[:4]}/{c}")
    short = "  ".join(parts) if parts else "—"
    nums = f"a6_only_g128_trunc={trunc}/3  [{short}]"
    return s.get("verdict", "—"), nums


def fmt_h0() -> tuple[str, str]:
    scope = VAL / "scope_fail_prompts.json"
    if not scope.exists():
        return "—", "—"
    try:
        n = len(json.loads(scope.read_text(encoding="utf-8")))
        return "DONE", f"{n} fail prompts → runs/validation/scope_fail_prompts.json"
    except Exception:
        return "—", "—"


# 可能的表头（EN / ZH 任何一个命中就更新）
KNOWN_HEADERS = [
    "| Hypothesis | Script | Verdict | Key numbers | Date |",
    "| 假设 | 脚本 | Verdict | 关键数字 | 日期 |",
]


def _rewrite_table(md_path: Path, rows: dict[str, tuple[str, str, str]]) -> bool:
    if not md_path.exists():
        return False
    md = md_path.read_text(encoding="utf-8")
    today = datetime.now().strftime("%Y-%m-%d")
    lines = md.splitlines()
    hi = -1
    for hdr in KNOWN_HEADERS:
        if hdr in lines:
            hi = lines.index(hdr)
            break
    if hi < 0:
        print(f"[aggregate] 未在 {md_path.name} 中找到已知表头，跳过（请确认已 git pull 最新 archive 模板）")
        return False
    data_start = hi + 2
    new_lines = lines[:data_start]
    for key in ("H0", "H1", "H2", "H3", "A3", "A4", "A5", "A6", "A4x5", "E1", "E5"):
        script, verdict, nums = rows.get(key, ("—", "—", "—"))
        date = today if verdict not in ("—",) else "—"
        new_lines.append(f"| {key} | `{script}` | {verdict} | {nums} | {date} |")
    tail_start = data_start
    for i in range(data_start, len(lines)):
        if not lines[i].startswith("|"):
            tail_start = i
            break
    else:
        tail_start = len(lines)
    new_lines.extend(lines[tail_start:])
    md_path.write_text("\n".join(new_lines) + ("\n" if md.endswith("\n") else ""),
                       encoding="utf-8")
    return True


def update_md(rows: dict[str, tuple[str, str, str]]) -> None:
    """rows: {H_key: (script, verdict, nums)}. Updates EN and ZH tables if present."""
    for p in (HYP_MD_EN, HYP_MD_ZH):
        _rewrite_table(p, rows)


def main():
    h0_v, h0_nums = fmt_h0()

    h1s = latest_summary("h1_remask_*")
    h2s = latest_summary("h2_order_content_*")
    h3s = latest_summary("h3_passN_*")
    a3s = latest_summary("a3_span_revise_*")
    a4s = latest_summary("a4_block_rerank_*")
    a5s = latest_summary("a5_prompt_template_*")
    a6s = latest_summary("a6_gen_length_*")
    jointers = latest_summary("a4x5_joint_*")
    e1s = latest_summary("e1_gen_vs_steps_*")
    e5s = latest_summary("e5_truncation_*")

    def _or_dash(s, fmt):
        return fmt(s) if s else ("—", "—")

    h1_v, h1_nums = _or_dash(h1s, fmt_h1)
    h2_v, h2_nums = _or_dash(h2s, fmt_h2)
    h3_v, h3_nums = _or_dash(h3s, fmt_h3)
    a3_v, a3_nums = _or_dash(a3s, fmt_a3)
    a4_v, a4_nums = _or_dash(a4s, fmt_a4)
    a5_v, a5_nums = _or_dash(a5s, fmt_a5)
    a6_v, a6_nums = _or_dash(a6s, fmt_a6)
    a4x5_v, a4x5_nums = _or_dash(jointers, fmt_a4x5)
    e1_v, e1_nums = _or_dash(e1s, fmt_e1)
    e5_v, e5_nums = _or_dash(e5s, fmt_e5)

    rows = {
        "H0": ("h0_forensics.py", h0_v, h0_nums),
        "H1": ("h1_remask_rescue.py", h1_v, h1_nums),
        "H2": ("h2_order_vs_content.py", h2_v, h2_nums),
        "H3": ("h3_passN_at_temperature.py", h3_v, h3_nums),
        "A3": ("a3_span_revise.py", a3_v, a3_nums),
        "A4": ("a4_block_rerank.py", a4_v, a4_nums),
        "A5": ("a5_prompt_template.py", a5_v, a5_nums),
        "A6": ("a6_gen_length.py", a6_v, a6_nums),
        "A4x5": ("a4x5_joint.py", a4x5_v, a4x5_nums),
        "E1": ("e1_gen_vs_steps.py", e1_v, e1_nums),
        "E5": ("e5_truncation_check.py", e5_v, e5_nums),
    }

    print("═" * 60)
    print("[aggregate] 最新 verdicts")
    print("═" * 60)
    for k, (s, v, n) in rows.items():
        print(f"  {k}: {v:<14} | {n}")

    update_md(rows)
    for p in (HYP_MD_EN, HYP_MD_ZH):
        if p.exists():
            print(f"\n[aggregate] 已更新 {p.relative_to(ROOT)}")
    for name, s in (("H1", h1s), ("H2", h2s), ("H3", h3s),
                    ("A3", a3s), ("A4", a4s), ("A5", a5s),
                    ("A6", a6s), ("A4x5", jointers),
                    ("E1", e1s), ("E5", e5s)):
        if s:
            print(f"  {name} run: {s['_run_dir']}")


if __name__ == "__main__":
    main()
