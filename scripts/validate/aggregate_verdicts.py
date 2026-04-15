"""扫 runs/validation/h{1,2,3}_*/summary.json，更新 hypotheses.md 结论板。

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
HYP_MD = ROOT / "docs" / "archive" / "hypotheses.md"


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


def fmt_h0() -> tuple[str, str]:
    scope = VAL / "scope_fail_prompts.json"
    if not scope.exists():
        return "—", "—"
    try:
        n = len(json.loads(scope.read_text(encoding="utf-8")))
        return "DONE", f"{n} fail prompts → runs/validation/scope_fail_prompts.json"
    except Exception:
        return "—", "—"


def update_md(rows: dict[str, tuple[str, str, str]]) -> None:
    """rows: {H_key: (script, verdict, nums)}"""
    md = HYP_MD.read_text(encoding="utf-8")
    today = datetime.now().strftime("%Y-%m-%d")

    table_header = "| 假设 | 脚本 | Verdict | 关键数字 | 日期 |"
    lines = md.splitlines()
    try:
        hi = lines.index(table_header)
    except ValueError:
        print(f"[aggregate] 找不到表头 '{table_header}'，请检查 hypotheses.md")
        return

    # 表头之后 2 行开始为数据（header + 分隔 + 数据）
    data_start = hi + 2
    new_lines = lines[:data_start]

    for key in ("H0", "H1", "H2", "H3"):
        script, verdict, nums = rows.get(key, ("—", "—", "—"))
        date = today if verdict not in ("—",) else "—"
        new_lines.append(f"| {key} | `{script}` | {verdict} | {nums} | {date} |")

    # 剩余非表格内容保留（例如末尾空行）
    # 找表格结束位置（连续的 | 行）
    tail_start = data_start
    for i in range(data_start, len(lines)):
        if not lines[i].startswith("|"):
            tail_start = i
            break
    else:
        tail_start = len(lines)
    new_lines.extend(lines[tail_start:])

    HYP_MD.write_text("\n".join(new_lines) + ("\n" if md.endswith("\n") else ""), encoding="utf-8")


def main():
    h0_v, h0_nums = fmt_h0()

    h1s = latest_summary("h1_remask_*")
    h2s = latest_summary("h2_order_content_*")
    h3s = latest_summary("h3_passN_*")

    if h1s:
        h1_v, h1_nums = fmt_h1(h1s)
    else:
        h1_v, h1_nums = "—", "—"
    if h2s:
        h2_v, h2_nums = fmt_h2(h2s)
    else:
        h2_v, h2_nums = "—", "—"
    if h3s:
        h3_v, h3_nums = fmt_h3(h3s)
    else:
        h3_v, h3_nums = "—", "—"

    rows = {
        "H0": ("h0_forensics.py", h0_v, h0_nums),
        "H1": ("h1_remask_rescue.py", h1_v, h1_nums),
        "H2": ("h2_order_vs_content.py", h2_v, h2_nums),
        "H3": ("h3_passN_at_temperature.py", h3_v, h3_nums),
    }

    print("═" * 60)
    print("[aggregate] 最新 verdicts")
    print("═" * 60)
    for k, (s, v, n) in rows.items():
        print(f"  {k}: {v:<14} | {n}")

    update_md(rows)
    print(f"\n[aggregate] 已更新 {HYP_MD.relative_to(ROOT)}")
    for name, s in (("H1", h1s), ("H2", h2s), ("H3", h3s)):
        if s:
            print(f"  {name} run: {s['_run_dir']}")


if __name__ == "__main__":
    main()
