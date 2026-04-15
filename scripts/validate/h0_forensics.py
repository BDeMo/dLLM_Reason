"""H0 (exploratory)：失败案例 forensics + 生成 scope_fail_prompts.json

读 episodes.db 里 correct=0 的 137 条，按 error 类型分桶：
  - numeric_close   : gt 是数字，output 里出现的数字与 gt 差 < 5x
  - numeric_far     : gt 是数字，output 数字与 gt 差很远或不含
  - format_bad      : output 没给出最终数字 / 被截断
  - unknown         : 其它

产出：runs/validation/scope_fail_prompts.json — 后续 H1/H2/H3 实验共用
Usage:
    python scripts/validate/h0_forensics.py
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "runs" / "research_20260411_030422" / "stage2_discovery" / "episodes.db"
OUT_DIR = ROOT / "runs" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "scope_fail_prompts.json"


def extract_numbers(s: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", s or "")]


def classify(gt: str, output: str) -> str:
    try:
        gt_num = float(re.sub(r"[^0-9.\-]", "", gt or "") or "nan")
    except Exception:
        gt_num = float("nan")

    nums_out = extract_numbers(output or "")

    # 格式问题：output 太短、没数字、或明显被截断
    if not output or len(output.strip()) < 20:
        return "format_bad"
    if not nums_out:
        return "format_bad"
    if gt_num != gt_num:  # NaN
        return "unknown"

    # 与 gt 相对接近程度 — 对所有 output 数字算 min rel-err
    best_rel = min(
        abs(n - gt_num) / max(abs(gt_num), 1.0) for n in nums_out
    )
    if best_rel < 0.1:
        return "numeric_close"      # 差 < 10%：小算术错 / 四舍五入
    if best_rel < 1.0:
        return "numeric_mid"        # 差 < 100%：reasoning 链中间断了
    if best_rel < 10.0:
        return "numeric_far"        # 同数量级 ± 1
    return "numeric_order"          # 数量级都错了


def main():
    assert DB.exists(), f"DB not found: {DB}"
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT episode_id, prompt, ground_truth, output, correct, "
        "dag_seq_len, num_steps, block_length FROM episodes WHERE correct=0",
        conn,
    )
    conn.close()
    print(f"[H0] fail rows: {len(df)}")

    df["error_type"] = [classify(gt, out) for gt, out in zip(df.ground_truth, df.output)]
    print("\n[H0] error_type 分布:")
    print(df["error_type"].value_counts())

    # 示例
    print("\n[H0] 每类抽 1 条:")
    for et in df["error_type"].unique():
        row = df[df.error_type == et].iloc[0]
        print(f"  ── {et} ──")
        print(f"     gt={row.ground_truth!r}")
        print(f"     out tail: ...{row.output[-200:]!r}")

    # 保存
    records = df.to_dict(orient="records")
    OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[H0] saved {len(records)} fail prompts → {OUT.relative_to(ROOT)}")
    print("[H0] 无 verdict（H0 是 scope 生成，非假设验证）")


if __name__ == "__main__":
    main()
