"""H3 post-hoc aggregator — 从已存在的 per_prompt/*.json 重建 summary.json

原 `h3_passN_at_temperature.py` 跑到一半中断（没到 rd.write_summary），导致
`aggregate_verdicts.py` 看不到 H3。本脚本读 per_prompt 文件离线聚合。

同时生成 `h3_rescue_crossref.json`：
  per-prompt 细粒度：
    - any-T pass@{1,4,8} rescued
    - rescued by which temperature(s)
  以便跟 A4/A5/A6 base_fail 做交叉对比（不依赖 p5_h3_crossref.py 那个 bug 脚本）。

用法:
    python scripts/validate/h3_summarize.py  \\
        --run_dir runs/validation/h3_passN_20260415_133254
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_group(run_dir: Path, group: str) -> list[dict]:
    files = sorted((run_dir / "per_prompt").glob(f"{group}_*.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


def aggregate_group_stats(records: list[dict], temps: list[str]) -> dict:
    stats = {T: {"pass@1": 0.0, "pass@4": 0.0, "pass@8": 0.0, "n": 0} for T in temps}
    for r in records:
        for T in temps:
            e = r.get("temps", {}).get(T)
            if e is None:
                continue
            for k in ("pass@1", "pass@4", "pass@8"):
                stats[T][k] += e[k]
            stats[T]["n"] += 1
    for T in stats:
        n = max(stats[T]["n"], 1)
        for k in ("pass@1", "pass@4", "pass@8"):
            stats[T][k] /= n
    return stats


def compute_verdict(fail_recs: list[dict], ok_recs: list[dict],
                    temps: list[str]) -> dict:
    fail_stats = aggregate_group_stats(fail_recs, temps)
    ok_stats = aggregate_group_stats(ok_recs, temps)
    fail_p8 = max((fail_stats[T]["pass@8"] for T in temps), default=0.0)
    ok_p8 = max((ok_stats[T]["pass@8"] for T in temps), default=0.0)

    # Additional aggregates: any-T pass@k (per-prompt → mean)
    any_t_rescue = {}
    for k in (1, 4, 8):
        rescued = sum(
            1 for r in fail_recs
            if any(r["temps"][T][f"pass@{k}"] >= 1.0 for T in temps)
        )
        any_t_rescue[f"fail_anyT_pass@{k}"] = {
            "rescued": rescued,
            "n": len(fail_recs),
            "rate": rescued / max(len(fail_recs), 1),
        }

    if fail_p8 < 0.05 and ok_p8 > 0.90:
        verdict = "SUPPORTED"
    elif fail_p8 > 0.20:
        verdict = "REJECTED"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "fail_stats": fail_stats,
        "ok_stats": ok_stats,
        "fail_pass@8_max": fail_p8,
        "ok_pass@8_max": ok_p8,
        "any_t_rescue": any_t_rescue,
        "n_fail": len(fail_recs),
        "n_ok": len(ok_recs),
        "verdict": verdict,
    }


def build_crossref(fail_recs: list[dict], temps: list[str]) -> list[dict]:
    out = []
    for r in fail_recs:
        row = {"idx": r["idx"], "gt": r["gt"]}
        for T in temps:
            e = r.get("temps", {}).get(T, {})
            row[f"T{T}_pass@1"] = e.get("pass@1", 0)
            row[f"T{T}_pass@8"] = e.get("pass@8", 0)
        row["anyT_rescued"] = any(
            r["temps"][T]["pass@8"] >= 1.0 for T in temps
        )
        row["rescued_by_T"] = [T for T in temps
                               if r["temps"][T]["pass@8"] >= 1.0]
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True,
                    help="H3 run dir containing per_prompt/")
    args = ap.parse_args()

    rd = Path(args.run_dir).resolve()
    assert rd.exists(), f"{rd} doesn't exist"

    cfg_path = rd / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    temps = [str(T) for T in cfg.get("temps", [0.3, 0.7, 1.0])]

    fail_recs = load_group(rd, "fail")
    ok_recs = load_group(rd, "ok")
    print(f"[H3] {len(fail_recs)} fail + {len(ok_recs)} ok records loaded")
    print(f"[H3] temps = {temps}")

    verdict = compute_verdict(fail_recs, ok_recs, temps)
    from datetime import datetime
    summary = {
        **verdict,
        "config": cfg,
        "timestamp_end": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(rd),
        "regenerated_by": "h3_summarize.py (post-hoc aggregation)",
    }
    summary_path = rd / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    # Crossref with A-axis base_fail
    crossref = build_crossref(fail_recs, temps)
    crossref_path = rd / "h3_rescue_crossref.json"
    crossref_path.write_text(json.dumps(crossref, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    print()
    print("═" * 60)
    print("[H3] FAIL pass summary:")
    for T, s in verdict["fail_stats"].items():
        print(f"     T={T}  p@1={s['pass@1']:.2%}  "
              f"p@4={s['pass@4']:.2%}  p@8={s['pass@8']:.2%}  (n={s['n']})")
    print("[H3] OK   pass summary:")
    for T, s in verdict["ok_stats"].items():
        print(f"     T={T}  p@1={s['pass@1']:.2%}  "
              f"p@4={s['pass@4']:.2%}  p@8={s['pass@8']:.2%}  (n={s['n']})")
    print()
    print("[H3] any-T rescue (per-prompt union across temps):")
    for k, v in verdict["any_t_rescue"].items():
        print(f"     {k}: {v['rescued']}/{v['n']}  ({v['rate']:.2%})")
    print()
    print(f"[H3] Verdict: {verdict['verdict']}")
    print(f"[H3] summary  → {summary_path.relative_to(rd.parent.parent)}")
    print(f"[H3] crossref → {crossref_path.relative_to(rd.parent.parent)}")


if __name__ == "__main__":
    main()
