"""H3：LLaDA-instruct 在 gsm8k 的 137 条 init_fail 上达到能力上限

断言：即使加温度 + 多次重采样，这些 prompt 的 pass@N 依然 ≈ 0。

做法：
  fail 集取 K 条 (默认 30) + 对照组 K 条 (init_ok)
  每条 × T ∈ {0.3, 0.7, 1.0} × N=8 次
  算 pass@1 / pass@4 / pass@8（at least one correct）

Verdict 阈值：
  fail_pass@8 < 5% 且 ok_pass@8 > 90%  → SUPPORTED（能力上限）
  fail_pass@8 > 20%                      → REJECTED（采样 diversity 能救）
  否则                                    → INCONCLUSIVE

每条 prompt 一个 per_prompt/XXXX.json，文件名用 "{group}_{idx:04d}.json"
(group ∈ {"fail", "ok"})，支持 resume。

Usage:
    python scripts/validate/h3_passN_at_temperature.py --n_fail 2 --n_ok 2 --dry_run
    python scripts/validate/h3_passN_at_temperature.py --n_fail 30 --n_ok 30 --n_samples 8
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from _runlib import RunDir, ProgressPrinter, add_common_args, resolve_run_dir

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[2]
SCOPE = ROOT / "runs" / "validation" / "scope_fail_prompts.json"
OUT_BASE = ROOT / "runs" / "validation"


def find_latest_db() -> Path | None:
    cands = sorted((ROOT / "runs").glob("research_*/stage2_discovery/episodes.db"))
    return cands[-1] if cands else None


def load_init_ok(k: int, db_path: Path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT prompt, ground_truth FROM episodes WHERE correct=1 LIMIT ?", (k,))
    rows = cur.fetchall()
    conn.close()
    return [{"prompt": p, "ground_truth": gt} for p, gt in rows]


def pass_at_k(corrects: list[bool], k: int) -> float:
    return 1.0 if any(corrects[:k]) else 0.0


class H3RunDir(RunDir):
    """H3 自定义文件名：{group}_{idx:04d}.json"""
    def prompt_key(self, group: str, idx: int) -> str:
        return f"{group}_{idx:04d}"

    def prompt_path_group(self, group: str, idx: int) -> Path:
        return self.per_prompt / f"{self.prompt_key(group, idx)}.json"

    def has_prompt_group(self, group: str, idx: int) -> bool:
        return self.prompt_path_group(group, idx).exists()

    def save_prompt_group(self, group: str, idx: int, record: dict) -> None:
        import os
        p = self.prompt_path_group(group, idx)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)
        from datetime import datetime
        with self.progress.open("a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"group": group, "idx": idx,
                 "ts": datetime.now().isoformat(timespec="seconds"),
                 **{k: v for k, v in record.items() if not isinstance(v, (dict, list))}},
                ensure_ascii=False,
            ) + "\n")


def aggregate_group_stats(records: list[dict], temps: list[float]) -> dict:
    stats = {str(T): {"pass@1": 0, "pass@4": 0, "pass@8": 0, "n": 0} for T in temps}
    for r in records:
        for T in temps:
            tk = str(T)
            if tk not in r.get("temps", {}):
                continue
            e = r["temps"][tk]
            stats[tk]["pass@1"] += e["pass@1"]
            stats[tk]["pass@4"] += e["pass@4"]
            stats[tk]["pass@8"] += e["pass@8"]
            stats[tk]["n"] += 1
    for tk in stats:
        n = max(stats[tk]["n"], 1)
        for k in ("pass@1", "pass@4", "pass@8"):
            stats[tk][k] /= n
    return stats


def compute_verdict(fail_recs: list[dict], ok_recs: list[dict], temps: list[float]) -> dict:
    fail_stats = aggregate_group_stats(fail_recs, temps)
    ok_stats = aggregate_group_stats(ok_recs, temps)
    fail_p8 = max((fail_stats[str(T)]["pass@8"] for T in temps), default=0.0)
    ok_p8 = max((ok_stats[str(T)]["pass@8"] for T in temps), default=0.0)

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
        "n_fail": len(fail_recs),
        "n_ok": len(ok_recs),
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/llada-instruct")
    ap.add_argument("--n_fail", type=int, default=30)
    ap.add_argument("--n_ok", type=int, default=30)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--gen_length", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--temps", type=float, nargs="+", default=[0.3, 0.7, 1.0])
    ap.add_argument("--db", type=str, default=None,
                    help="episodes.db 路径，留空则自动取 runs/research_*/stage2_discovery/episodes.db 最新一个")
    add_common_args(ap)
    args = ap.parse_args()

    assert SCOPE.exists(), f"先跑 h0_forensics.py 生成 {SCOPE}"
    fail_prompts = json.loads(SCOPE.read_text(encoding="utf-8"))[: args.n_fail]

    db_path = Path(args.db) if args.db else find_latest_db()
    if args.dry_run and (db_path is None or not db_path.exists()):
        ok_prompts = [{"prompt": "(dry_run placeholder)", "ground_truth": "0"}] * args.n_ok
        print(f"[H3] DRY RUN: 无 db，使用 placeholder ok prompts")
    else:
        assert db_path is not None and db_path.exists(), (
            f"DB not found: {db_path}\n"
            f"显式传 --db 或确保 runs/research_*/stage2_discovery/episodes.db 存在"
        )
        print(f"[H3] using DB: {db_path.relative_to(ROOT) if db_path.is_relative_to(ROOT) else db_path}")
        ok_prompts = load_init_ok(args.n_ok, db_path)
    print(f"[H3] fail={len(fail_prompts)}  ok={len(ok_prompts)}  "
          f"temps={args.temps}  N={args.n_samples}")

    run_dir = resolve_run_dir(args, "h3_passN", OUT_BASE)
    rd = H3RunDir(run_dir, "H3", config=vars(args), resume=args.resume)
    print(f"[H3] run_dir = {rd.dir}")

    fail_todo = [i for i in range(len(fail_prompts)) if not rd.has_prompt_group("fail", i)]
    ok_todo = [i for i in range(len(ok_prompts)) if not rd.has_prompt_group("ok", i)]
    print(f"[H3] fail_todo={len(fail_todo)}  ok_todo={len(ok_todo)}")

    if args.dry_run:
        per_prompt = len(args.temps) * args.n_samples
        total = (len(fail_todo) + len(ok_todo)) * per_prompt
        print("[H3] DRY RUN — 不加载模型")
        print(f"     每条 prompt: {per_prompt} 次 generate")
        print(f"     共 {total} 次 generate，保存到 {rd.dir}/per_prompt/")
        return

    # Lazy import
    import torch
    from transformers import AutoModel, AutoTokenizer
    from h1_remask_rescue import generate, _get_mask_token_id, is_correct

    print(f"[H3] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).cuda().eval()
    mask_id = _get_mask_token_id(model, tok)
    print(f"[H3] mask_id = {mask_id}")

    def run_one(group: str, idx: int, rec: dict):
        prompt, gt = rec["prompt"], rec["ground_truth"]
        row = {"group": group, "idx": idx, "gt": gt, "temps": {}}
        for T in args.temps:
            corrects = []
            for _ in range(args.n_samples):
                out = generate(model, tok, prompt,
                               gen_length=args.gen_length, steps=args.steps,
                               block_length=args.block_length, temperature=T,
                               revise_every=0, mask_id=mask_id)
                corrects.append(is_correct(out, gt))
            p1 = pass_at_k(corrects, 1)
            p4 = pass_at_k(corrects, min(4, args.n_samples))
            p8 = pass_at_k(corrects, args.n_samples)
            row["temps"][str(T)] = {"correct_list": [bool(c) for c in corrects],
                                    "pass@1": p1, "pass@4": p4, "pass@8": p8}
        rd.save_prompt_group(group, idx, row)
        return row

    total = len(fail_todo) + len(ok_todo)
    pp = ProgressPrinter(total, tag="H3 ")
    for i in fail_todo:
        row = run_one("fail", i, fail_prompts[i])
        p8s = [row["temps"][str(T)]["pass@8"] for T in args.temps]
        pp.tick(f"FAIL[{i}] p@8={'/'.join(f'{x:.0f}' for x in p8s)}")
    for i in ok_todo:
        row = run_one("ok", i, ok_prompts[i])
        p8s = [row["temps"][str(T)]["pass@8"] for T in args.temps]
        pp.tick(f"OK  [{i}] p@8={'/'.join(f'{x:.0f}' for x in p8s)}")

    # Aggregate
    fail_recs = [json.loads(p.read_text(encoding="utf-8"))
                 for p in sorted(rd.per_prompt.glob("fail_*.json"))]
    ok_recs = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(rd.per_prompt.glob("ok_*.json"))]
    verdict = compute_verdict(fail_recs, ok_recs, args.temps)
    rd.write_summary({**verdict, "config": rd.config})

    print()
    print("═" * 60)
    print("[H3] fail 集 pass 汇总:")
    for T, s in verdict["fail_stats"].items():
        print(f"     T={T}  p@1={s['pass@1']:.2%}  p@4={s['pass@4']:.2%}  p@8={s['pass@8']:.2%}")
    print("[H3] ok   集 pass 汇总:")
    for T, s in verdict["ok_stats"].items():
        print(f"     T={T}  p@1={s['pass@1']:.2%}  p@4={s['pass@4']:.2%}  p@8={s['pass@8']:.2%}")
    print(f"[H3] fail_p@8_max = {verdict['fail_pass@8_max']:.2%}   "
          f"ok_p@8_max = {verdict['ok_pass@8_max']:.2%}")
    print(f"[H3] Verdict: {verdict['verdict']}")
    print(f"[H3] summary → {rd.summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
