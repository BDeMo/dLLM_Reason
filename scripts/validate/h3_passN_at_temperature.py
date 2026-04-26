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

依赖：
    runs/validation/scope_fail_prompts.json  (H0 产出)
    runs/validation/scope_ok_prompts.json    (H0 产出，init_ok 对照组)

Usage:
    python scripts/validate/h3_passN_at_temperature.py --n_fail 2 --n_ok 2 --dry_run
    python scripts/validate/h3_passN_at_temperature.py --n_fail 30 --n_ok 30 --n_samples 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _runlib import RunDir, ProgressPrinter, add_common_args, resolve_run_dir

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[2]
SCOPE = ROOT / "runs" / "validation" / "scope_fail_prompts.json"
SCOPE_OK = ROOT / "runs" / "validation" / "scope_ok_prompts.json"
OUT_BASE = ROOT / "runs" / "validation"


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
    ap.add_argument("--prompt_batch", default="auto",
                    help="cross-prompt batching: stack P prompts × N samples "
                         "into one forward (B=P*N). 'auto' (default) probes "
                         "at startup, finding the largest P that fits memory "
                         "given worst-case prompt length. Or pass an int to "
                         "force a value. Set 1 to disable cross-prompt "
                         "batching (single-prompt B=N).")
    ap.add_argument("--prompt_shard", default="0/1",
                    help="<idx>/<total> — round-robin partition of (fail+ok) "
                         "prompts across shards. Each shard processes prompts "
                         "where i %% total == idx. Outputs share the same "
                         "run_dir; per_prompt/<group>_<idx>.json is keyed by "
                         "global idx so shards never collide. Used by "
                         "t6_decode_ablate to spread one (T, N) cell across "
                         "multiple GPUs.")
    add_common_args(ap)
    args = ap.parse_args()
    try:
        SHARD_IDX, SHARD_TOTAL = (int(x) for x in args.prompt_shard.split("/"))
        assert 0 <= SHARD_IDX < SHARD_TOTAL
    except Exception:
        raise SystemExit(f"--prompt_shard must be 'idx/total', got {args.prompt_shard!r}")

    assert SCOPE.exists(), f"先跑 h0_forensics.py 生成 {SCOPE}"
    assert SCOPE_OK.exists(), f"先跑 h0_forensics.py 生成 {SCOPE_OK}（H3 需要 init_ok 对照组）"
    fail_prompts = json.loads(SCOPE.read_text(encoding="utf-8"))[: args.n_fail]
    ok_prompts = json.loads(SCOPE_OK.read_text(encoding="utf-8"))[: args.n_ok]
    print(f"[H3] fail={len(fail_prompts)}  ok={len(ok_prompts)}  "
          f"temps={args.temps}  N={args.n_samples}")

    run_dir = resolve_run_dir(args, "h3_passN", OUT_BASE)
    rd = H3RunDir(run_dir, "H3", config=vars(args), resume=args.resume)
    print(f"[H3] run_dir = {rd.dir}")

    fail_todo = [i for i in range(len(fail_prompts))
                 if not rd.has_prompt_group("fail", i)
                 and i % SHARD_TOTAL == SHARD_IDX]
    ok_todo = [i for i in range(len(ok_prompts))
               if not rd.has_prompt_group("ok", i)
               and i % SHARD_TOTAL == SHARD_IDX]
    print(f"[H3] shard {SHARD_IDX}/{SHARD_TOTAL}: "
          f"fail_todo={len(fail_todo)}  ok_todo={len(ok_todo)}")

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
    from h1_remask_rescue import (generate, generate_batched, generate_batched_multi,
                                    _get_mask_token_id, is_correct, extract_answer)

    print(f"[H3] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).cuda().eval()
    mask_id = _get_mask_token_id(model, tok)
    print(f"[H3] mask_id = {mask_id}")

    # ── Resolve (P_BATCH, N_INNER) — auto-probe or explicit ────────────────
    # Strategy: probe forward passes at decreasing B = P × N_inner candidates;
    # pick the largest one that fits worst-case prompt length without OOM.
    # N_total = args.n_samples (metric requirement). N_inner ≤ N_total; we
    # iterate ceil(N_total / N_inner) times if needed to collect N_total
    # samples per prompt.
    N_TOTAL = args.n_samples

    def _autotune_batch():
        from itertools import product
        # Worst-case sequence length: tokenize each prompt header
        all_prompts = ([rec["prompt"] for rec in fail_prompts]
                       + [rec["prompt"] for rec in ok_prompts])
        worst_real = 0
        for p in all_prompts:
            text = tok.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True, tokenize=False)
            worst_real = max(worst_real, len(tok(text)["input_ids"]))
        L = worst_real + args.gen_length
        print(f"[H3] autotune: worst_prompt={worst_real}, total_seq_L={L}")

        # Candidates: (P, N_inner). Prefer larger B; among ties prefer P
        # since cross-prompt is more SM-saturating than within-prompt.
        Ps = [16, 12, 8, 4, 2, 1]
        Ns = sorted({N_TOTAL, max(1, N_TOTAL // 2),
                     max(1, N_TOTAL // 4), 1}, reverse=True)
        cands = sorted({(P, N) for P, N in product(Ps, Ns)},
                       key=lambda pn: (-(pn[0] * pn[1]), -pn[0]))

        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        attn_kw_ok = True
        for P, N_in in cands:
            B = P * N_in
            try:
                torch.cuda.empty_cache()
                x = torch.full((B, L), mask_id, dtype=torch.long, device="cuda")
                attn = torch.ones((B, L), dtype=torch.long, device="cuda")
                with torch.no_grad():
                    if attn_kw_ok:
                        try:
                            out = model(x, attention_mask=attn).logits
                        except TypeError:
                            attn_kw_ok = False
                            out = model(x).logits
                    else:
                        out = model(x).logits
                del x, attn, out
                torch.cuda.empty_cache()
                print(f"[H3] autotune: ✓ P={P}, N_inner={N_in} (B={B}) fits")
                return P, N_in
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[H3] autotune: ✗ P={P}, N_inner={N_in} (B={B}) OOM")
                continue
        print("[H3] autotune: nothing fits — fallback to (1, 1)")
        return 1, 1

    pb_arg = str(args.prompt_batch).strip().lower()
    if pb_arg in ("auto", "0"):
        P_BATCH, N_INNER = _autotune_batch()
    else:
        P_BATCH = max(1, int(pb_arg))
        N_INNER = N_TOTAL
    print(f"[H3] using P_BATCH={P_BATCH}, N_INNER={N_INNER}, "
          f"N_TOTAL={N_TOTAL}, B={P_BATCH*N_INNER}")

    # Probe attention_mask support once for the hot loop in
    # generate_batched_multi (avoids per-step try/except overhead).
    print("[H3] probing model.forward(attention_mask=...) ...")
    try:
        with torch.no_grad():
            _probe_x = torch.full((1, 16), mask_id, dtype=torch.long, device="cuda")
            _probe_a = torch.ones((1, 16), dtype=torch.long, device="cuda")
            _ = model(_probe_x, attention_mask=_probe_a).logits
            del _probe_x, _probe_a, _
        ATTN_MASK_OK = True
        print("[H3]   attention_mask supported ✓")
    except TypeError:
        ATTN_MASK_OK = False
        print("[H3]   attention_mask NOT supported — falling back to model(x)")
    torch.cuda.empty_cache()

    # Pre-tokenize all prompts to find their length, then sort each todo
    # list by length DESCENDING so each chunk has prompts of similar
    # length (minimizing pad → minimizing position-id shift for shorter
    # prompts in the chunk → cleaner generation distribution).
    def _prompt_len(rec):
        msgs = [{"role": "user", "content": rec["prompt"]}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return len(tok(text)["input_ids"])

    print("[H3] sorting todos by prompt length (reduces in-chunk padding)...")
    fail_lens = {i: _prompt_len(fail_prompts[i]) for i in fail_todo}
    ok_lens = {i: _prompt_len(ok_prompts[i]) for i in ok_todo}
    fail_todo = sorted(fail_todo, key=lambda i: -fail_lens[i])
    ok_todo = sorted(ok_todo, key=lambda i: -ok_lens[i])
    if fail_todo:
        print(f"[H3]   fail lens: max={max(fail_lens.values())} "
              f"min={min(fail_lens.values())} "
              f"med={sorted(fail_lens.values())[len(fail_lens)//2]}")
    if ok_todo:
        print(f"[H3]   ok lens:   max={max(ok_lens.values())} "
              f"min={min(ok_lens.values())} "
              f"med={sorted(ok_lens.values())[len(ok_lens)//2]}")

    def _build_temp_row(T, outs, gt):
        """Compute correct/answer lists + pass@k metrics for one (T, prompt) cell."""
        corrects = [is_correct(o, gt) for o in outs]
        answers  = [extract_answer(o) for o in outs]
        answers  = [None if a is None else float(a) for a in answers]
        n_s = args.n_samples
        return {"correct_list": [bool(c) for c in corrects],
                "answer_list": answers,
                "pass@1": pass_at_k(corrects, 1),
                "pass@4": pass_at_k(corrects, min(4, n_s)),
                "pass@8": pass_at_k(corrects, n_s)}

    def run_batch(group: str, idxs: list[int], recs: list[dict]):
        """Process P_BATCH prompts together (B = P × N_inner per forward).
        If N_INNER < N_TOTAL, iterate ceil(N_TOTAL/N_INNER) times collecting
        N_TOTAL samples per prompt."""
        prompts = [r["prompt"] for r in recs]
        gts     = [r["ground_truth"] for r in recs]
        rows = [{"group": group, "idx": i, "gt": g, "temps": {}}
                for i, g in zip(idxs, gts)]

        def _collect_samples(prompts, T, n_target):
            """Collect n_target samples per prompt via N_INNER-sized forwards."""
            collected = [[] for _ in prompts]
            iters = (n_target + N_INNER - 1) // N_INNER
            for _ in range(iters):
                outs = generate_batched_multi(
                    model, tok, prompts,
                    n_samples=N_INNER,
                    gen_length=args.gen_length, steps=args.steps,
                    block_length=args.block_length, temperature=T,
                    mask_id=mask_id,
                    _attn_mask_supported=ATTN_MASK_OK,
                )
                for p_idx, group_outs in enumerate(outs):
                    collected[p_idx].extend(group_outs)
            # trim to exactly n_target (last iter may overshoot)
            return [grp[:n_target] for grp in collected]

        for T in args.temps:
            if T == 0.0:
                # Deterministic: 1 sample suffices, replicate N times.
                outs_per_prompt = generate_batched_multi(
                    model, tok, prompts,
                    n_samples=1,
                    gen_length=args.gen_length, steps=args.steps,
                    block_length=args.block_length, temperature=0.0,
                    mask_id=mask_id,
                    _attn_mask_supported=ATTN_MASK_OK,
                )
                outs_per_prompt = [grp * N_TOTAL for grp in outs_per_prompt]
            else:
                outs_per_prompt = _collect_samples(prompts, T, N_TOTAL)
            for row, gt, outs in zip(rows, gts, outs_per_prompt):
                row["temps"][str(T)] = _build_temp_row(T, outs, gt)
        for i, row in zip(idxs, rows):
            rd.save_prompt_group(group, i, row)
        return rows

    def chunked(lst, k):
        for i in range(0, len(lst), k):
            yield lst[i:i+k]

    total = len(fail_todo) + len(ok_todo)
    pp = ProgressPrinter(total, tag="H3 ")
    for chunk in chunked(fail_todo, P_BATCH):
        rows = run_batch("fail", chunk, [fail_prompts[i] for i in chunk])
        for i, row in zip(chunk, rows):
            p8s = [row["temps"][str(T)]["pass@8"] for T in args.temps]
            pp.tick(f"FAIL[{i}] p@8={'/'.join(f'{x:.0f}' for x in p8s)}")
    for chunk in chunked(ok_todo, P_BATCH):
        rows = run_batch("ok", chunk, [ok_prompts[i] for i in chunk])
        for i, row in zip(chunk, rows):
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
