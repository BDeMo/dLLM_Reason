"""Regenerate scope_fail_prompts.json + scope_ok_prompts.json from scratch.

Background:
    The original scope files in runs/validation/ came from an old
    stage-2 research DB and may have drifted from the current model +
    canonical config. To get an accurate "baseline fails" / "baseline
    succeeds" split, we re-run the current LLaDA-8B-Instruct ckpt on
    the gsm8k test split under the canonical config and partition.

Canonical config (per docs/archive/ablation_index.zh.md § Setting):
    temperature       = 0
    block_length      = 32
    gen_length        = 128
    num_steps         = 128  (coupled)
    remasking         = low_confidence
    prompt template   = bare question (no prefix)

This script talks to a running LLaDA serve.py (same pattern as
t7_gen_correct_samples.py) and writes both scope files in the same
schema that downstream scripts expect (prompt, ground_truth, output,
correct, num_steps, block_length, dag_seq_len=gen_length).

Usage:
    # start serve first
    CUDA_VISIBLE_DEVICES=0 python scripts/serve.py --port 8000 &

    # regen on full gsm8k test (1319 prompts; ~110 min on single A100)
    python scripts/validate/regen_scope.py

    # or on a subset for iteration
    python scripts/validate/regen_scope.py --max_prompts 200

    # write to custom paths
    python scripts/validate/regen_scope.py \\
        --fail_out runs/validation/scope_fail_v2.json \\
        --ok_out   runs/validation/scope_ok_v2.json

Multi-GPU option: use --prompt_start / --prompt_end for sharding (similar
pattern to strategy_search.py); final aggregate pass merges.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from _runlib import RunDir, ProgressPrinter, add_common_args, resolve_run_dir
from _http_client import ValidationAPIClient, add_server_arg

sys.path.insert(0, str(Path(__file__).parent))
from h1_remask_rescue import is_correct
from load_gsm8k_train import apply_mirror, extract_gt as _gsm_extract_gt

ROOT = Path(__file__).resolve().parents[2]
OUT_BASE = ROOT / "runs" / "validation"
DEFAULT_FAIL = OUT_BASE / "scope_fail_prompts.json"
DEFAULT_OK = OUT_BASE / "scope_ok_prompts.json"


def load_gsm8k_test(max_prompts: int | None, mirror: str | None,
                    local_path: str | None = None,
                    offline: bool = False) -> list[dict]:
    """Load gsm8k test prompts.

    Resolution order (network-cheapest first):
      1. ``local_path`` JSON (e.g. runs/validation/gsm8k_test_prompts.json) —
         no HF call, fully offline. PREFERRED for re-runs / mirror outages.
      2. HF cache only (``offline=True``): set HF_DATASETS_OFFLINE=1 so
         load_dataset reuses the prior download without verifying
         the hub. Fails if dataset isn't already cached locally.
      3. HF + mirror (default): apply HF_ENDPOINT, call load_dataset.
    """
    # Path 1: explicit local JSON (no HF involvement)
    if local_path and Path(local_path).is_file():
        print(f"[REGEN] loading gsm8k test from local JSON: {local_path}")
        data = json.loads(Path(local_path).read_text(encoding="utf-8"))
        if max_prompts:
            data = data[:max_prompts]
        out = []
        for i, item in enumerate(data):
            # Accept either {prompt, ground_truth} (our scope schema) or
            # {question, answer} (raw HF schema).
            if "prompt" in item and "ground_truth" in item:
                out.append({
                    "source_idx": item.get("source_idx", i),
                    "prompt": item["prompt"],
                    "ground_truth": item["ground_truth"],
                })
            else:
                gt = _gsm_extract_gt(item.get("answer", ""))
                if gt is None:
                    continue
                out.append({
                    "source_idx": i,
                    "prompt": item["question"],
                    "ground_truth": gt,
                })
        print(f"[REGEN] loaded {len(out)} prompts from local")
        return out

    # Path 2 + 3: HF, with optional offline / mirror
    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        print(f"[REGEN] HF_DATASETS_OFFLINE=1 — using HF cache only")
    else:
        apply_mirror(mirror)

    # Project-first: prefer datasets/gsm8k/test/ (registered local cache)
    # before falling back to HF download.
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from dllm_reason.utils.local_resolve import resolve_dataset
        print(f"[REGEN] resolving via project registry "
              f"(datasets/gsm8k/test/ first)")
        ds = resolve_dataset("openai/gsm8k", config="main", split="test")
    except ImportError:
        try:
            from datasets import load_dataset
        except ImportError:
            print("[REGEN] ERROR: pip install datasets", file=sys.stderr)
            sys.exit(1)
        ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"[REGEN] gsm8k test loaded: {len(ds)}")
    if max_prompts:
        ds = ds.select(range(min(max_prompts, len(ds))))
        print(f"[REGEN] capped to {len(ds)}")
    out = []
    for i, item in enumerate(ds):
        gt = _gsm_extract_gt(item["answer"])
        if gt is None:
            continue
        out.append({
            "source_idx": i,
            "prompt": item["question"],
            "ground_truth": gt,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    # Canonical baseline config (DO NOT change without re-creating scope)
    ap.add_argument("--gen_length", type=int, default=128,
                    help="canonical scope config; do not change")
    ap.add_argument("--block_length", type=int, default=32,
                    help="canonical scope config; do not change")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="canonical scope config; do not change")
    ap.add_argument("--num_steps", type=int, default=None,
                    help="default = gen_length (coupled per E1 finding)")
    ap.add_argument("--remasking", type=str, default="low_confidence")
    # Scope
    ap.add_argument("--max_prompts", type=int, default=None,
                    help="cap #prompts (default: full gsm8k test = 1319)")
    ap.add_argument("--mirror", type=str, default=None,
                    help="HF endpoint: default / hf-mirror / modelscope / URL")
    ap.add_argument("--gsm8k_test_path", type=str, default=None,
                    help="(rarely needed) Bypass resolver, load gsm8k test from "
                         "this local JSON. Use only if datasets/gsm8k/test/ is "
                         "unavailable and you have a custom prompt list.")
    ap.add_argument("--offline", action="store_true",
                    help="(rarely needed) Force HF cache-only mode. Normally "
                         "the project's resolve_dataset() prefers local "
                         "datasets/gsm8k/test/ already; this flag is only "
                         "useful if HF cache verification is causing trouble.")
    # Sharding
    ap.add_argument("--prompt_start", type=int, default=None)
    ap.add_argument("--prompt_end", type=int, default=None)
    ap.add_argument("--skip_aggregate", action="store_true",
                    help="shard workers set this; run one final pass without "
                         "it to aggregate scope_*.json from per_prompt/*")
    # Output paths
    ap.add_argument("--fail_out", type=str, default=str(DEFAULT_FAIL))
    ap.add_argument("--ok_out", type=str, default=str(DEFAULT_OK))
    add_common_args(ap)
    add_server_arg(ap)
    args = ap.parse_args()

    if args.num_steps is None:
        args.num_steps = args.gen_length

    prompts_all = load_gsm8k_test(
        args.max_prompts, args.mirror,
        local_path=args.gsm8k_test_path,
        offline=args.offline,
    )

    # Apply shard slice for work (but always aggregate from all)
    if args.prompt_start is not None or args.prompt_end is not None:
        s = args.prompt_start if args.prompt_start is not None else 0
        e = args.prompt_end if args.prompt_end is not None else len(prompts_all)
        prompts = prompts_all[s:e]
        print(f"[REGEN] shard slice: prompts[{s}:{e}] = {len(prompts)} / "
              f"{len(prompts_all)} total")
    else:
        prompts = prompts_all

    run_dir = resolve_run_dir(args, "scope_regen", OUT_BASE)
    rd = RunDir(
        run_dir, "ScopeRegen",
        config={**vars(args), "n_total": len(prompts_all),
                "n_shard": len(prompts)},
        resume=args.resume,
    )
    print(f"[REGEN] run_dir = {rd.dir}")
    print(f"[REGEN] config: T={args.temperature} bl={args.block_length} "
          f"g={args.gen_length} steps={args.num_steps} "
          f"rem={args.remasking}")

    def prompt_key(i: int) -> str:
        return f"t{i:04d}"

    def is_done(key: str) -> bool:
        return (rd.per_prompt / f"{key}.json").exists()

    todo = [(i, p) for (i, p) in enumerate(prompts) if not is_done(prompt_key(i))]
    print(f"[REGEN] done={len(prompts) - len(todo)}  todo={len(todo)}")

    if args.dry_run:
        print(f"[REGEN] DRY RUN — would evaluate {len(todo)} prompts "
              f"× 1 sample each")
        return

    api = ValidationAPIClient(args.server_url)
    api.check_health()

    pp = ProgressPrinter(len(todo), tag="REGEN ")
    for i, rec in todo:
        prompt, gt = rec["prompt"], rec["ground_truth"]
        key = prompt_key(i)
        t0 = time.time()
        try:
            out = api.generate(
                prompt, strategy="confidence",
                max_new_tokens=args.gen_length,
                num_steps=args.num_steps,
                block_length=args.block_length,
                temperature=args.temperature,
                remasking=args.remasking,
            )
        except Exception as e:
            print(f"[REGEN] WARN {key}: {e}")
            out = ""
        dt = time.time() - t0
        correct = bool(is_correct(out, gt))
        out_rec = {
            "source_idx": rec["source_idx"],
            "prompt": prompt,
            "ground_truth": gt,
            "output": out,
            "correct": correct,
            "num_steps": args.num_steps,
            "block_length": args.block_length,
            "dag_seq_len": args.gen_length,
            "elapsed_s": dt,
        }
        path = rd.per_prompt / f"{key}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out_rec, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, path)
        pp.tick(f"{key} {'✓' if correct else '✗'}")

    if args.skip_aggregate:
        print(f"[REGEN] --skip_aggregate set; shard done.")
        return

    # ── Aggregate: split all per_prompt into fail + ok ───────────────────────
    all_recs = []
    for i in range(len(prompts_all)):
        p = rd.per_prompt / f"{prompt_key(i)}.json"
        if p.exists():
            all_recs.append(json.loads(p.read_text(encoding="utf-8")))

    fail_items = []
    ok_items = []
    for r in all_recs:
        base = {
            "episode_id": f"gsm8k_test_{r['source_idx']:05d}",
            "prompt": r["prompt"],
            "ground_truth": r["ground_truth"],
            "output": r["output"],
            "correct": r["correct"],
            "dag_seq_len": r["dag_seq_len"],
            "num_steps": r["num_steps"],
            "block_length": r["block_length"],
        }
        if r["correct"]:
            ok_items.append(base)
        else:
            base["error_type"] = "regen"  # placeholder; full bucket taxonomy TBD
            fail_items.append(base)

    # Write canonical scope paths (overwrite)
    Path(args.fail_out).write_text(
        json.dumps(fail_items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(args.ok_out).write_text(
        json.dumps(ok_items, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "n_total": len(all_recs),
        "n_fail": len(fail_items),
        "n_ok": len(ok_items),
        "pass_rate": len(ok_items) / max(len(all_recs), 1),
        "fail_out": str(args.fail_out),
        "ok_out": str(args.ok_out),
        "config": {
            "temperature": args.temperature,
            "block_length": args.block_length,
            "gen_length": args.gen_length,
            "num_steps": args.num_steps,
            "remasking": args.remasking,
        },
    }
    rd.write_summary(summary)

    print()
    print("═" * 60)
    print(f"[REGEN] total:    {len(all_recs)}")
    print(f"[REGEN] pass:     {len(ok_items)} "
          f"({len(ok_items)/max(len(all_recs),1):.2%})")
    print(f"[REGEN] fail:     {len(fail_items)}")
    print(f"[REGEN] fail →    {args.fail_out}")
    print(f"[REGEN] ok →      {args.ok_out}")
    print(f"[REGEN] summary → {rd.summary_path}")


if __name__ == "__main__":
    main()
