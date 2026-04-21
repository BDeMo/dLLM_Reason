"""v1.6 evaluation — T=0 greedy pass@1 on held-out scope_fail + scope_ok.

Loads a trained LLaDA checkpoint (HF format dir, produced by t6t7_train.py's
HF-export step), runs deterministic greedy generation on each held-out
test prompt, and reports:

  - Pass@1 at T=0 (fail subset)
  - Pass@1 at T=0 (ok subset, regression guard)
  - FAIL18 rescue count + which indices broken
  - Ceiling 5 break count
  - Per-prompt delta vs the baseline LLaDA-Instruct numbers we have on file

Usage:
    # eval a single ckpt
    python scripts/validate/v16_eval.py \\
        --ckpt runs/training/v16_t7_stage1/hf \\
        --label t7_stage1

    # eval multiple ckpts, comparison table
    python scripts/validate/v16_eval.py \\
        --ckpts \\
            'baseline=GSAI-ML/LLaDA-8B-Instruct' \\
            't7=runs/training/v16_t7_stage1/hf' \\
            't6=runs/training/v16_t6_stage2/hf'

Output:
    runs/validation/v16_eval_<ts>/
      per_prompt/{fail|ok}_{idx}.json
      summary.json           aggregated numbers
      comparison.md          markdown table ckpt × metric
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "validate"))

SCOPE_FAIL = ROOT / "runs" / "validation" / "scope_fail_prompts.json"
SCOPE_OK = ROOT / "runs" / "validation" / "scope_ok_prompts.json"
OUT_BASE = ROOT / "runs" / "validation"

# Canonical named sets (same source as strategy_search.py)
FAIL18 = [0, 4, 5, 8, 10, 13, 14, 15, 19, 28, 35, 41, 42, 48, 51, 53, 55, 59]
CEILING5 = [4, 5, 14, 41, 42]


def _lazy_imports():
    """Import heavy deps lazily so --help / dry paths don't need torch."""
    global torch, LLaDAWrapper, is_correct
    import torch as _torch
    from dllm_reason.models.llada import LLaDAWrapper as _LW
    from h1_remask_rescue import is_correct as _ic
    torch = _torch
    LLaDAWrapper = _LW
    is_correct = _ic


def eval_ckpt(ckpt_path: str, out_dir: Path, label: str,
              gen_length: int, block_length: int,
              temperature: float, max_new_tokens: int | None = None) -> dict:
    """Load ckpt + eval on fail + ok scope + return aggregated stats."""
    _lazy_imports()
    print(f"[EVAL] {label}  →  {ckpt_path}")
    print(f"[EVAL] loading model ...")
    # max_seq_len needs room for prompt + gen_length
    model = LLaDAWrapper(model_id=ckpt_path, max_seq_len=max(512, gen_length + 256))
    # LLaDAWrapper wraps HF model as self._llada; set eval mode on the inner
    # model. Calling .eval() on the wrapper itself also cascades to submodules
    # via nn.Module semantics.
    model.eval()

    fails = json.loads(SCOPE_FAIL.read_text(encoding="utf-8"))
    oks = json.loads(SCOPE_OK.read_text(encoding="utf-8"))
    results = {"fail": [], "ok": []}

    max_new = max_new_tokens or gen_length

    # Sanity check: one generate() call before the main loop so we fail loud
    # if signature / setup is broken (previously silently swallowed exceptions
    # and returned empty strings for every prompt).
    print(f"[EVAL] sanity check: single generate on 'What is 2+2?' ...")
    try:
        sanity_out = model.generate(
            "What is 2+2?",
            generation_len=32, block_length=32, num_steps=32,
            temperature=0.0, remasking="low_confidence",
        )
        print(f"[EVAL] sanity ✓ ({len(sanity_out)} chars returned)")
    except Exception as e:
        print(f"[EVAL] FATAL sanity failure: {e!r}", file=sys.stderr)
        raise

    for group, prompts in [("fail", fails), ("ok", oks)]:
        pp_dir = out_dir / "per_prompt"
        pp_dir.mkdir(parents=True, exist_ok=True)
        print(f"[EVAL] {label}  group={group}  n={len(prompts)} ...")
        for i, rec in enumerate(prompts):
            prompt = rec["prompt"]
            gt = rec.get("ground_truth") or rec.get("gt")
            t0 = time.time()
            try:
                # LLaDAWrapper.generate signature: generation_len (not
                # max_new_tokens), remasking (not strategy); prior v1.6
                # draft used FastAPI-style kwargs which silently broke.
                out = model.generate(
                    prompt,
                    generation_len=max_new,
                    num_steps=max_new,
                    block_length=block_length,
                    temperature=temperature,
                    remasking="low_confidence",
                )
            except Exception as e:
                # Fail fast on the very first per-prompt error: previously
                # repeated exceptions produced an all-zero comparison.md
                # that looked like model failure rather than eval bug.
                if group == "fail" and i == 0:
                    print(f"[EVAL] FATAL first-prompt failure: {e!r}",
                          file=sys.stderr)
                    raise
                print(f"[EVAL] WARN {group}_{i}: {e!r}")
                out = ""
            dt = time.time() - t0
            correct = bool(is_correct(out, gt))
            row = {
                "group": group, "idx": i, "gt": gt,
                "prompt": prompt,
                "output": out,
                "correct": correct,
                "elapsed_s": dt,
            }
            results[group].append(row)
            (pp_dir / f"{label}__{group}_{i:04d}.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if (i + 1) % 10 == 0:
                n_ok = sum(r["correct"] for r in results[group])
                print(f"[EVAL] {label}  {group}  {i+1}/{len(prompts)}  "
                      f"acc={n_ok}/{i+1}")

    fail_correct = sum(r["correct"] for r in results["fail"])
    ok_correct = sum(r["correct"] for r in results["ok"])
    fail_rescued_idx = [r["idx"] for r in results["fail"] if r["correct"]]
    fail18_rescued = sorted(set(fail_rescued_idx) & set(FAIL18))
    ceiling_broken = sorted(set(fail_rescued_idx) & set(CEILING5))

    stats = {
        "label": label,
        "ckpt": ckpt_path,
        "n_fail": len(results["fail"]),
        "n_ok": len(results["ok"]),
        "fail_pass@1": fail_correct / max(len(results["fail"]), 1),
        "ok_pass@1": ok_correct / max(len(results["ok"]), 1),
        "fail_correct": fail_correct,
        "ok_correct": ok_correct,
        "fail_rescued_idx": fail_rescued_idx,
        "fail18_rescued": fail18_rescued,
        "fail18_rescued_count": len(fail18_rescued),
        "ceiling_broken": ceiling_broken,
        "ceiling_broken_count": len(ceiling_broken),
        "config": {
            "gen_length": gen_length,
            "block_length": block_length,
            "temperature": temperature,
        },
    }

    # Free GPU for next ckpt
    del model
    import gc; gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return stats


def parse_ckpt_spec(spec: str) -> tuple[str, str]:
    """'label=path' → (label, path). Plain path → (basename, path)."""
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), path.strip()
    return Path(spec).name or "ckpt", spec


def render_comparison_md(all_stats: list[dict]) -> str:
    lines = ["# v1.6 Eval Comparison\n"]
    lines.append("| Label | fail pass@1 | ok pass@1 | FAIL18 rescued | ceiling broken |")
    lines.append("|---|---|---|---|---|")
    for s in all_stats:
        lines.append(
            f"| {s['label']} "
            f"| {s['fail_pass@1']:.2%} ({s['fail_correct']}/{s['n_fail']}) "
            f"| {s['ok_pass@1']:.2%} ({s['ok_correct']}/{s['n_ok']}) "
            f"| {s['fail18_rescued_count']}/18  `{s['fail18_rescued']}` "
            f"| {s['ceiling_broken_count']}/5  `{s['ceiling_broken']}` |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None,
                    help="single ckpt: HF dir path or HF repo id")
    ap.add_argument("--label", type=str, default="ckpt")
    ap.add_argument("--ckpts", type=str, nargs="+", default=None,
                    help="multiple ckpts in 'label=path' form")
    ap.add_argument("--gen_length", type=int, default=192)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out_dir", type=str, default=None,
                    help="default: runs/validation/v16_eval_<ts>")
    args = ap.parse_args()

    # Resolve list of (label, path) to eval
    if args.ckpts:
        targets = [parse_ckpt_spec(s) for s in args.ckpts]
    elif args.ckpt:
        targets = [(args.label, args.ckpt)]
    else:
        print("[EVAL] ERROR: provide --ckpt or --ckpts", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else \
        OUT_BASE / f"v16_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[EVAL] out_dir = {out_dir}")
    print(f"[EVAL] targets: {targets}")

    all_stats = []
    for label, path in targets:
        s = eval_ckpt(
            path, out_dir, label,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
        )
        all_stats.append(s)
        # write incremental summary so partial runs still produce output
        (out_dir / "summary.json").write_text(
            json.dumps({"ckpts": all_stats, "config": vars(args)},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Final comparison markdown
    md = render_comparison_md(all_stats)
    (out_dir / "comparison.md").write_text(md, encoding="utf-8")
    print()
    print("═" * 60)
    print(md)
    print()
    print(f"[EVAL] summary   → {out_dir / 'summary.json'}")
    print(f"[EVAL] comparison → {out_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
