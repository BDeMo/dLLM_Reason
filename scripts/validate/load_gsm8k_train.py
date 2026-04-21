"""Load GSM8K train split from HuggingFace and convert to scope schema.

The v1.6 'Full' training pipeline (see docs/plans/2026-04-19_v1.6_plan.zh.md)
uses GSM8K train (7473 problems) as the source for T6 teacher traces and
T7 self-distill sampling. These are separate from the test set that our
60 fail + 49 ok scope already samples from — no train/test overlap.

Output: a JSON list with the same schema as runs/validation/scope_*.json
so the T6/T7 scripts can read it directly via --scope_path.

Supports HF mirror selection for users behind GFW (hf-mirror.com etc.).

Usage:
    python scripts/validate/load_gsm8k_train.py
        → runs/validation/gsm8k_train_prompts.json

    python scripts/validate/load_gsm8k_train.py \\
        --mirror hf-mirror \\
        --max_samples 2000

    python scripts/validate/load_gsm8k_train.py --check_only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "runs" / "validation" / "gsm8k_train_prompts.json"

# Mirror map (same as download_qwen.py for consistency)
MIRRORS = {
    "default":    "https://huggingface.co",
    "hf-mirror":  "https://hf-mirror.com",
    "modelscope": "https://www.modelscope.cn",
}


def apply_mirror(mirror: str | None) -> str:
    if not mirror or mirror == "default":
        return MIRRORS["default"]
    endpoint = MIRRORS.get(mirror, mirror)
    if not (endpoint.startswith("http://") or endpoint.startswith("https://")):
        print(f"[MIRROR] ERROR: invalid mirror {mirror!r}.", file=sys.stderr)
        sys.exit(1)
    os.environ["HF_ENDPOINT"] = endpoint
    # datasets library also checks HF_DATASETS_ENDPOINT; set both for safety
    os.environ.setdefault("HF_DATASETS_ENDPOINT", endpoint)
    print(f"[MIRROR] HF_ENDPOINT = {endpoint}")
    return endpoint


# ── GSM8K answer extraction ──────────────────────────────────────────────────
# Official GSM8K answers look like: "...reasoning...#### 72"
# We extract the number after '####' as ground_truth.
ANSWER_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


def extract_gt(answer_text: str) -> str | None:
    m = ANSWER_RE.search(answer_text or "")
    return m.group(1) if m else None


def check_output_file(path: Path) -> tuple[bool, list[str]]:
    """Verify the produced JSON file looks correct."""
    issues: list[str] = []
    if not path.exists():
        return False, [f"file does not exist: {path}"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, [f"not valid JSON: {e}"]
    if not isinstance(data, list):
        issues.append("top-level not a list")
        return False, issues
    if len(data) == 0:
        issues.append("empty list")
        return False, issues
    # Schema spot check
    required_keys = {"prompt", "ground_truth"}
    for i, item in enumerate(data[:5]):
        missing = required_keys - set(item.keys())
        if missing:
            issues.append(f"item[{i}] missing keys: {sorted(missing)}")
    if any(not str(item.get("ground_truth", "")).strip() for item in data[:20]):
        issues.append("some items have empty ground_truth")
    return (len(issues) == 0), issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="train",
                    choices=["train", "test"],
                    help="gsm8k split to load (default train)")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="cap #samples (default = full split)")
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--mirror", type=str, default=None,
                    help="HF endpoint: default / hf-mirror / modelscope / URL")
    ap.add_argument("--offline", action="store_true",
                    help="Force HF cache-only (HF_DATASETS_OFFLINE=1). Use "
                         "after a prior successful download to avoid re-pinging "
                         "the hub on flaky-mirror days.")
    ap.add_argument("--check_only", action="store_true",
                    help="only verify existing output file; skip download")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-download even if output file exists")
    args = ap.parse_args()

    out_path = Path(args.output)

    if args.check_only:
        ok, issues = check_output_file(out_path)
        print(f"{'✓' if ok else '✗'} {out_path}")
        for iss in issues:
            print(f"    - {iss}")
        sys.exit(0 if ok else 1)

    if out_path.exists() and not args.overwrite:
        print(f"[GSM8K] Output already exists: {out_path}")
        ok, issues = check_output_file(out_path)
        if ok:
            data = json.loads(out_path.read_text(encoding="utf-8"))
            print(f"[GSM8K] Verified ✓ ({len(data)} items). "
                  f"Use --overwrite to re-download.")
            return
        print(f"[GSM8K] Existing file failed verification — re-downloading")
        for iss in issues:
            print(f"    - {iss}")

    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        print(f"[GSM8K] HF_DATASETS_OFFLINE=1 — using HF cache only")
    else:
        apply_mirror(args.mirror)

    # Try project-registered local-first resolver (datasets/gsm8k/<split>/)
    # before falling back to HF download. Honors MODEL_REGISTRY in
    # src/dllm_reason/utils/resource_registry.py.
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from dllm_reason.utils.local_resolve import resolve_dataset
        print(f"[GSM8K] resolving via project registry "
              f"(datasets/gsm8k/{args.split}/ first; HF fallback)")
        ds = resolve_dataset("openai/gsm8k", config="main", split=args.split)
    except ImportError:
        from datasets import load_dataset
        print(f"[GSM8K] dllm_reason not importable; loading via HF directly")
        ds = load_dataset("openai/gsm8k", "main", split=args.split)

    print(f"[GSM8K] loaded: {len(ds)} items")

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
        print(f"[GSM8K] capped to {len(ds)} items")

    out: list[dict] = []
    n_skipped = 0
    for i, item in enumerate(ds):
        gt = extract_gt(item["answer"])
        if gt is None:
            n_skipped += 1
            continue
        out.append({
            "episode_id": f"gsm8k_{args.split}_{i:05d}",
            "prompt": item["question"],
            "ground_truth": gt,
            "full_answer": item["answer"],        # preserve for reference
            "source_split": args.split,
            "source_idx": i,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print()
    print("═" * 60)
    print(f"[GSM8K] written: {out_path}")
    print(f"[GSM8K] items:   {len(out)}")
    print(f"[GSM8K] skipped: {n_skipped} (no '####' answer marker)")
    print()
    print("Verifying output…")
    ok, issues = check_output_file(out_path)
    if ok:
        print(f"[GSM8K] VERIFIED ✓")
        print()
        print("Next steps:")
        print(f"  T7 data gen on this set:")
        print(f"    python scripts/validate/t7_gen_correct_samples.py \\")
        print(f"        --scope_path {out_path} --groups gsm8k \\")
        print(f"        --temperatures 0.7 --n_samples 8")
        print(f"  T6 teacher on this set:")
        print(f"    python scripts/validate/t6_teacher_trace.py \\")
        print(f"        --scope_path {out_path} --groups gsm8k \\")
        print(f"        --teacher local --local_model checkpoints/Qwen__Qwen3.5-9B-Instruct")
    else:
        print(f"[GSM8K] verification FAILED")
        for iss in issues:
            print(f"    - {iss}")
        sys.exit(1)


if __name__ == "__main__":
    main()
