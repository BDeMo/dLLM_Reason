"""Download Qwen (or any HuggingFace) model to checkpoints/.

Thin wrapper around `huggingface_hub.snapshot_download` with progress bars,
resume-on-failure, and a sensible default candidate list for Qwen3 family
(used as the T6 AR-teacher in the v1.6 plan).

Storage:
    checkpoints/<org>__<model>/   (e.g. checkpoints/Qwen__Qwen3-8B/)
    — gitignored via existing 'checkpoints/' entry in .gitignore.

Usage:
    # List recommended candidates without downloading
    python scripts/download_qwen.py --list

    # Download default set (Qwen3-4B + Qwen3-8B + Qwen3-32B)
    python scripts/download_qwen.py

    # Pick specific sizes
    python scripts/download_qwen.py --sizes 4B,8B

    # Download a specific model id (bypasses the candidate list)
    python scripts/download_qwen.py --models Qwen/Qwen3-30B-A3B

    # Dry-run (show plan, no download)
    python scripts/download_qwen.py --dry_run

Notes:
    - Requires `pip install huggingface_hub`. Fast transfer via
      `pip install hf_transfer` + `HF_HUB_ENABLE_HF_TRANSFER=1`.
    - HF_TOKEN env var used automatically if set (needed for gated repos).
    - Auto-resumes on re-run if partially downloaded.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT_ROOT = ROOT / "checkpoints"


# ── Candidate model IDs (Qwen3 family) ───────────────────────────────────────
# Pinned to concrete HuggingFace repo names as of 2026-04. Adjust if the
# upstream repos are renamed.
CANDIDATES = {
    # size_label : hf_repo_id
    "4B":   "Qwen/Qwen3-4B",
    "8B":   "Qwen/Qwen3-8B",
    "14B":  "Qwen/Qwen3-14B",
    "30B":  "Qwen/Qwen3-30B-A3B",    # MoE, ~3B activated params
    "32B":  "Qwen/Qwen3-32B",
    # Qwen2.5 family (fallback if Qwen3 not available)
    "q25-7B":  "Qwen/Qwen2.5-7B-Instruct",
    "q25-32B": "Qwen/Qwen2.5-32B-Instruct",
    "q25-72B": "Qwen/Qwen2.5-72B-Instruct",
}

# Default download set — matches v1.6 plan's T6 teacher candidates + a small
# model for quick sanity checks.
DEFAULT_SIZES = ["4B", "8B", "32B"]


def local_dir_for(repo_id: str) -> Path:
    """Map 'Qwen/Qwen3-8B' → checkpoints/Qwen__Qwen3-8B/."""
    safe = repo_id.replace("/", "__")
    return CKPT_ROOT / safe


def download_one(repo_id: str, local_dir: Path, dry_run: bool = False,
                 allow_patterns: list[str] | None = None) -> None:
    """Resume-friendly snapshot download. HF hub handles retry/resume itself."""
    print(f"[DL] {repo_id}  →  {local_dir}")
    if dry_run:
        print(f"     (dry-run, no download)")
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[DL] ERROR: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=allow_patterns,  # None = all files
        token=os.environ.get("HF_TOKEN"),  # auto-pick if set
    )
    print(f"[DL] done: {local_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default=",".join(DEFAULT_SIZES),
                    help=f"comma-separated size labels from CANDIDATES. "
                         f"Available: {list(CANDIDATES)}. Default: "
                         f"{','.join(DEFAULT_SIZES)}")
    ap.add_argument("--models", type=str, default=None,
                    help="explicit HF repo id(s), comma-separated. "
                         "Overrides --sizes.")
    ap.add_argument("--list", action="store_true",
                    help="list candidates and exit")
    ap.add_argument("--dry_run", action="store_true",
                    help="print plan, no download")
    ap.add_argument("--skip_safetensors_only", action="store_true",
                    help="download ALL files including PyTorch .bin (default "
                         "skips .bin to save space since safetensors suffice)")
    args = ap.parse_args()

    if args.list:
        print("Candidate Qwen models:")
        for sz, rid in CANDIDATES.items():
            print(f"  {sz:<10} → {rid}")
        print(f"\nDefault set (--sizes {','.join(DEFAULT_SIZES)}):")
        for sz in DEFAULT_SIZES:
            print(f"  {sz:<10} → {CANDIDATES[sz]}")
        return

    # Resolve target repo ids
    if args.models:
        targets = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
        unknown = [s for s in sizes if s not in CANDIDATES]
        if unknown:
            print(f"[DL] ERROR: unknown sizes {unknown}. "
                  f"Use --list to see options.", file=sys.stderr)
            sys.exit(1)
        targets = [CANDIDATES[s] for s in sizes]

    print(f"[DL] checkpoints root: {CKPT_ROOT}")
    print(f"[DL] targets ({len(targets)}):")
    for r in targets:
        print(f"       - {r}")
    print()

    allow_patterns = None
    if not args.skip_safetensors_only:
        # skip .bin (PyTorch pickle) — usually duplicate of .safetensors
        allow_patterns = [
            "*.json", "*.txt", "*.md", "*.py",
            "*.safetensors", "*.model", "tokenizer*", "vocab*",
            "merges.txt", "chat_template*",
        ]

    for rid in targets:
        ld = local_dir_for(rid)
        download_one(rid, ld, dry_run=args.dry_run, allow_patterns=allow_patterns)

    # Summary
    if not args.dry_run:
        print()
        print("═" * 60)
        print("[DL] Download complete. Local paths:")
        for rid in targets:
            ld = local_dir_for(rid)
            if ld.exists():
                size_mb = sum(f.stat().st_size for f in ld.rglob("*") if f.is_file()) / 1e9
                print(f"       {rid}  →  {ld}  ({size_mb:.1f} GB)")
        print()
        print("Next steps:")
        print("  (a) Use as T6 teacher (local mode):")
        print("      python scripts/validate/t6_teacher_trace.py \\")
        print("          --teacher local \\")
        print("          --local_model checkpoints/Qwen__Qwen3-8B")
        print("  (b) Verify model loads:")
        print("      python -c \"from transformers import AutoTokenizer; "
              "AutoTokenizer.from_pretrained('checkpoints/Qwen__Qwen3-8B')\"")


if __name__ == "__main__":
    main()
