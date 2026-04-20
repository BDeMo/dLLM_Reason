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

# ── HF mirror support ────────────────────────────────────────────────────────
MIRRORS = {
    "default": "https://huggingface.co",           # official
    "hf-mirror": "https://hf-mirror.com",           # community mirror (China)
    "modelscope": "https://www.modelscope.cn",      # ModelScope gateway (caveat:
                                                    # not drop-in, model IDs differ)
}


def apply_mirror(mirror: str | None) -> str:
    """Set HF_ENDPOINT env var so huggingface_hub + datasets routes via mirror.

    Called BEFORE any `import huggingface_hub` / `from datasets import ...`
    in the current process; env var is the canonical toggle.
    """
    if not mirror or mirror == "default":
        return MIRRORS["default"]
    endpoint = MIRRORS.get(mirror, mirror)  # also accept raw URL
    if not (endpoint.startswith("http://") or endpoint.startswith("https://")):
        print(f"[MIRROR] ERROR: invalid mirror {mirror!r}. Use 'default', "
              f"'hf-mirror', 'modelscope', or a full URL.", file=sys.stderr)
        sys.exit(1)
    os.environ["HF_ENDPOINT"] = endpoint
    print(f"[MIRROR] HF_ENDPOINT = {endpoint}")
    return endpoint


# ── Candidate model IDs ──────────────────────────────────────────────────────
# Pinned to concrete HuggingFace repo names. If a repo is renamed upstream,
# override with --models <Qwen/...> explicitly. Many repos have *-Instruct
# and *-Base variants; the Instruct variant is usually what we want for T6
# teacher (follows the <SETUP>/<STEP_x>/<ANSWER> format instruction).
CANDIDATES = {
    # Qwen3.5 family (target of v1.6 T6 teacher)
    "3.5-4B":    "Qwen/Qwen3.5-4B-Instruct",
    "3.5-9B":    "Qwen/Qwen3.5-9B-Instruct",
    "3.5-27B":   "Qwen/Qwen3.5-27B-Instruct",
    # Qwen3 family (likely fallback if Qwen3.5 repos not yet up)
    "3-4B":      "Qwen/Qwen3-4B",
    "3-8B":      "Qwen/Qwen3-8B",
    "3-14B":     "Qwen/Qwen3-14B",
    "3-30B":     "Qwen/Qwen3-30B-A3B",
    "3-32B":     "Qwen/Qwen3-32B",
    # Qwen2.5 family (further fallback)
    "2.5-7B":    "Qwen/Qwen2.5-7B-Instruct",
    "2.5-32B":   "Qwen/Qwen2.5-32B-Instruct",
    "2.5-72B":   "Qwen/Qwen2.5-72B-Instruct",
}

# Default download set — matches user request 4B/9B/27B in Qwen3.5 family.
DEFAULT_SIZES = ["3.5-4B", "3.5-9B", "3.5-27B"]


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


# ── Post-download file integrity check ───────────────────────────────────────

REQUIRED_PATTERNS = [
    "config.json",
    "tokenizer.json",           # most modern tokenizers
]
OPTIONAL_BUT_EXPECTED = [
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
]


def check_downloaded(local_dir: Path, min_weights_gb: float = 1.0) -> tuple[bool, list[str]]:
    """Verify a snapshot_download target: required files present, at least
    one non-empty weight file, total weight size >= min_weights_gb.

    Returns (ok, issues) — `issues` is a list of human-readable problems.
    """
    issues: list[str] = []
    if not local_dir.exists():
        return False, [f"directory does not exist: {local_dir}"]

    # required files
    for patt in REQUIRED_PATTERNS:
        matches = list(local_dir.glob(patt))
        if not matches:
            issues.append(f"missing: {patt}")
        elif any(m.stat().st_size == 0 for m in matches):
            issues.append(f"zero-size: {patt}")

    # weight files (safetensors or pytorch bin)
    weight_files = (list(local_dir.glob("*.safetensors"))
                    + list(local_dir.glob("*.bin")))
    if not weight_files:
        issues.append("no *.safetensors or *.bin weight files")
    else:
        total_bytes = sum(f.stat().st_size for f in weight_files)
        total_gb = total_bytes / 1e9
        if total_gb < min_weights_gb:
            issues.append(
                f"weight size too small: {total_gb:.2f} GB "
                f"(expected >= {min_weights_gb:.1f} GB)"
            )

    return (len(issues) == 0), issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default=",".join(DEFAULT_SIZES),
                    help=f"comma-separated size labels from CANDIDATES. "
                         f"Available: {list(CANDIDATES)}. Default: "
                         f"{','.join(DEFAULT_SIZES)}")
    ap.add_argument("--models", type=str, default=None,
                    help="explicit HF repo id(s), comma-separated. "
                         "Overrides --sizes.")
    ap.add_argument("--mirror", type=str, default=None,
                    help="HF endpoint override. Options: 'default' (huggingface.co), "
                         "'hf-mirror' (hf-mirror.com, China), 'modelscope' "
                         "(modelscope.cn — note: model IDs differ), or a raw URL. "
                         "Sets HF_ENDPOINT env var for this process.")
    ap.add_argument("--list", action="store_true",
                    help="list candidates and exit")
    ap.add_argument("--dry_run", action="store_true",
                    help="print plan, no download")
    ap.add_argument("--skip_safetensors_only", action="store_true",
                    help="download ALL files including PyTorch .bin (default "
                         "skips .bin to save space since safetensors suffice)")
    ap.add_argument("--check_only", action="store_true",
                    help="only verify existing local dirs, no download")
    ap.add_argument("--min_weights_gb", type=float, default=1.0,
                    help="minimum expected total weight size in GB (per model). "
                         "Downgrade for small models like Qwen3-4B (~8GB).")
    args = ap.parse_args()

    # Apply mirror BEFORE importing huggingface_hub (env var must be set early)
    apply_mirror(args.mirror)

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

    if args.check_only:
        print("[DL] --check_only: verifying local dirs without download")
        all_ok = True
        for rid in targets:
            ld = local_dir_for(rid)
            ok, issues = check_downloaded(ld, min_weights_gb=args.min_weights_gb)
            status = "✓" if ok else "✗"
            print(f"  {status} {rid}  ({ld})")
            for iss in issues:
                print(f"      - {iss}")
            all_ok = all_ok and ok
        sys.exit(0 if all_ok else 1)

    for rid in targets:
        ld = local_dir_for(rid)
        download_one(rid, ld, dry_run=args.dry_run, allow_patterns=allow_patterns)

    # Summary + verification
    if not args.dry_run:
        print()
        print("═" * 60)
        print("[DL] Download complete. Verifying files…")
        all_ok = True
        for rid in targets:
            ld = local_dir_for(rid)
            if not ld.exists():
                print(f"  ✗ {rid}: dir missing {ld}")
                all_ok = False
                continue
            size_gb = sum(f.stat().st_size for f in ld.rglob("*") if f.is_file()) / 1e9
            ok, issues = check_downloaded(ld, min_weights_gb=args.min_weights_gb)
            status = "✓" if ok else "✗"
            print(f"  {status} {rid}  →  {ld}  ({size_gb:.1f} GB)")
            for iss in issues:
                print(f"      - {iss}")
            all_ok = all_ok and ok

        print()
        if all_ok:
            print("[DL] ALL MODELS VERIFIED ✓")
            print()
            example = targets[0]
            ld = local_dir_for(example)
            print("Next steps:")
            print(f"  (a) Use as T6 teacher (local mode):")
            print(f"      python scripts/validate/t6_teacher_trace.py \\")
            print(f"          --teacher local --local_model {ld}")
            print(f"  (b) Verify model loads in transformers:")
            print(f"      python -c \"from transformers import AutoTokenizer; "
                  f"AutoTokenizer.from_pretrained('{ld}')\"")
        else:
            print("[DL] Some models FAILED verification. Re-run to resume, "
                  "or inspect issues above.")
            sys.exit(1)


if __name__ == "__main__":
    main()
