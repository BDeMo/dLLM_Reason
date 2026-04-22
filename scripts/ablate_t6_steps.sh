#!/usr/bin/env bash
# ablate_t6_steps.sh — ablation over T6 SFT max_steps to find the
# fail-rescue vs ok-retention Pareto sweet spot.
#
# Context: v1.6.1 T6 SFT at default max_steps=2000 on a 1350-sample
# train split (effective batch = 1 × 16 × 8 = 128; backward every 8
# samples → ~12 epochs) overfits: fail +26.6% but ok -27%, net -179.
#
# Physical meaning of `steps` (global_step in Finetuner):
#   - 1 step   = 1 forward + 1 backward (grad_accum'd)
#   - samples  = steps × batch_size × world_size   (= steps × 8)
#   - epoch    = steps × 8 / 1350  (~169 steps per epoch under current cfg)
#
# Default ablation range covers ~0.6 / 1.2 / 2.4 / 4.7 epochs.
#
# For each step count S in STEPS_LIST:
#   1. wipe runs/training/v161_t6
#   2. run Phase 4 (SFT with --t6_max_steps S) + Phase 5 (eval)
#   3. rename eval dir → runs/validation/ablate_t6_steps/steps_S
#   4. collect summary.json from each run
# Final: write runs/validation/ablate_t6_steps/summary.md with one row per S.
#
# Usage:
#   bash scripts/ablate_t6_steps.sh                      # default 100 200 400 800
#   bash scripts/ablate_t6_steps.sh 50 100 200 400 700
#   bash scripts/ablate_t6_steps.sh --lora 100 300 600   # ablation under LoRA
#   bash scripts/ablate_t6_steps.sh --dry_run 100

set -euo pipefail

USE_LORA=0
DRY_RUN=0
STEPS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lora)     USE_LORA=1; shift ;;
        --dry_run)  DRY_RUN=1;  shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) STEPS+=("$1"); shift ;;
    esac
done
[[ "${#STEPS[@]}" -eq 0 ]] && STEPS=(100 200 400 800)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

ABL_DIR="$ROOT/runs/validation/ablate_t6_steps"
mkdir -p "$ABL_DIR"
LOG_DIR="$ROOT/runs/ablate_logs"
mkdir -p "$LOG_DIR"

TS_ALL=$(date +%Y%m%d_%H%M%S)
MANIFEST="$ABL_DIR/manifest_${TS_ALL}.txt"
{
    echo "ablation started: $TS_ALL"
    echo "STEPS = ${STEPS[*]}"
    echo "USE_LORA = $USE_LORA"
    echo "DRY_RUN = $DRY_RUN"
} > "$MANIFEST"

echo "[ABL] ============================================================"
echo "[ABL]   T6_MAX_STEPS ablation:  ${STEPS[*]}"
echo "[ABL]   USE_LORA=$USE_LORA  DRY_RUN=$DRY_RUN"
echo "[ABL]   output dir: $ABL_DIR"
echo "[ABL] ============================================================"

T6_CKPT_DIR="$ROOT/runs/training/v161_t6"

# Assemble run_all extra flags once
EXTRA_FLAGS=()
[[ "$USE_LORA" -eq 1 ]] && EXTRA_FLAGS+=(--t6_use_lora --t6_parallel ddp)

for S in "${STEPS[@]}"; do
    echo
    echo "[ABL] >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  steps=$S  <<<<<<<<<<<<<<<<<<<<<<<<<<"
    LOG="$LOG_DIR/ablate_steps_${S}_${TS_ALL}.log"

    # 1. wipe prior T6 ckpt so Phase 4 re-runs (skip-if-present guard bypass)
    if [[ -d "$T6_CKPT_DIR" && "$DRY_RUN" -eq 0 ]]; then
        echo "[ABL] wiping $T6_CKPT_DIR"
        rm -rf "$T6_CKPT_DIR"
    fi

    # 2. run Phase 4 + Phase 5
    dry_flag=()
    [[ "$DRY_RUN" -eq 1 ]] && dry_flag=(--dry_run)
    echo "[ABL] launching Phase 4+5 for steps=$S  (log: $LOG)"
    if ! bash scripts/run_all_v1.6.1.sh \
            --from_phase 4 --to_phase 5 \
            --t6_max_steps "$S" \
            "${EXTRA_FLAGS[@]}" \
            "${dry_flag[@]}" \
            > "$LOG" 2>&1; then
        echo "[ABL] ✗ steps=$S FAILED — see $LOG"
        echo "steps=$S FAILED" >> "$MANIFEST"
        continue
    fi

    # 3. locate + rename the eval dir we just produced
    [[ "$DRY_RUN" -eq 1 ]] && { echo "[ABL] dry: skip rename/collect"; continue; }

    latest_eval=$(ls -dt "$ROOT"/runs/validation/v161_eval_* 2>/dev/null | head -1)
    if [[ -z "$latest_eval" || ! -d "$latest_eval" ]]; then
        echo "[ABL] ✗ steps=$S: no v161_eval_* dir produced"
        echo "steps=$S NO_EVAL_DIR" >> "$MANIFEST"
        continue
    fi
    DEST="$ABL_DIR/steps_${S}"
    rm -rf "$DEST"
    mv "$latest_eval" "$DEST"
    echo "[ABL]   steps=$S → $DEST"
    echo "steps=$S ok → $DEST" >> "$MANIFEST"
done

# 4. aggregate summary into one markdown table
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[ABL] dry-run done"; exit 0
fi

echo
echo "[ABL] aggregating summary..."
python - <<PY
import json
from pathlib import Path

abl = Path("$ABL_DIR")
rows = []
for d in sorted(abl.glob("steps_*"),
                key=lambda p: int(p.name.split("_")[1])):
    sj = d / "summary.json"
    if not sj.exists():
        rows.append((d.name, "MISSING", None))
        continue
    data = json.load(open(sj))
    ckpts = data.get("ckpts", [])
    base = next((c for c in ckpts if c["label"] == "baseline"), None)
    t6   = next((c for c in ckpts if c["label"] == "t6"), None)
    rows.append((d.name, base, t6))

lines = [
    "# T6 SFT max_steps ablation",
    "",
    "Finding the Pareto sweet spot for fail rescue vs ok retention.",
    "Physical: 1 step ≈ 8 samples ≈ 1/169 epoch (train split = 1350).",
    "",
    "| steps | epochs | fail rescued | ok retention | Δ fail | Δ ok | net | FAIL18 | ceiling-5 |",
    "|---|---|---|---|---|---|---|---|---|",
]
for name, base, t6 in rows:
    if base is None or t6 is None:
        lines.append(f"| {name} | — | — | — | — | — | — | — | — |")
        continue
    s = int(name.split("_")[1])
    ep = s * 8 / 1350
    fail_r = t6["fail_correct"]
    ok_r   = t6["ok_correct"]
    fail_b = base["fail_correct"]
    ok_b   = base["ok_correct"]
    d_fail = fail_r - fail_b          # gained on fail
    d_ok   = ok_r   - ok_b            # (negative = lost on ok)
    net    = d_fail + d_ok
    lines.append(
        f"| {s} "
        f"| {ep:.2f} "
        f"| {fail_r}/{t6['n_fail']} ({t6['fail_pass@1']:.1%}) "
        f"| {ok_r}/{t6['n_ok']} ({t6['ok_pass@1']:.1%}) "
        f"| +{d_fail} "
        f"| {d_ok:+d} "
        f"| {net:+d} "
        f"| {t6['fail18_rescued_count']}/18 "
        f"| {t6['ceiling_broken_count']}/5 |"
    )
lines += [
    "",
    "**Target**: max net positive, ideally fail rescue ≥ 15% with ok retention ≥ 95%.",
    "",
]
out_md = abl / f"summary_${TS_ALL}.md"
out_md.write_text("\n".join(lines), encoding="utf-8")
# also overwrite latest
(abl / "summary.md").write_text("\n".join(lines), encoding="utf-8")
print(open(out_md).read())
print(f"\n[ABL] summary → {out_md}")
PY

echo "[ABL] done. All artefacts under $ABL_DIR"
