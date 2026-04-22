#!/usr/bin/env bash
# sweep_t6_steps.sh — sweep T6_MAX_STEPS to find fail/ok Pareto sweet spot.
#
# Context: v1.6.1 T6 SFT at max_steps=2000 → 190 epochs on a 1350-sample
# split → catastrophic forgetting (fail +26.6% but ok -27%). Need to find
# steps where fail rescue stays while ok retention is ≥ 95%.
#
# For each step count S in STEPS_LIST:
#   1. wipe runs/training/v161_t6
#   2. run Phase 4 (SFT with --t6_max_steps S) + Phase 5 (eval)
#   3. rename eval dir → runs/validation/sweep_t6_steps/steps_S
#   4. collect summary.json from each run
# Final: write runs/validation/sweep_t6_steps/summary.md with one row per S.
#
# Usage:
#   bash scripts/sweep_t6_steps.sh                       # default 40,80,120,200
#   bash scripts/sweep_t6_steps.sh 30 60 100 150 250
#   bash scripts/sweep_t6_steps.sh --lora 40 80 160      # sweep under LoRA
#   bash scripts/sweep_t6_steps.sh --dry_run 40

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
[[ "${#STEPS[@]}" -eq 0 ]] && STEPS=(40 80 120 200)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

SWEEP_DIR="$ROOT/runs/validation/sweep_t6_steps"
mkdir -p "$SWEEP_DIR"
LOG_DIR="$ROOT/runs/sweep_logs"
mkdir -p "$LOG_DIR"

TS_ALL=$(date +%Y%m%d_%H%M%S)
MANIFEST="$SWEEP_DIR/manifest_${TS_ALL}.txt"
{
    echo "sweep started: $TS_ALL"
    echo "STEPS = ${STEPS[*]}"
    echo "USE_LORA = $USE_LORA"
    echo "DRY_RUN = $DRY_RUN"
} > "$MANIFEST"

echo "[SWEEP] ============================================================"
echo "[SWEEP]   T6_MAX_STEPS sweep:  ${STEPS[*]}"
echo "[SWEEP]   USE_LORA=$USE_LORA  DRY_RUN=$DRY_RUN"
echo "[SWEEP]   output dir: $SWEEP_DIR"
echo "[SWEEP] ============================================================"

T6_CKPT_DIR="$ROOT/runs/training/v161_t6"

# Assemble run_all extra flags once
EXTRA_FLAGS=()
[[ "$USE_LORA" -eq 1 ]] && EXTRA_FLAGS+=(--t6_use_lora --t6_parallel ddp)

for S in "${STEPS[@]}"; do
    echo
    echo "[SWEEP] >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  steps=$S  <<<<<<<<<<<<<<<<<<<<<<<<<<"
    LOG="$LOG_DIR/sweep_steps_${S}_${TS_ALL}.log"

    # 1. wipe prior T6 ckpt so Phase 4 re-runs (skip-if-present guard bypass)
    if [[ -d "$T6_CKPT_DIR" && "$DRY_RUN" -eq 0 ]]; then
        echo "[SWEEP] wiping $T6_CKPT_DIR"
        rm -rf "$T6_CKPT_DIR"
    fi

    # 2. run Phase 4 + Phase 5
    dry_flag=()
    [[ "$DRY_RUN" -eq 1 ]] && dry_flag=(--dry_run)
    echo "[SWEEP] launching Phase 4+5 for steps=$S  (log: $LOG)"
    if ! bash scripts/run_all_v1.6.1.sh \
            --from_phase 4 --to_phase 5 \
            --t6_max_steps "$S" \
            "${EXTRA_FLAGS[@]}" \
            "${dry_flag[@]}" \
            > "$LOG" 2>&1; then
        echo "[SWEEP] ✗ steps=$S FAILED — see $LOG"
        echo "steps=$S FAILED" >> "$MANIFEST"
        continue
    fi

    # 3. locate + rename the eval dir we just produced
    [[ "$DRY_RUN" -eq 1 ]] && { echo "[SWEEP] dry: skip rename/collect"; continue; }

    latest_eval=$(ls -dt "$ROOT"/runs/validation/v161_eval_* 2>/dev/null | head -1)
    if [[ -z "$latest_eval" || ! -d "$latest_eval" ]]; then
        echo "[SWEEP] ✗ steps=$S: no v161_eval_* dir produced"
        echo "steps=$S NO_EVAL_DIR" >> "$MANIFEST"
        continue
    fi
    DEST="$SWEEP_DIR/steps_${S}"
    rm -rf "$DEST"
    mv "$latest_eval" "$DEST"
    echo "[SWEEP]   steps=$S → $DEST"
    echo "steps=$S ok → $DEST" >> "$MANIFEST"
done

# 4. aggregate summary into one markdown table
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[SWEEP] dry-run done"; exit 0
fi

echo
echo "[SWEEP] aggregating summary..."
python - <<PY
import json
from pathlib import Path

sweep = Path("$SWEEP_DIR")
rows = []
for d in sorted(sweep.glob("steps_*"),
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
    "# T6 SFT max_steps sweep",
    "",
    "Finding the Pareto sweet spot for fail rescue vs ok retention.",
    "",
    "| steps | fail rescued | ok retention | Δ fail | Δ ok | net | FAIL18 | ceiling-5 |",
    "|---|---|---|---|---|---|---|---|",
]
for name, base, t6 in rows:
    if base is None or t6 is None:
        lines.append(f"| {name} | — | — | — | — | — | — | — |")
        continue
    s = name.split("_")[1]
    fail_r = t6["fail_correct"]
    ok_r   = t6["ok_correct"]
    fail_b = base["fail_correct"]
    ok_b   = base["ok_correct"]
    d_fail = fail_r - fail_b          # gained on fail
    d_ok   = ok_r   - ok_b            # (negative = lost on ok)
    net    = d_fail + d_ok
    lines.append(
        f"| {s} "
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
out_md = sweep / f"summary_${TS_ALL}.md"
out_md.write_text("\n".join(lines), encoding="utf-8")
# also overwrite sweep/summary.md for latest
(sweep / "summary.md").write_text("\n".join(lines), encoding="utf-8")
print(open(out_md).read())
print(f"\n[SWEEP] summary → {out_md}")
PY

echo "[SWEEP] done. All artefacts under $SWEEP_DIR"
