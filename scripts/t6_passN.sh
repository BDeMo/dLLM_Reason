#!/usr/bin/env bash
# t6_passN.sh — pass@N eval on a T6-trained ckpt under T>0 sampling.
#
# Complements the canonical T=0 pass@1 ablation (scripts/t6_ablate.sh):
# measures the diversity/capacity frontier of the trained model under
# stochastic decoding. Uses scripts/validate/h3_passN_at_temperature.py
# which subsamples scope_fail / scope_ok and reports pass@k for k≤N at
# each temperature.
#
# Usage:
#   # Auto-pick the best ckpt from t6_ablate: pass --auto
#   bash scripts/t6_passN.sh --auto
#
#   # Specific ckpt (any HF-format dir with config.json):
#   bash scripts/t6_passN.sh --ckpt runs/training/v161_t6_ablate/hf_step_169
#
#   # Override N / temps / scope subset size:
#   bash scripts/t6_passN.sh --ckpt <path> --n_samples 16 --temps 0.3 0.7 1.0
#   bash scripts/t6_passN.sh --ckpt <path> --n_fail 60 --n_ok 60
#
#   # Sweep ALL checkpoints in a t6_ablate training run:
#   bash scripts/t6_passN.sh --sweep runs/training/v161_t6_ablate
#
#   # Plan only:
#   bash scripts/t6_passN.sh --auto --dry_run

set -euo pipefail

CKPT=""
AUTO=0
SWEEP_DIR=""
N_SAMPLES=8
GEN_LENGTH=128
BLOCK_LENGTH=32
STEPS_=128                      # MDLM steps (= gen_length by canonical)
TEMPS=(0.3 0.7 1.0)
N_FAIL=30
N_OK=30
EVAL_GPUS=8                     # parallel ckpts across this many GPUs
GPU_CSV=""
AUTO_GPUS=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)         CKPT="$2"; shift 2 ;;
        --auto)         AUTO=1; shift ;;
        --sweep)        SWEEP_DIR="$2"; shift 2 ;;
        --n_samples)    N_SAMPLES="$2"; shift 2 ;;
        --gen_length)   GEN_LENGTH="$2"; shift 2 ;;
        --block_length) BLOCK_LENGTH="$2"; shift 2 ;;
        --steps)        STEPS_="$2"; shift 2 ;;
        --temps)        shift; TEMPS=(); while [[ $# -gt 0 && "$1" != --* ]]; do TEMPS+=("$1"); shift; done ;;
        --n_fail)       N_FAIL="$2"; shift 2 ;;
        --n_ok)         N_OK="$2"; shift 2 ;;
        --eval_gpus)    EVAL_GPUS="$2"; shift 2 ;;
        --gpus)         GPU_CSV="$2"; shift 2 ;;
        --auto_gpus)    AUTO_GPUS=1; shift ;;
        --dry_run)      DRY_RUN=1; shift ;;
        -h|--help)      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[PASSN] unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Resolve ckpts to eval
CKPT_LIST=()
if [[ -n "$SWEEP_DIR" ]]; then
    for d in "$SWEEP_DIR"/hf_step_*; do
        [[ -f "$d/config.json" ]] && CKPT_LIST+=("$d")
    done
elif [[ "$AUTO" -eq 1 ]]; then
    # auto = latest t6_ablate training, pick whichever step had the best
    # net score from the most recent summary.md. If no summary, default to
    # the final hf/ ckpt.
    LATEST_TRAIN=$(ls -dt "$ROOT"/runs/training/v161_t6_ablate* 2>/dev/null | head -1)
    [[ -z "$LATEST_TRAIN" ]] && { echo "[PASSN] --auto: no v161_t6_ablate dir"; exit 1; }
    SUMMARY="$ROOT/runs/validation/t6_ablate/summary.md"
    if [[ -f "$SUMMARY" ]]; then
        BEST_STEP=$(python - <<PY
import re
from pathlib import Path
best_step = None; best_net = -10**9
for line in open("$SUMMARY"):
    m = re.match(r"\| (\d+) \|.*\| ([+-]?\d+) \| ([+-]?\d+) \| ([+-]?\d+) \|", line)
    if m:
        step, df, do, net = int(m[1]), int(m[2]), int(m[3]), int(m[4])
        if net > best_net:
            best_net, best_step = net, step
print(best_step or "")
PY
)
        if [[ -n "$BEST_STEP" ]]; then
            CKPT="$LATEST_TRAIN/hf_step_${BEST_STEP}"
            echo "[PASSN] auto-picked best-net ckpt: step=$BEST_STEP → $CKPT"
        fi
    fi
    [[ -z "$CKPT" ]] && CKPT="$LATEST_TRAIN/hf"
    CKPT_LIST+=("$CKPT")
elif [[ -n "$CKPT" ]]; then
    CKPT_LIST+=("$CKPT")
else
    echo "[PASSN] ERROR: pass --ckpt <path> | --auto | --sweep <train_dir>" >&2
    exit 2
fi

OUT_BASE="$ROOT/runs/validation/t6_passN"
mkdir -p "$OUT_BASE"
TS=$(date +%Y%m%d_%H%M%S)

echo "[PASSN] ============================================================"
echo "[PASSN]   ckpts to eval : ${#CKPT_LIST[@]}"
for c in "${CKPT_LIST[@]}"; do echo "[PASSN]     $c"; done
echo "[PASSN]   N_SAMPLES     = $N_SAMPLES"
echo "[PASSN]   TEMPS         = ${TEMPS[*]}"
echo "[PASSN]   n_fail / n_ok = $N_FAIL / $N_OK"
echo "[PASSN]   gen/block/steps= $GEN_LENGTH / $BLOCK_LENGTH / $STEPS_"
echo "[PASSN] ============================================================"

echo "[PASSN]   EVAL_GPUS      = $EVAL_GPUS (ckpts run in parallel)"

# Resolve GPU indices
if [[ -n "$GPU_CSV" ]]; then
    IFS=',' read -r -a GPUS_ARR <<< "$GPU_CSV"
    EVAL_GPUS="${#GPUS_ARR[@]}"
elif [[ "$AUTO_GPUS" -eq 1 ]]; then
    source "$SCRIPT_DIR/_select_gpus.sh"
    SEL=$(select_free_gpus "$EVAL_GPUS")
    IFS=',' read -r -a GPUS_ARR <<< "$SEL"
    echo "[PASSN]   auto-selected  = $SEL"
else
    GPUS_ARR=(); for i in $(seq 0 $((EVAL_GPUS - 1))); do GPUS_ARR+=("$i"); done
fi
echo

declare -a PIDS=()
declare -a PID_LABELS=()
g=0
for CK in "${CKPT_LIST[@]}"; do
    [[ ! -f "$CK/config.json" ]] && { echo "[PASSN] skip (no config.json): $CK"; continue; }
    LABEL=$(basename "$CK")
    PARENT=$(basename "$(dirname "$CK")")
    RUN_DIR="$OUT_BASE/${PARENT}_${LABEL}_${TS}"
    LOG="$OUT_BASE/${PARENT}_${LABEL}_${TS}.log"

    GPU="${GPUS_ARR[$g]}"
    echo "[PASSN] launching on GPU $GPU (slot $g): ckpt=$CK → $RUN_DIR"
    [[ "$DRY_RUN" -eq 1 ]] && { echo "[PASSN]   (dry-run, skip)"; continue; }

    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python -u \
        scripts/validate/h3_passN_at_temperature.py \
        --model "$CK" \
        --run_dir "$RUN_DIR" \
        --n_samples "$N_SAMPLES" \
        --gen_length "$GEN_LENGTH" \
        --block_length "$BLOCK_LENGTH" \
        --steps "$STEPS_" \
        --temps "${TEMPS[@]}" \
        --n_fail "$N_FAIL" \
        --n_ok "$N_OK" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
    PID_LABELS+=("${PARENT}/${LABEL}")
    g=$(( (g + 1) % EVAL_GPUS ))
    if [[ "${#PIDS[@]}" -ge "$EVAL_GPUS" ]]; then
        wait "${PIDS[0]}" || echo "[PASSN] ✗ ${PID_LABELS[0]} FAILED"
        PIDS=("${PIDS[@]:1}")
        PID_LABELS=("${PID_LABELS[@]:1}")
    fi
done
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || echo "[PASSN] ✗ ${PID_LABELS[$i]} FAILED"
done

# ── aggregate via standalone script (single source of truth) ─────────────
# Previously had an inline aggregator here that conflicted with
# scripts/t6_passN_aggregate.py — both wrote to t6_passN/summary.md
# with different semantics (TS-filtered batch vs full history).
# Now delegate to the standalone, which full-scans every subdir.
if [[ "$DRY_RUN" -eq 1 ]]; then exit 0; fi

echo
echo "[PASSN] aggregating via t6_passN_aggregate.py ..."
python scripts/t6_passN_aggregate.py --dir "$OUT_BASE"

echo "[PASSN] done. → $OUT_BASE"
