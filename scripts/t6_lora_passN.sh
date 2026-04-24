#!/usr/bin/env bash
# t6_lora_passN.sh — pass@N eval sweep across LoRA (rank × epoch) ckpts.
#
# Wraps t6_passN.sh --sweep per rank. Each rank training dir contains
# hf_step_<S>_merged/ for each target step (produced by t6_lora_ablate
# Phase A.5). The --sweep glob matches hf_step_* but only merged dirs
# have config.json, so adapter dirs are naturally filtered out.
#
# Output goes to runs/validation/t6_passN/<parent>_<label>_<TS>/ per
# ckpt, shared with full-SFT passN (same aggregator handles both).
#
# Usage:
#   bash scripts/t6_lora_passN.sh                # default ranks 1,2,4,8,16
#   bash scripts/t6_lora_passN.sh --ranks 1,4    # subset
#   bash scripts/t6_lora_passN.sh --n_samples 16 --temps 0.3 0.7 1.0
#
# After completion run:
#   python scripts/t6_passN_aggregate.py
# to merge into one summary.md with both full-SFT and LoRA rows.

set -euo pipefail

RANKS_CSV="1,2,4,8,16"
N_SAMPLES=8
TEMPS=(0.3 0.7 1.0)
N_FAIL=30
N_OK=30
EVAL_GPUS=8

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ranks)     RANKS_CSV="$2"; shift 2 ;;
        --n_samples) N_SAMPLES="$2"; shift 2 ;;
        --n_fail)    N_FAIL="$2"; shift 2 ;;
        --n_ok)      N_OK="$2"; shift 2 ;;
        --eval_gpus) EVAL_GPUS="$2"; shift 2 ;;
        --temps)     shift; TEMPS=(); while [[ $# -gt 0 && "$1" != --* ]]; do TEMPS+=("$1"); shift; done ;;
        -h|--help)   grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[LORA-PASSN] unknown arg: $1" >&2; exit 1 ;;
    esac
done

IFS=',' read -r -a RANKS <<< "$RANKS_CSV"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

echo "[LORA-PASSN] ranks=${RANKS[*]}  n_samples=$N_SAMPLES  temps=${TEMPS[*]}"
echo "[LORA-PASSN] n_fail=$N_FAIL  n_ok=$N_OK  eval_gpus=$EVAL_GPUS"

for R in "${RANKS[@]}"; do
    TRAIN_DIR="$ROOT/runs/training/v161_t6_lora_r${R}"
    if [[ ! -d "$TRAIN_DIR" ]]; then
        echo "[LORA-PASSN] ✗ no training dir for rank=$R at $TRAIN_DIR; skipping"
        continue
    fi
    # Verify at least one merged ckpt exists
    if ! ls "$TRAIN_DIR"/hf_step_*_merged/config.json >/dev/null 2>&1; then
        echo "[LORA-PASSN] ✗ rank=$R has no hf_step_*_merged (run t6_lora_ablate Phase A.5 first)"
        continue
    fi

    echo
    echo "[LORA-PASSN] ===== rank=$R  sweep $TRAIN_DIR ====="
    bash scripts/t6_passN.sh --sweep "$TRAIN_DIR" \
        --n_samples "$N_SAMPLES" \
        --n_fail "$N_FAIL" --n_ok "$N_OK" \
        --eval_gpus "$EVAL_GPUS" \
        --temps "${TEMPS[@]}"
done

echo
echo "[LORA-PASSN] done. Aggregate all (full-SFT + LoRA) results:"
echo "    python scripts/t6_passN_aggregate.py"
