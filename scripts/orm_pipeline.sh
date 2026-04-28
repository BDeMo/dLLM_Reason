#!/usr/bin/env bash
# orm_pipeline.sh — end-to-end ORM training + BoN eval.
#
# Stages:
#   A. Collect data: T6 ckpt × gsm8k train, N=8 samples per prompt at T=0.7,
#      save (prompt, output, label∈{0,1}) — 8-shard parallel on 8 GPU
#   B. Concat shards into single orm_train.jsonl
#   C. Train ORM head (single-GPU, ~30 min)
#   D. BoN eval on scope_fail + scope_ok (8-GPU sharded sampling + scoring)
#   E. Print comparison table (greedy vs SC vs BoN vs pass@N)
#
# References:
#   Cobbe 2021 (arXiv:2110.14168), V-STaR (arXiv:2402.06457)
#
# Usage:
#   bash scripts/orm_pipeline.sh                                  # all defaults
#   bash scripts/orm_pipeline.sh --base_ckpt <path>               # different base
#   bash scripts/orm_pipeline.sh --skip_collect --jsonl <path>    # reuse data

set -euo pipefail

BASE_CKPT="runs/training/v161_t6_ablate/hf_step_336"
SCOPE_PATH="runs/validation/gsm8k_train_prompts.json"
N_SAMPLES=8
TEMPERATURE=0.7
GEN_LENGTH=192
GEN_GPUS=8
PROMPT_BATCH=4

ORM_MAX_STEPS=2000
ORM_LR=1e-4
ORM_BATCH_SIZE=8
ORM_POOLING=mean    # "last" or "mean" — mean over output region (recommended for diffusion bidirectional model)

EVAL_N_FAIL=0   # 0 = all fail prompts
EVAL_N_OK=0     # 0 = all ok prompts (full retention measurement)

SKIP_COLLECT=0
JSONL_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base_ckpt)    BASE_CKPT="$2"; shift 2 ;;
        --scope_path)   SCOPE_PATH="$2"; shift 2 ;;
        --n_samples)    N_SAMPLES="$2"; shift 2 ;;
        --temperature)  TEMPERATURE="$2"; shift 2 ;;
        --gen_length)   GEN_LENGTH="$2"; shift 2 ;;
        --gen_gpus)     GEN_GPUS="$2"; shift 2 ;;
        --prompt_batch) PROMPT_BATCH="$2"; shift 2 ;;
        --max_steps)    ORM_MAX_STEPS="$2"; shift 2 ;;
        --lr)           ORM_LR="$2"; shift 2 ;;
        --batch_size)   ORM_BATCH_SIZE="$2"; shift 2 ;;
        --pooling)      ORM_POOLING="$2"; shift 2 ;;
        --n_fail)       EVAL_N_FAIL="$2"; shift 2 ;;
        --n_ok)         EVAL_N_OK="$2"; shift 2 ;;
        --skip_collect) SKIP_COLLECT=1; shift ;;
        --jsonl)        JSONL_OVERRIDE="$2"; SKIP_COLLECT=1; shift 2 ;;
        -h|--help)      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[ORM] unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

TS=$(date +%Y%m%d_%H%M%S)
DATA_DIR="$ROOT/runs/validation/orm_data_${TS}"
TRAIN_DIR="$ROOT/runs/training/orm_${TS}"
EVAL_DIR="$ROOT/runs/validation/orm_eval_${TS}"
LOG_DIR="$ROOT/runs/orm_logs"
mkdir -p "$DATA_DIR" "$LOG_DIR"

JSONL="$DATA_DIR/orm_train.jsonl"

echo "[ORM] ============================================================"
echo "[ORM]   base_ckpt    = $BASE_CKPT"
echo "[ORM]   scope        = $SCOPE_PATH"
echo "[ORM]   N samples    = $N_SAMPLES @ T=$TEMPERATURE"
echo "[ORM]   data_dir     = $DATA_DIR"
echo "[ORM]   train_dir    = $TRAIN_DIR"
echo "[ORM]   eval_dir     = $EVAL_DIR"
echo "[ORM] ============================================================"

# ── Phase A: collect data, 8 GPU sharded ────────────────────────────────
if [[ "$SKIP_COLLECT" -eq 0 ]]; then
    echo "[ORM][A] data collection — $GEN_GPUS shards"
    declare -a PIDS=()
    for ((s=0; s<GEN_GPUS; s++)); do
        LOG="$LOG_DIR/orm_collect_shard${s}_${TS}.log"
        CUDA_VISIBLE_DEVICES=$s PYTHONUNBUFFERED=1 python -u \
            scripts/orm_collect_data.py \
            --model "$BASE_CKPT" \
            --scope_path "$SCOPE_PATH" \
            --n_samples "$N_SAMPLES" \
            --temperature "$TEMPERATURE" \
            --gen_length "$GEN_LENGTH" \
            --prompt_batch "$PROMPT_BATCH" \
            --prompt_shard "$s/$GEN_GPUS" \
            --out_jsonl "$JSONL" \
            > "$LOG" 2>&1 &
        PIDS+=($!)
    done
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}" || echo "[ORM][A] shard $i FAILED"
    done

    echo "[ORM][B] concat shards → $JSONL"
    cat "$DATA_DIR"/orm_train.shard*.jsonl > "$JSONL"
    n_lines=$(wc -l < "$JSONL")
    echo "[ORM][B] $n_lines total samples"
else
    JSONL="${JSONL_OVERRIDE:-$JSONL}"
    [[ ! -f "$JSONL" ]] && { echo "[ORM] ERROR: no jsonl at $JSONL"; exit 1; }
    echo "[ORM] reuse jsonl: $JSONL"
fi

# ── Phase C: train ORM head (DDP, 8-GPU) ────────────────────────────────
echo "[ORM][C] training head (DDP × $GEN_GPUS GPU) ..."
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node="$GEN_GPUS" \
    scripts/orm_train.py \
    --base_ckpt "$BASE_CKPT" \
    --train_jsonl "$JSONL" \
    --out_dir "$TRAIN_DIR" \
    --max_steps "$ORM_MAX_STEPS" \
    --batch_size "$ORM_BATCH_SIZE" \
    --lr "$ORM_LR" \
    --pooling "$ORM_POOLING" \
    2>&1 | tee "$LOG_DIR/orm_train_${TS}.log"

HEAD_CKPT="$TRAIN_DIR/head_final.pt"
[[ ! -f "$HEAD_CKPT" ]] && { echo "[ORM] ERROR: head training failed"; exit 1; }

# ── Phase D: BoN eval, $GEN_GPUS shards in parallel ────────────────────
echo "[ORM][D] BoN eval — $GEN_GPUS shards"
mkdir -p "$EVAL_DIR"
declare -a EVAL_PIDS=()
for ((s=0; s<GEN_GPUS; s++)); do
    LOG="$LOG_DIR/orm_eval_shard${s}_${TS}.log"
    CUDA_VISIBLE_DEVICES=$s PYTHONUNBUFFERED=1 python -u \
        scripts/orm_eval_bon.py \
        --base_ckpt "$BASE_CKPT" \
        --orm_head "$HEAD_CKPT" \
        --n_samples "$N_SAMPLES" \
        --temperature "$TEMPERATURE" \
        --n_fail "$EVAL_N_FAIL" --n_ok "$EVAL_N_OK" \
        --prompt_shard "$s/$GEN_GPUS" \
        --pooling "$ORM_POOLING" \
        --out_dir "$EVAL_DIR" \
        > "$LOG" 2>&1 &
    EVAL_PIDS+=($!)
done
for i in "${!EVAL_PIDS[@]}"; do
    wait "${EVAL_PIDS[$i]}" || echo "[ORM][D] eval shard $i FAILED"
done

echo "[ORM][D] aggregating shard summaries ..."
python -u scripts/orm_eval_aggregate.py --eval_dir "$EVAL_DIR" \
    2>&1 | tee "$LOG_DIR/orm_eval_${TS}.log"

# ── Phase E: print summary ─────────────────────────────────────────────
echo
echo "════════════════════ ORM Result ════════════════════"
cat "$EVAL_DIR/summary.md"
echo
echo "[ORM] DONE. data=$DATA_DIR  train=$TRAIN_DIR  eval=$EVAL_DIR"
