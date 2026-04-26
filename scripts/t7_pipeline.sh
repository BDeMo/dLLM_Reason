#!/usr/bin/env bash
# t7_pipeline.sh — T7 self-distill pipeline.
#
# Goal: bake pass@N capacity into greedy by SFT-ing on the model's own
# correct samples. Drives T=0 pass@1 from ~28% (T6 best) toward the ~65%
# pass@N ceiling we measured in t6_decode_ablate.
#
# Stages:
#   A. Generate correct samples on gsm8k train via T>0 sampling
#      (parallel across 8 GPU prompt-shards)
#   B. Aggregate per-shard outputs → single t7_sft.jsonl
#   C. SFT (FSDP, full-SFT) on T6 ckpt as warm-start
#   D. Eval T7 ckpt — canonical T=0 + decode_ablate
#
# Usage:
#   bash scripts/t7_pipeline.sh                          # default: T6 best ckpt
#   bash scripts/t7_pipeline.sh --base_ckpt <path>       # custom warm-start
#   bash scripts/t7_pipeline.sh --skip_gen --jsonl <path># reuse old gen run
#   bash scripts/t7_pipeline.sh --max_train 500          # subsample for smoke

set -euo pipefail

BASE_CKPT="runs/training/v161_t6_ablate/hf_step_336"   # default warm-start
SCOPE_PATH=""                                            # default: gsm8k train
N_PROMPTS=0                                              # 0 = all in scope
TEMPERATURES="0.7,1.0"
N_SAMPLES=8
GEN_LENGTH=192
BLOCK_LENGTH=32
PICK="shortest"
SFT_GPUS=8
EVAL_GPUS=8
T7_MAX_STEPS=1500
T7_LR=2e-5
GEN_GPUS=8
SKIP_GEN=0
JSONL_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base_ckpt)    BASE_CKPT="$2"; shift 2 ;;
        --scope_path)   SCOPE_PATH="$2"; shift 2 ;;
        --n_prompts)    N_PROMPTS="$2"; shift 2 ;;
        --temperatures) TEMPERATURES="$2"; shift 2 ;;
        --n_samples)    N_SAMPLES="$2"; shift 2 ;;
        --gen_length)   GEN_LENGTH="$2"; shift 2 ;;
        --pick)         PICK="$2"; shift 2 ;;
        --gen_gpus)     GEN_GPUS="$2"; shift 2 ;;
        --sft_gpus)     SFT_GPUS="$2"; shift 2 ;;
        --eval_gpus)    EVAL_GPUS="$2"; shift 2 ;;
        --max_train)    T7_MAX_STEPS="$2"; shift 2 ;;
        --lr)           T7_LR="$2"; shift 2 ;;
        --skip_gen)     SKIP_GEN=1; shift ;;
        --jsonl)        JSONL_OVERRIDE="$2"; SKIP_GEN=1; shift 2 ;;
        -h|--help)      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[T7] unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Default scope: gsm8k train prompts (full set)
if [[ -z "$SCOPE_PATH" ]]; then
    SCOPE_PATH="$ROOT/runs/validation/gsm8k_train_prompts.json"
fi
[[ ! -f "$SCOPE_PATH" ]] && {
    echo "[T7] ERROR: scope file missing: $SCOPE_PATH" >&2
    echo "    Run Phase 0 first or pass --scope_path PATH" >&2; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
GEN_RUN="$ROOT/runs/validation/t7_gen_${TS}"
LOG_DIR="$ROOT/runs/t7_logs"
mkdir -p "$LOG_DIR"

echo "[T7] ============================================================"
echo "[T7]   base_ckpt    = $BASE_CKPT"
echo "[T7]   scope        = $SCOPE_PATH"
echo "[T7]   n_prompts    = $N_PROMPTS  (0 = all)"
echo "[T7]   temperatures = $TEMPERATURES"
echo "[T7]   n_samples    = $N_SAMPLES"
echo "[T7]   gen_length   = $GEN_LENGTH"
echo "[T7]   pick policy  = $PICK"
echo "[T7]   gen_gpus     = $GEN_GPUS"
echo "[T7]   sft_gpus     = $SFT_GPUS  max_steps = $T7_MAX_STEPS"
echo "[T7]   eval_gpus    = $EVAL_GPUS"
echo "[T7]   gen_run      = $GEN_RUN"
echo "[T7] ============================================================"

# ── Phase A: parallel sample generation across $GEN_GPUS prompt-shards ──
if [[ "$SKIP_GEN" -eq 0 ]]; then
    echo "[T7][A] launching $GEN_GPUS prompt-shards on $BASE_CKPT ..."
    declare -a PIDS=()
    for ((s=0; s<GEN_GPUS; s++)); do
        LOG="$LOG_DIR/t7_gen_shard${s}_${TS}.log"
        CUDA_VISIBLE_DEVICES=$s PYTHONUNBUFFERED=1 python -u \
            scripts/validate/t7_gen_correct_samples.py \
            --model "$BASE_CKPT" \
            --scope_path "$SCOPE_PATH" \
            --scope_group gsm8k_train \
            --n "$N_PROMPTS" \
            --temperatures "$TEMPERATURES" \
            --n_samples "$N_SAMPLES" \
            --gen_length "$GEN_LENGTH" \
            --block_length "$BLOCK_LENGTH" \
            --pick "$PICK" \
            --run_dir "$GEN_RUN" \
            --resume \
            --prompt_shard "$s/$GEN_GPUS" \
            --prompt_batch auto \
            > "$LOG" 2>&1 &
        PIDS+=($!)
        echo "[T7][A]   shard $s/$GEN_GPUS on GPU $s → $LOG"
    done
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}" || echo "[T7][A] ✗ shard $i FAILED"
    done

    # Aggregate per-prompt JSON files into single t7_sft.jsonl
    echo "[T7][B] aggregating per_prompt → t7_sft.jsonl ..."
    python - <<PY
import json
from pathlib import Path
gen = Path("$GEN_RUN")
out = gen / "t7_sft.jsonl"
n_with = n_without = 0
with out.open("w", encoding="utf-8") as f:
    for p in sorted((gen / "per_prompt").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("answer"):
            f.write(json.dumps({
                "group": r["group"], "idx": r["idx"], "gt": r["gt"],
                "question": r["question"], "answer": r["answer"],
                "selection": r["selection"], "temperature": r["temperature"],
                "n_candidates": r["n_candidates"],
            }, ensure_ascii=False) + "\n")
            n_with += 1
        else:
            n_without += 1
total = n_with + n_without
cover = n_with / max(total, 1)
print(f"[T7][B] {n_with}/{total} prompts had ≥1 correct ({cover:.1%})")
print(f"[T7][B] SFT pairs → {out}")
PY
    JSONL="$GEN_RUN/t7_sft.jsonl"
else
    JSONL="${JSONL_OVERRIDE:-$ROOT/runs/validation/t7_gen_*/t7_sft.jsonl}"
    JSONL=$(ls -t $JSONL 2>/dev/null | head -1)
    [[ -z "$JSONL" ]] && { echo "[T7] ERROR: --skip_gen but no jsonl found"; exit 1; }
    echo "[T7] reuse existing JSONL: $JSONL"
fi

[[ ! -s "$JSONL" ]] && { echo "[T7] ERROR: empty JSONL: $JSONL"; exit 1; }

# ── Phase C: SFT ────────────────────────────────────────────────────────
T7_RUN_NAME="v161_t7_${TS}"
T7_DIR="$ROOT/runs/training/$T7_RUN_NAME"
LAUNCH="torchrun --standalone --nproc_per_node=$SFT_GPUS"
[[ "$SFT_GPUS" -le 1 ]] && LAUNCH="python"

echo "[T7][C] SFT on $JSONL → $T7_DIR (warm-start from $BASE_CKPT)"
$LAUNCH scripts/validate/t6t7_train.py \
    --jsonl_path "$JSONL" \
    --run_name "$T7_RUN_NAME" \
    --init_ckpt "$BASE_CKPT" \
    --max_steps "$T7_MAX_STEPS" \
    --batch_size 1 --grad_accum_steps 16 \
    --lr "$T7_LR" \
    --max_seq_len 768 \
    --parallel fsdp \
    2>&1 | tee "$LOG_DIR/t7_sft_${TS}.log"

[[ ! -f "$T7_DIR/hf/config.json" ]] && {
    echo "[T7][C] ERROR: SFT failed — no $T7_DIR/hf"; exit 1; }

# ── Phase D: eval ───────────────────────────────────────────────────────
echo "[T7][D] canonical eval ..."
EVAL_OUT="$ROOT/runs/validation/t7_eval_${TS}"
python scripts/validate/v16_eval.py \
    --ckpts "baseline=GSAI-ML/LLaDA-8B-Instruct" \
            "t6=$BASE_CKPT" \
            "t7=$T7_DIR/hf" \
    --out_dir "$EVAL_OUT" \
    --gen_length 128 --block_length 32 --temperature 0 \
    2>&1 | tee "$LOG_DIR/t7_eval_canonical_${TS}.log"

cat "$EVAL_OUT/comparison.md"

echo "[T7][D] decode_ablate on T7 ..."
bash scripts/t6_decode_ablate.sh \
    --ckpt "$T7_DIR/hf" \
    --auto_gpus \
    --prompt_batch 8 \
    2>&1 | tee "$LOG_DIR/t7_decode_ablate_${TS}.log"

echo
echo "[T7] ════════════════ DONE ════════════════"
echo "[T7]   gen_run      = $GEN_RUN"
echo "[T7]   t7_sft.jsonl = $JSONL"
echo "[T7]   T7 ckpt      = $T7_DIR/hf"
echo "[T7]   canonical    = $EVAL_OUT/comparison.md"
echo "[T7]   decode_ablate= runs/validation/t6_decode_ablate/v161_t7_${TS}_hf/"
