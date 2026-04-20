#!/usr/bin/env bash
# run_t6_shards.sh — Multi-GPU T6 teacher-trace generation.
#
# Pattern mirrors run_ss_shards.sh: one local Qwen instance per GPU, each
# handles a disjoint prompt slice, all write to a shared run_dir.
#
# Usage:
#   bash scripts/run_t6_shards.sh                        # 8 GPUs, defaults
#   bash scripts/run_t6_shards.sh -g 4 --max_train 500
#   bash scripts/run_t6_shards.sh --teacher_ckpt checkpoints/Qwen__Qwen3.5-4B
#   bash scripts/run_t6_shards.sh --dry_run
#
# After all shards finish, runs a final no-slice aggregate pass that
# rebuilds the global t6_sft.jsonl from per_prompt/*.json.

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
GPUS=8
MAX_TRAIN=2000
TEACHER_CKPT=""           # auto = checkpoints/Qwen__Qwen3.5-4B
TEACHER_SIZE="3.5-4B"
T6_RETRIES=3
T6_MAX_TOKENS=800
SCOPE_PATH=""             # auto = runs/validation/gsm8k_train_prompts.json
SCOPE_GROUP="gsm8k"
RUN_DIR=""                # auto = runs/validation/t6_teacher_trace_<ts>
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--gpus)        GPUS="$2";          shift 2 ;;
        --max_train)      MAX_TRAIN="$2";     shift 2 ;;
        --teacher_ckpt)   TEACHER_CKPT="$2";  shift 2 ;;
        --teacher_size)   TEACHER_SIZE="$2";  shift 2 ;;
        --retries)        T6_RETRIES="$2";    shift 2 ;;
        --max_tokens)     T6_MAX_TOKENS="$2"; shift 2 ;;
        --scope_path)     SCOPE_PATH="$2";    shift 2 ;;
        --scope_group)    SCOPE_GROUP="$2";   shift 2 ;;
        -r|--run_dir)     RUN_DIR="$2";       shift 2 ;;
        --dry_run)        DRY_RUN=1;          shift ;;
        -h|--help)        grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)                echo "[T6-SH] unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Defaults that depend on ROOT
[[ -z "$TEACHER_CKPT" ]] && TEACHER_CKPT="$ROOT/checkpoints/Qwen__Qwen${TEACHER_SIZE}"
[[ -z "$SCOPE_PATH" ]]   && SCOPE_PATH="$ROOT/runs/validation/gsm8k_train_prompts.json"
if [[ -z "$RUN_DIR" ]]; then
    TS="$(date +%Y%m%d_%H%M%S)"
    RUN_DIR="$ROOT/runs/validation/t6_teacher_trace_${TS}"
fi
mkdir -p "$RUN_DIR"
LOG_DIR="$RUN_DIR/shard_logs"
mkdir -p "$LOG_DIR"

echo "[T6-SH] ================================================================"
echo "[T6-SH]   Multi-GPU T6 teacher trace generation"
echo "[T6-SH] ================================================================"
echo "[T6-SH] GPUS          = $GPUS"
echo "[T6-SH] TEACHER_CKPT  = $TEACHER_CKPT"
echo "[T6-SH] SCOPE_PATH    = $SCOPE_PATH"
echo "[T6-SH] MAX_TRAIN     = $MAX_TRAIN"
echo "[T6-SH] T6_RETRIES    = $T6_RETRIES"
echo "[T6-SH] RUN_DIR       = $RUN_DIR"
echo "[T6-SH]"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if [[ ! -d "$TEACHER_CKPT" ]]; then
    echo "[T6-SH] ERROR: teacher ckpt dir not found: $TEACHER_CKPT" >&2
    echo "[T6-SH]        Download with:" >&2
    echo "[T6-SH]        python scripts/download_qwen.py --sizes $TEACHER_SIZE --mirror hf-mirror" >&2
    exit 1
fi
if [[ ! -f "$SCOPE_PATH" ]]; then
    echo "[T6-SH] ERROR: scope file not found: $SCOPE_PATH" >&2
    echo "[T6-SH]        Build with:" >&2
    echo "[T6-SH]        python scripts/validate/load_gsm8k_train.py --max_samples $MAX_TRAIN" >&2
    exit 1
fi

# Compute effective prompt count (scope file may have fewer items than MAX_TRAIN)
TOTAL_PROMPTS=$(python - <<PY 2>&1
import json
data = json.load(open("$SCOPE_PATH"))
n = int("$MAX_TRAIN")
print(min(len(data), n))
PY
)
if ! [[ "$TOTAL_PROMPTS" =~ ^[0-9]+$ ]]; then
    echo "[T6-SH] ERROR: failed to count prompts. Python output was:" >&2
    echo "$TOTAL_PROMPTS" >&2
    exit 1
fi
echo "[T6-SH] total prompts = $TOTAL_PROMPTS (scope has $(python -c "import json; print(len(json.load(open('$SCOPE_PATH'))))"))"

if [[ "$TOTAL_PROMPTS" -lt "$GPUS" ]]; then
    echo "[T6-SH] WARN: fewer prompts ($TOTAL_PROMPTS) than GPUs ($GPUS); "
    echo "[T6-SH]       some shards will be empty."
fi

# ── Compute shard slices (even split, remainder to first shards) ─────────────
declare -a SHARD_START SHARD_END
base=$((TOTAL_PROMPTS / GPUS))
rem=$((TOTAL_PROMPTS % GPUS))
acc=0
for ((g=0; g<GPUS; g++)); do
    sz=$base
    [[ "$g" -lt "$rem" ]] && sz=$((sz + 1))
    SHARD_START[$g]=$acc
    acc=$((acc + sz))
    SHARD_END[$g]=$acc
done

# ── Launch shards ─────────────────────────────────────────────────────────────
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[T6-SH] DRY RUN — would launch $GPUS shards:"
    for ((g=0; g<GPUS; g++)); do
        echo "  GPU $g  prompts[${SHARD_START[$g]}:${SHARD_END[$g]})"
    done
    exit 0
fi

SHARD_PIDS=()
cleanup() {
    echo "[T6-SH] cleanup: killing shards ${SHARD_PIDS[*]:-}"
    for pid in "${SHARD_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

echo
echo "[T6-SH] launching $GPUS shard clients ..."
for ((g=0; g<GPUS; g++)); do
    S=${SHARD_START[$g]}
    E=${SHARD_END[$g]}
    if [[ "$S" -ge "$E" ]]; then
        echo "[T6-SH]   shard $g empty; skipping"
        continue
    fi
    LOG="$LOG_DIR/shard${g}.log"
    echo "[T6-SH]   shard $g  GPU=$g  prompts[$S:$E)  LOG=$LOG"
    CUDA_VISIBLE_DEVICES=$g PYTHONUNBUFFERED=1 nohup python -u \
        "$ROOT/scripts/validate/t6_teacher_trace.py" \
        --run_dir "$RUN_DIR" \
        --resume \
        --scope_path "$SCOPE_PATH" --scope_group "$SCOPE_GROUP" \
        --n "$MAX_TRAIN" \
        --teacher local --local_model "$TEACHER_CKPT" \
        --retries_per_prompt "$T6_RETRIES" \
        --max_tokens "$T6_MAX_TOKENS" --temperature 0.0 \
        --prompt_start "$S" --prompt_end "$E" \
        --skip_aggregate \
        > "$LOG" 2>&1 &
    SHARD_PIDS+=($!)
done
echo "[T6-SH] shard PIDs: ${SHARD_PIDS[*]}"

if [[ ${#SHARD_PIDS[@]} -eq 0 ]]; then
    echo "[T6-SH] ERROR: no shards launched" >&2
    exit 1
fi

# ── Wait for all shards ───────────────────────────────────────────────────────
echo
echo "[T6-SH] waiting for ${#SHARD_PIDS[@]} shard(s) to finish ..."
echo "[T6-SH]   tail -f $LOG_DIR/shard*.log  to monitor"
EXIT_CODE=0
for pid in "${SHARD_PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "[T6-SH] ERROR: shard PID $pid exited non-zero"
        EXIT_CODE=1
    fi
done

if [[ "$EXIT_CODE" -ne 0 ]]; then
    echo "[T6-SH] one or more shards failed. Not aggregating."
    exit "$EXIT_CODE"
fi

# ── Final aggregate (no slice, no skip_aggregate) ────────────────────────────
echo
echo "[T6-SH] all shards complete. Aggregating to t6_sft.jsonl ..."
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u \
    "$ROOT/scripts/validate/t6_teacher_trace.py" \
    --run_dir "$RUN_DIR" \
    --resume \
    --scope_path "$SCOPE_PATH" --scope_group "$SCOPE_GROUP" \
    --n "$MAX_TRAIN" \
    --teacher local --local_model "$TEACHER_CKPT" \
    --retries_per_prompt "$T6_RETRIES" \
    --max_tokens "$T6_MAX_TOKENS" --temperature 0.0

echo
echo "[T6-SH] DONE. run_dir = $RUN_DIR"
echo "[T6-SH] SFT JSONL    = $RUN_DIR/t6_sft.jsonl"
