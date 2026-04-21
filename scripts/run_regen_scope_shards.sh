#!/usr/bin/env bash
# run_regen_scope_shards.sh — Multi-GPU scope regeneration.
#
# regen_scope.py talks to a single LLaDA serve.py over HTTP. Multi-GPU
# means one serve per GPU + one regen client per serve, sharded over
# the gsm8k test prompts.
#
# Usage:
#   bash scripts/run_regen_scope_shards.sh                    # 8 GPUs
#   bash scripts/run_regen_scope_shards.sh -g 4
#   bash scripts/run_regen_scope_shards.sh --max_prompts 200  # subset
#   bash scripts/run_regen_scope_shards.sh --dry_run

set -euo pipefail

GPUS=8
BASE_PORT=8100                 # avoid colliding with default 8000 (run_all serve)
MAX_PROMPTS=""                 # "" = full gsm8k test (1319)
MIRROR="default"
RUN_DIR=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--gpus)         GPUS="$2";         shift 2 ;;
        -p|--base_port)    BASE_PORT="$2";    shift 2 ;;
        --max_prompts)     MAX_PROMPTS="$2";  shift 2 ;;
        --mirror)          MIRROR="$2";       shift 2 ;;
        -r|--run_dir)      RUN_DIR="$2";      shift 2 ;;
        --dry_run)         DRY_RUN=1;         shift ;;
        -h|--help)         grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)                 echo "[REGEN-SH] unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

[[ -z "$RUN_DIR" ]] && \
    RUN_DIR="$ROOT/runs/validation/scope_regen_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR" "$RUN_DIR/shard_logs" "$RUN_DIR/serve_logs"

echo "[REGEN-SH] ================================================================"
echo "[REGEN-SH]   Multi-GPU scope regeneration"
echo "[REGEN-SH] ================================================================"
echo "[REGEN-SH] GPUS         = $GPUS"
echo "[REGEN-SH] BASE_PORT    = $BASE_PORT  (serves on $BASE_PORT..$((BASE_PORT + GPUS - 1)))"
echo "[REGEN-SH] MAX_PROMPTS  = ${MAX_PROMPTS:-'all 1319'}"
echo "[REGEN-SH] MIRROR       = $MIRROR"
echo "[REGEN-SH] RUN_DIR      = $RUN_DIR"
echo "[REGEN-SH]"

# ── Compute total prompts ─────────────────────────────────────────────────────
TOTAL=$(python - <<PY
from datasets import load_dataset
import os
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
ds = load_dataset("openai/gsm8k", "main", split="test")
n = len(ds)
m = "$MAX_PROMPTS"
if m: n = min(n, int(m))
print(n)
PY
)
if ! [[ "$TOTAL" =~ ^[0-9]+$ ]]; then
    echo "[REGEN-SH] ERROR: failed to count prompts" >&2; exit 1
fi
echo "[REGEN-SH] TOTAL_PROMPTS = $TOTAL"

if [[ "$DRY_RUN" -eq 1 ]]; then
    base=$((TOTAL / GPUS)); rem=$((TOTAL % GPUS))
    acc=0
    echo "[REGEN-SH] DRY RUN \u2014 shard plan:"
    for ((g=0; g<GPUS; g++)); do
        sz=$base; [[ "$g" -lt "$rem" ]] && sz=$((sz + 1))
        e=$((acc + sz))
        echo "  GPU $g  port $((BASE_PORT + g))  prompts[$acc:$e)"
        acc=$e
    done
    exit 0
fi

# ── Pre-check: no port collisions ─────────────────────────────────────────────
for ((g=0; g<GPUS; g++)); do
    P=$((BASE_PORT + g))
    if command -v ss >/dev/null 2>&1; then
        if ss -tlnH "sport = :$P" 2>/dev/null | grep -q ":$P\b"; then
            echo "[REGEN-SH] ERROR: port $P already in use" >&2
            echo "[REGEN-SH] cleanup: pkill -f 'scripts/serve.py'" >&2
            exit 1
        fi
    fi
done

# ── Launch serves ─────────────────────────────────────────────────────────────
SERVE_PIDS=()
SHARD_PIDS=()
cleanup() {
    echo "[REGEN-SH] cleanup: killing serves ${SERVE_PIDS[*]:-} + shards ${SHARD_PIDS[*]:-}"
    for pid in "${SHARD_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${SERVE_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

for ((g=0; g<GPUS; g++)); do
    P=$((BASE_PORT + g))
    LOG="$RUN_DIR/serve_logs/serve_gpu${g}.log"
    echo "[REGEN-SH] launching serve: GPU=$g  port=$P  LOG=$LOG"
    CUDA_VISIBLE_DEVICES=$g nohup python "$ROOT/scripts/serve.py" \
        --model_id GSAI-ML/LLaDA-8B-Instruct \
        --port "$P" --host 127.0.0.1 \
        > "$LOG" 2>&1 &
    SERVE_PIDS+=($!)
done

# Wait for each serve to be ready
echo "[REGEN-SH] waiting for serves..."
for ((g=0; g<GPUS; g++)); do
    P=$((BASE_PORT + g))
    waited=0
    until curl -sf "http://127.0.0.1:$P/health" > /dev/null 2>&1; do
        if ! kill -0 "${SERVE_PIDS[$g]}" 2>/dev/null; then
            echo "[REGEN-SH] ERROR: serve on GPU $g died; see $RUN_DIR/serve_logs/serve_gpu${g}.log"
            exit 1
        fi
        sleep 5; waited=$((waited + 5))
        if [[ "$waited" -ge 600 ]]; then
            echo "[REGEN-SH] ERROR: serve on GPU $g not ready after 10 min"; exit 1
        fi
    done
    echo "[REGEN-SH]   GPU $g  port $P  ready (${waited}s)"
done

# ── Compute shard slices ──────────────────────────────────────────────────────
declare -a SS SE
base=$((TOTAL / GPUS)); rem=$((TOTAL % GPUS)); acc=0
for ((g=0; g<GPUS; g++)); do
    sz=$base; [[ "$g" -lt "$rem" ]] && sz=$((sz + 1))
    SS[$g]=$acc
    acc=$((acc + sz))
    SE[$g]=$acc
done

# ── Launch shard clients ──────────────────────────────────────────────────────
echo "[REGEN-SH] launching $GPUS shard clients..."
for ((g=0; g<GPUS; g++)); do
    s=${SS[$g]}; e=${SE[$g]}
    [[ "$s" -ge "$e" ]] && { echo "[REGEN-SH]   shard $g empty, skip"; continue; }
    P=$((BASE_PORT + g))
    LOG="$RUN_DIR/shard_logs/shard${g}.log"
    echo "[REGEN-SH]   shard $g  prompts[$s:$e)  serve=:$P  LOG=$LOG"
    PYTHONUNBUFFERED=1 nohup python -u \
        "$ROOT/scripts/validate/regen_scope.py" \
        --run_dir "$RUN_DIR" \
        --resume \
        --server_url "http://127.0.0.1:$P" \
        --mirror "$MIRROR" \
        ${MAX_PROMPTS:+--max_prompts "$MAX_PROMPTS"} \
        --prompt_start "$s" --prompt_end "$e" \
        --skip_aggregate \
        > "$LOG" 2>&1 &
    SHARD_PIDS+=($!)
done

# ── Wait shards ───────────────────────────────────────────────────────────────
echo "[REGEN-SH] waiting for ${#SHARD_PIDS[@]} shards..."
EXIT_CODE=0
for pid in "${SHARD_PIDS[@]}"; do
    wait "$pid" || { echo "[REGEN-SH] shard pid $pid failed"; EXIT_CODE=1; }
done
[[ "$EXIT_CODE" -ne 0 ]] && { echo "[REGEN-SH] some shard failed"; exit 1; }

# ── Aggregate (no slice, no skip_aggregate) ───────────────────────────────────
echo "[REGEN-SH] aggregating to scope_fail/ok..."
PYTHONUNBUFFERED=1 python -u \
    "$ROOT/scripts/validate/regen_scope.py" \
    --run_dir "$RUN_DIR" --resume \
    --server_url "http://127.0.0.1:$BASE_PORT" \
    --mirror "$MIRROR" \
    ${MAX_PROMPTS:+--max_prompts "$MAX_PROMPTS"}

echo "[REGEN-SH] DONE. run_dir = $RUN_DIR"
