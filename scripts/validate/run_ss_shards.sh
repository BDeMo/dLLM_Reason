#!/usr/bin/env bash
# run_ss_shards.sh — Multi-GPU orchestrator for strategy_search.py
#
# Scheme A (shared run_dir, prompt sharding across GPUs):
#   1. Launch one FastAPI server per GPU, each pinned via CUDA_VISIBLE_DEVICES
#      and listening on a distinct port.
#   2. Launch one strategy_search.py client per server, each handling a disjoint
#      slice of the prompts list via --prompt_start/--prompt_end. All shards
#      write to the SAME run_dir (per_prompt/{group}_{idx}.json files don't
#      collide; each prompt is handled by exactly one shard).
#   3. Shard clients run with --skip_summary so nobody races on writing
#      winners.json / summary.json.
#   4. After all shards finish, run ONE final client with no slice + --resume
#      + no --skip_summary. Its work-loop becomes a no-op (all per_prompt
#      files exist), but it aggregates them into the global summary.
#
# Usage:
#   bash scripts/validate/run_ss_shards.sh              # default: 8 GPUs
#   bash scripts/validate/run_ss_shards.sh -g 4         # only 4 GPUs
#   bash scripts/validate/run_ss_shards.sh -n 30        # only first 30 prompts per group
#   bash scripts/validate/run_ss_shards.sh --groups fail
#
# Additional flags after `--` are forwarded to the strategy_search.py client:
#   bash scripts/validate/run_ss_shards.sh -- --values temperature=0.0
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
GPUS=8
BASE_PORT=8000
N_PER_GROUP=60
PROMPT_GROUPS="fail,ok"
RUN_DIR=""
SERVER_MODEL="${SERVER_MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
SERVER_READY_TIMEOUT=600  # seconds to wait for server /health
LOG_DIR=""                # auto-set from RUN_DIR below

# ── CLI parsing ───────────────────────────────────────────────────────────────
POSITIONAL_EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--gpus)        GPUS="$2"; shift 2 ;;
        -p|--base_port)   BASE_PORT="$2"; shift 2 ;;
        -n|--n)           N_PER_GROUP="$2"; shift 2 ;;
        --groups)         PROMPT_GROUPS="$2"; shift 2 ;;
        -r|--run_dir)     RUN_DIR="$2"; shift 2 ;;
        --model)          SERVER_MODEL="$2"; shift 2 ;;
        --)               shift; POSITIONAL_EXTRA=("$@"); break ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//' ; exit 0 ;;
        *)
            echo "[SS-SH] unknown arg: $1" >&2 ; exit 1 ;;
    esac
done

# ── Repo root (assumes this script is in scripts/validate/) ──────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

# ── run_dir: if not set, generate timestamped path aligned with _runlib.py ───
if [[ -z "$RUN_DIR" ]]; then
    TS="$(date +%Y%m%d_%H%M%S)"
    RUN_DIR="$ROOT/runs/validation/strategy_search_${TS}"
fi
mkdir -p "$RUN_DIR"
LOG_DIR="$RUN_DIR/shard_logs"
mkdir -p "$LOG_DIR"

echo "[SS-SH] ================================================================"
echo "[SS-SH]   Multi-GPU strategy_search orchestrator"
echo "[SS-SH] ================================================================"
echo "[SS-SH] GPUS         = $GPUS"
echo "[SS-SH] BASE_PORT    = $BASE_PORT  (servers on $BASE_PORT..$((BASE_PORT + GPUS - 1)))"
echo "[SS-SH] N_PER_GROUP  = $N_PER_GROUP"
echo "[SS-SH] PROMPT_GROUPS       = $PROMPT_GROUPS"
echo "[SS-SH] RUN_DIR      = $RUN_DIR"
echo "[SS-SH] MODEL        = $SERVER_MODEL"
echo "[SS-SH] EXTRA_ARGS   = ${POSITIONAL_EXTRA[*]:-<none>}"
echo "[SS-SH] LOG_DIR      = $LOG_DIR"
echo "[SS-SH]"

# ── Compute total prompts (need to match load_prompts semantics in script) ────
# load_prompts reads scope_fail_prompts.json[:n] + scope_ok_prompts.json[:n].
# We recompute here via python so sharding matches exactly.
TOTAL_PROMPTS=$(python - <<PY 2>&1
import json, pathlib, sys
root = pathlib.Path("$ROOT")
n = int("$N_PER_GROUP")
groups = "$PROMPT_GROUPS".split(",")
total = 0
missing = []
if "fail" in groups:
    p = root / "runs/validation/scope_fail_prompts.json"
    if not p.exists():
        missing.append(str(p))
    else:
        total += len(json.loads(p.read_text())[:n])
if "ok" in groups:
    p = root / "runs/validation/scope_ok_prompts.json"
    if not p.exists():
        missing.append(str(p))
    else:
        total += len(json.loads(p.read_text())[:n])
if missing:
    sys.stderr.write("MISSING_FILES: " + ";".join(missing) + "\n")
    sys.exit(2)
print(total)
PY
)
# Validate we got a non-empty integer.
if ! [[ "$TOTAL_PROMPTS" =~ ^[0-9]+$ ]]; then
    echo "[SS-SH] ERROR: failed to compute TOTAL_PROMPTS. Python output was:" >&2
    echo "----" >&2
    echo "$TOTAL_PROMPTS" >&2
    echo "----" >&2
    echo "[SS-SH] Check that scope_fail_prompts.json / scope_ok_prompts.json exist under" >&2
    echo "[SS-SH]   $ROOT/runs/validation/" >&2
    echo "[SS-SH] These are produced by scripts/validate/h0_forensics.py." >&2
    exit 1
fi
echo "[SS-SH] total prompts to shard = $TOTAL_PROMPTS"

if [[ "$TOTAL_PROMPTS" -eq 0 ]]; then
    echo "[SS-SH] ERROR: 0 prompts resolved from groups='$PROMPT_GROUPS' with --n $N_PER_GROUP" >&2
    echo "[SS-SH] scope files may be empty JSON arrays. Rerun h0_forensics.py or fix --groups/--n." >&2
    exit 1
fi

if [[ "$TOTAL_PROMPTS" -lt "$GPUS" ]]; then
    echo "[SS-SH] WARN: fewer prompts ($TOTAL_PROMPTS) than GPUs ($GPUS); "
    echo "[SS-SH]       some shards will be empty (cheap no-ops)."
fi

# ── Checkpoint-patch warmup (single-process, idempotent) ─────────────────────
# First-time transformers-5.x patch rewrites modeling_llada.py in the local
# checkpoint and clears the HF dynamic-module cache. Doing this single-threaded
# BEFORE we fan out to 8 concurrent server starts avoids a write race on:
#   - $CKPT/modeling_llada.py
#   - ~/.cache/huggingface/modules/transformers_modules/llada*/
# Second run onward is a no-op (sentinel check exits early).
echo "[SS-SH]"
echo "[SS-SH] warming up LLaDA checkpoint patch (single-process) ..."
if ! python - <<PY
import sys
try:
    from dllm_reason.utils.local_resolve import resolve_model_path
    from dllm_reason.utils.llada_checkpoint_patch import ensure_llada_checkpoint_patched
except ImportError as e:
    sys.stderr.write(f"[warmup] ImportError: {e}\n")
    sys.stderr.write("[warmup] Did you 'pip install -r requirements.txt' after the latest pull?\n")
    sys.exit(1)

ckpt = resolve_model_path("$SERVER_MODEL")
applied = ensure_llada_checkpoint_patched(ckpt)
print(f"[warmup] checkpoint: {ckpt}")
print(f"[warmup] patch applied this call: {applied}  (False = already patched or no-op)")
PY
then
    echo "[SS-SH] warmup failed; aborting." >&2
    exit 1
fi

# ── Port pre-check ────────────────────────────────────────────────────────────
# Catch orphaned servers from previous runs before we try to bind. Avoids the
# silent case where serve.py loads the whole model (~5s) and THEN fails on
# bind — burns memory + confuses the health-poll timer.
PORT_BUSY=()
for ((g=0; g<GPUS; g++)); do
    CHECK_PORT=$((BASE_PORT + g))
    # Prefer `ss` (iproute2, always available on modern Linux); fall back to lsof.
    if command -v ss >/dev/null 2>&1; then
        if ss -tlnH "sport = :$CHECK_PORT" 2>/dev/null | grep -q ":$CHECK_PORT\b"; then
            PORT_BUSY+=("$CHECK_PORT")
        fi
    elif command -v lsof >/dev/null 2>&1; then
        if lsof -iTCP:"$CHECK_PORT" -sTCP:LISTEN -t 2>/dev/null | grep -q .; then
            PORT_BUSY+=("$CHECK_PORT")
        fi
    else
        echo "[SS-SH] WARN: neither ss nor lsof found; skipping port pre-check"
        break
    fi
done
if [[ ${#PORT_BUSY[@]} -gt 0 ]]; then
    echo "[SS-SH] ERROR: ports already in use: ${PORT_BUSY[*]}" >&2
    echo "[SS-SH] Likely orphaned servers from a previous run. Clean up with:" >&2
    echo "[SS-SH]   pkill -f 'scripts/serve.py'" >&2
    echo "[SS-SH] Or inspect with:" >&2
    echo "[SS-SH]   ss -tlnp | grep -E ':($(IFS='|'; echo "${PORT_BUSY[*]}"))\\b'" >&2
    echo "[SS-SH] Alternatively, pass -p <base_port> to shift to a free range." >&2
    exit 1
fi

# ── Launch servers ────────────────────────────────────────────────────────────
SERVER_PIDS=()
cleanup() {
    echo "[SS-SH] cleanup: killing servers ${SERVER_PIDS[*]:-}"
    for pid in "${SERVER_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

for ((g=0; g<GPUS; g++)); do
    PORT=$((BASE_PORT + g))
    LOG="$LOG_DIR/server_gpu${g}.log"
    echo "[SS-SH] launching server: GPU=$g PORT=$PORT LOG=$LOG"
    CUDA_VISIBLE_DEVICES=$g nohup python "$ROOT/scripts/serve.py" \
        --model_id "$SERVER_MODEL" \
        --port "$PORT" \
        --host 127.0.0.1 \
        > "$LOG" 2>&1 &
    SERVER_PIDS+=($!)
done
echo "[SS-SH] server PIDs: ${SERVER_PIDS[*]}"

# ── Wait for each server to respond /health ──────────────────────────────────
echo "[SS-SH] waiting for servers to become ready (timeout ${SERVER_READY_TIMEOUT}s each) ..."
for ((g=0; g<GPUS; g++)); do
    PORT=$((BASE_PORT + g))
    URL="http://127.0.0.1:$PORT/health"
    WAITED=0
    until curl -sf "$URL" > /dev/null 2>&1; do
        if ! kill -0 "${SERVER_PIDS[$g]}" 2>/dev/null; then
            echo "[SS-SH] ERROR: server on GPU $g (port $PORT) died; check $LOG_DIR/server_gpu${g}.log"
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
        if [[ "$WAITED" -ge "$SERVER_READY_TIMEOUT" ]]; then
            echo "[SS-SH] ERROR: server on GPU $g (port $PORT) not ready after ${SERVER_READY_TIMEOUT}s"
            exit 1
        fi
    done
    echo "[SS-SH]   GPU $g  port $PORT  ready (${WAITED}s)"
done

# ── Compute shard slices ──────────────────────────────────────────────────────
# Even split with remainder distributed to the first shards.
# Shard g covers [start_g, end_g) of the full prompts list.
declare -a SHARD_START SHARD_END
base=$((TOTAL_PROMPTS / GPUS))
rem=$((TOTAL_PROMPTS % GPUS))
acc=0
for ((g=0; g<GPUS; g++)); do
    sz=$base
    if [[ "$g" -lt "$rem" ]]; then sz=$((sz + 1)); fi
    SHARD_START[$g]=$acc
    acc=$((acc + sz))
    SHARD_END[$g]=$acc
done

# ── Launch shard clients ──────────────────────────────────────────────────────
SHARD_PIDS=()
echo "[SS-SH]"
echo "[SS-SH] launching $GPUS shard clients ..."
for ((g=0; g<GPUS; g++)); do
    PORT=$((BASE_PORT + g))
    S=${SHARD_START[$g]}
    E=${SHARD_END[$g]}
    if [[ "$S" -ge "$E" ]]; then
        echo "[SS-SH]   shard $g is empty (prompts[$S:$E]); skipping"
        continue
    fi
    LOG="$LOG_DIR/client_shard${g}.log"
    echo "[SS-SH]   shard $g  prompts[$S:$E)  server=http://127.0.0.1:$PORT  LOG=$LOG"
    nohup python "$ROOT/scripts/validate/strategy_search.py" \
        --run_dir "$RUN_DIR" \
        --resume \
        --server_url "http://127.0.0.1:$PORT" \
        --n "$N_PER_GROUP" \
        --groups "$PROMPT_GROUPS" \
        --prompt_start "$S" \
        --prompt_end "$E" \
        --skip_summary \
        "${POSITIONAL_EXTRA[@]}" \
        > "$LOG" 2>&1 &
    SHARD_PIDS+=($!)
done
echo "[SS-SH] shard PIDs: ${SHARD_PIDS[*]}"

# ── Wait for shards ──────────────────────────────────────────────────────────
if [[ ${#SHARD_PIDS[@]} -eq 0 ]]; then
    echo "[SS-SH] ERROR: no shards were launched. This should not happen unless "
    echo "[SS-SH]        TOTAL_PROMPTS=0 (already bailed) or all shards computed "
    echo "[SS-SH]        to empty slices. Aborting." >&2
    exit 1
fi

echo "[SS-SH]"
echo "[SS-SH] waiting for ${#SHARD_PIDS[@]} shard(s) to finish ... "
echo "[SS-SH]   tail -f $LOG_DIR/client_shard*.log to monitor"
EXIT_CODE=0
for pid in "${SHARD_PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "[SS-SH] ERROR: shard PID $pid exited non-zero"
        EXIT_CODE=1
    fi
done

if [[ "$EXIT_CODE" -ne 0 ]]; then
    echo "[SS-SH] one or more shards failed. Not aggregating. Inspect $LOG_DIR/client_shard*.log"
    exit "$EXIT_CODE"
fi

# ── Final aggregate pass (no slice, no skip_summary) ──────────────────────────
echo "[SS-SH]"
echo "[SS-SH] all shards complete. running final aggregation pass ..."
python "$ROOT/scripts/validate/strategy_search.py" \
    --run_dir "$RUN_DIR" \
    --resume \
    --server_url "http://127.0.0.1:$BASE_PORT" \
    --n "$N_PER_GROUP" \
    --groups "$PROMPT_GROUPS" \
    "${POSITIONAL_EXTRA[@]}"

echo "[SS-SH]"
echo "[SS-SH] DONE. run_dir = $RUN_DIR"
