#!/usr/bin/env bash
# run_v1.6.sh — End-to-end execution script for v1.6 (T6 + T7 SFT)
#
# Walks through the full pipeline with checkpoints between each stage, so
# you can re-run to pick up where you left off:
#
#   Phase 0  Dependencies + downloads (Qwen teacher + GSM8K train)
#   Phase 1  T7: LLaDA self-sample correct → SFT stage 1
#   Phase 2  T6: Qwen teacher trace → SFT stage 2 (warm-start from Phase 1)
#   Phase 3  Eval on held-out 60 fail + 49 ok
#   Phase 4  Archive results to docs/archive/
#
# Each phase has a "check" step that verifies outputs exist + non-empty
# before the next phase runs. Safe to interrupt and re-run.
#
# Usage:
#   bash scripts/run_v1.6.sh              # defaults (4B teacher, 2000 prompts)
#   bash scripts/run_v1.6.sh --mirror hf-mirror
#   bash scripts/run_v1.6.sh --teacher_size 3.5-9B --max_train 5000
#   bash scripts/run_v1.6.sh --from_phase 2   # skip phases 0-1, start from T6
#   bash scripts/run_v1.6.sh --dry_run        # print plan, no execution
#   bash scripts/run_v1.6.sh --check_only     # verify artifacts, no work

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MIRROR="default"                # default / hf-mirror / modelscope / full URL
TEACHER_SIZE="3.5-4B"           # see download_qwen.py CANDIDATES for labels
TEACHER_CKPT=""                 # auto-resolved from TEACHER_SIZE below
MAX_TRAIN=2000                  # how many gsm8k train samples to use
T7_TEMPS="0.3,0.7,1.0"
T7_N_SAMPLES=8
T7_GEN_LENGTH=192
T6_RETRIES=3
T7_MAX_STEPS=2000
T6_MAX_STEPS=1500
BATCH_SIZE=4
GRAD_ACCUM=4
SERVER_URL="http://127.0.0.1:8000"
FROM_PHASE=0
TO_PHASE=4
DRY_RUN=0
CHECK_ONLY=0

# ── CLI ───────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mirror)         MIRROR="$2";         shift 2 ;;
        --teacher_size)   TEACHER_SIZE="$2";   shift 2 ;;
        --teacher_ckpt)   TEACHER_CKPT="$2";   shift 2 ;;
        --max_train)      MAX_TRAIN="$2";      shift 2 ;;
        --t7_max_steps)   T7_MAX_STEPS="$2";   shift 2 ;;
        --t6_max_steps)   T6_MAX_STEPS="$2";   shift 2 ;;
        --batch_size)     BATCH_SIZE="$2";     shift 2 ;;
        --grad_accum)     GRAD_ACCUM="$2";     shift 2 ;;
        --server_url)     SERVER_URL="$2";     shift 2 ;;
        --from_phase)     FROM_PHASE="$2";     shift 2 ;;
        --to_phase)       TO_PHASE="$2";       shift 2 ;;
        --dry_run)        DRY_RUN=1;           shift ;;
        --check_only)     CHECK_ONLY=1;        shift ;;
        -h|--help)        grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)                echo "[V1.6] unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

GSM8K_TRAIN_JSON="$ROOT/runs/validation/gsm8k_train_prompts.json"
T7_RUN_DIR=""  # resolved post-run to the timestamped dir
T6_RUN_DIR=""
T7_SFT_JSONL=""
T6_SFT_JSONL=""
T7_CKPT_DIR="$ROOT/runs/training/v16_t7_stage1"
T6_CKPT_DIR="$ROOT/runs/training/v16_t6_stage2"
EVAL_DIR="$ROOT/runs/validation/v16_eval"

if [[ -z "$TEACHER_CKPT" ]]; then
    # Qwen3.5 repos have NO '-Instruct' suffix (verified on HF 2026-04-19);
    # the default snapshot IS the chat model. Resolve via download_qwen.py's
    # CANDIDATES mapping by asking it to dry-print the plan.
    RESOLVED=$(python scripts/download_qwen.py --sizes "$TEACHER_SIZE" --dry_run 2>&1 \
                 | grep -oE 'checkpoints[\\/]+[A-Za-z0-9._]+[-\._A-Za-z0-9]*' \
                 | head -1)
    if [[ -n "$RESOLVED" ]]; then
        TEACHER_CKPT="$ROOT/${RESOLVED//\\//}"   # normalise any backslashes on win
    else
        # Fallback: naive construction for sizes like "3.5-4B" → Qwen3.5-4B
        TEACHER_CKPT="$ROOT/checkpoints/Qwen__Qwen${TEACHER_SIZE}"
    fi
fi

# ── Logging helpers ───────────────────────────────────────────────────────────
hdr() {
    echo
    echo "══════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "══════════════════════════════════════════════════════════════"
}

run_or_dry() {
    local desc="$1"; shift
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY-RUN] $desc"
        echo "         $*"
        return 0
    fi
    echo "[RUN] $desc"
    echo "      $*"
    eval "$@"
}

check_file() {
    local path="$1" ; local desc="$2" ; local min_bytes="${3:-1}"
    if [[ ! -f "$path" ]]; then
        echo "  ✗ missing: $desc  ($path)"
        return 1
    fi
    local sz
    sz=$(stat -c%s "$path" 2>/dev/null || stat -f%z "$path" 2>/dev/null || echo 0)
    if [[ "$sz" -lt "$min_bytes" ]]; then
        echo "  ✗ too small ($sz < $min_bytes): $desc"
        return 1
    fi
    echo "  ✓ $desc  ($sz bytes)"
    return 0
}

check_dir_nonempty() {
    local path="$1" ; local desc="$2"
    if [[ ! -d "$path" ]]; then
        echo "  ✗ missing dir: $desc  ($path)"
        return 1
    fi
    if ! find "$path" -type f -size +0c -print -quit | grep -q .; then
        echo "  ✗ empty dir: $desc  ($path)"
        return 1
    fi
    echo "  ✓ $desc  ($path)"
    return 0
}

# ── Banner ────────────────────────────────────────────────────────────────────
hdr "v1.6 Pipeline (T6 + T7 SFT)"
cat <<EOF
  ROOT            = $ROOT
  MIRROR          = $MIRROR
  TEACHER_SIZE    = $TEACHER_SIZE
  TEACHER_CKPT    = $TEACHER_CKPT
  MAX_TRAIN       = $MAX_TRAIN (gsm8k train subset)
  PHASES          = $FROM_PHASE ... $TO_PHASE
  DRY_RUN         = $DRY_RUN   CHECK_ONLY = $CHECK_ONLY
  T7              temps=$T7_TEMPS  N=$T7_N_SAMPLES  gen=$T7_GEN_LENGTH  steps=$T7_MAX_STEPS
  T6              retries=$T6_RETRIES  steps=$T6_MAX_STEPS
  SFT             bs=$BATCH_SIZE  grad_accum=$GRAD_ACCUM  (effective=$((BATCH_SIZE * GRAD_ACCUM)))
EOF

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 0 — Dependencies: Qwen teacher + gsm8k train data
# ═════════════════════════════════════════════════════════════════════════════
phase_0() {
    hdr "Phase 0 — Downloads + data prep"

    # 0a: Qwen teacher
    run_or_dry "Download Qwen teacher ($TEACHER_SIZE)" \
        python scripts/download_qwen.py \
        --sizes "$TEACHER_SIZE" --mirror "$MIRROR"

    # 0b: gsm8k train
    run_or_dry "Load gsm8k train prompts (max $MAX_TRAIN)" \
        python scripts/validate/load_gsm8k_train.py \
        --mirror "$MIRROR" --max_samples "$MAX_TRAIN"

    # 0c: verify
    echo
    echo "[CHECK Phase 0 outputs]"
    local ok=1
    python scripts/download_qwen.py --sizes "$TEACHER_SIZE" --check_only \
        --min_weights_gb 1.0 || ok=0
    check_file "$GSM8K_TRAIN_JSON" "gsm8k train JSON" 1000 || ok=0
    [[ "$ok" -eq 1 ]] && echo "[PHASE 0] ✓ PASS" || { echo "[PHASE 0] ✗ FAIL"; exit 1; }
}

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — T7 self-distill: sample + filter + SFT
# ═════════════════════════════════════════════════════════════════════════════
phase_1() {
    hdr "Phase 1 — T7 self-distill"

    # 1a: require serve.py up on SERVER_URL
    if [[ "$CHECK_ONLY" -eq 0 && "$DRY_RUN" -eq 0 ]]; then
        if ! curl -sf "$SERVER_URL/health" > /dev/null 2>&1; then
            echo "[PHASE 1] ERROR: server not reachable at $SERVER_URL"
            echo "  Start: python scripts/serve.py --port 8000"
            exit 1
        fi
        echo "[PHASE 1] server OK at $SERVER_URL"
    fi

    # 1b: T7 data generation
    run_or_dry "T7 sampling + filter correct" \
        python scripts/validate/t7_gen_correct_samples.py \
        --scope_path "$GSM8K_TRAIN_JSON" --scope_group gsm8k \
        --n "$MAX_TRAIN" \
        --temperatures "$T7_TEMPS" --n_samples "$T7_N_SAMPLES" \
        --gen_length "$T7_GEN_LENGTH" --block_length 32 \
        --pick shortest \
        --server_url "$SERVER_URL"

    # Resolve latest T7 run dir
    T7_RUN_DIR=$(ls -dt "$ROOT"/runs/validation/t7_selfdistill_* 2>/dev/null | head -1)
    T7_SFT_JSONL="$T7_RUN_DIR/t7_sft.jsonl"

    # 1c: T7 SFT
    run_or_dry "T7 SFT (stage 1)" \
        python scripts/validate/t6t7_train.py \
        --jsonl_path "$T7_SFT_JSONL" \
        --run_name v16_t7_stage1 \
        --max_steps "$T7_MAX_STEPS" \
        --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" \
        --lr 2e-5

    # 1d: verify
    echo
    echo "[CHECK Phase 1 outputs]"
    local ok=1
    check_file "$T7_SFT_JSONL" "T7 SFT JSONL" 100 || ok=0
    check_dir_nonempty "$T7_CKPT_DIR" "T7 stage1 ckpt dir" || ok=0
    [[ "$ok" -eq 1 ]] && echo "[PHASE 1] ✓ PASS" || { echo "[PHASE 1] ✗ FAIL"; exit 1; }
}

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — T6 canvas distill: teacher trace + SFT warm-start
# ═════════════════════════════════════════════════════════════════════════════
phase_2() {
    hdr "Phase 2 — T6 canvas distill (warm-start from T7)"

    # 2a: teacher trace
    run_or_dry "T6 teacher trace generation" \
        python scripts/validate/t6_teacher_trace.py \
        --scope_path "$GSM8K_TRAIN_JSON" --scope_group gsm8k \
        --n "$MAX_TRAIN" \
        --teacher local --local_model "$TEACHER_CKPT" \
        --retries_per_prompt "$T6_RETRIES" \
        --max_tokens 800 --temperature 0.0

    # Resolve latest T6 run dir
    T6_RUN_DIR=$(ls -dt "$ROOT"/runs/validation/t6_teacher_trace_* 2>/dev/null | head -1)
    T6_SFT_JSONL="$T6_RUN_DIR/t6_sft.jsonl"

    # 2b: T6 SFT warm-start
    local init_ckpt="$T7_CKPT_DIR/best.pt"
    if [[ ! -f "$init_ckpt" && -f "$T7_CKPT_DIR/step_${T7_MAX_STEPS}.pt" ]]; then
        init_ckpt="$T7_CKPT_DIR/step_${T7_MAX_STEPS}.pt"
    fi
    run_or_dry "T6 SFT (stage 2, warm-start)" \
        python scripts/validate/t6t7_train.py \
        --jsonl_path "$T6_SFT_JSONL" \
        --run_name v16_t6_stage2 \
        --init_ckpt "$init_ckpt" \
        --max_steps "$T6_MAX_STEPS" \
        --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" \
        --lr 1e-5

    # 2c: verify
    echo
    echo "[CHECK Phase 2 outputs]"
    local ok=1
    check_file "$T6_SFT_JSONL" "T6 SFT JSONL" 100 || ok=0
    check_dir_nonempty "$T6_CKPT_DIR" "T6 stage2 ckpt dir" || ok=0
    [[ "$ok" -eq 1 ]] && echo "[PHASE 2] ✓ PASS" || { echo "[PHASE 2] ✗ FAIL"; exit 1; }
}

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — Eval on held-out test set
# ═════════════════════════════════════════════════════════════════════════════
phase_3() {
    hdr "Phase 3 — Eval on held-out scope_fail + scope_ok"
    echo "[PHASE 3] TODO: wire a dedicated eval runner (next commit)."
    echo "[PHASE 3] For now, manually eval by spinning up serve with the new ckpt:"
    echo "   python scripts/serve.py --model_id $T6_CKPT_DIR/best.pt --port 8000"
    echo "   python scripts/validate/h3_passN_at_temperature.py --n 60 --temperatures 0"
}

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — Archive
# ═════════════════════════════════════════════════════════════════════════════
phase_4() {
    hdr "Phase 4 — Archive results"
    echo "[PHASE 4] TODO: write to docs/archive/finding_v1.6_*.zh.md"
    echo "[PHASE 4] Inputs needed:"
    echo "          - Phase 3 eval numbers"
    echo "          - FAIL18 / ceiling5 rescue delta"
    echo "          - Training loss curves"
}

# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════
if [[ "$CHECK_ONLY" -eq 1 ]]; then
    hdr "Check-only mode"
    local ok=1
    python scripts/download_qwen.py --sizes "$TEACHER_SIZE" --check_only || ok=0
    check_file "$GSM8K_TRAIN_JSON" "gsm8k train JSON" 1000 || ok=0
    [[ -d "$T7_CKPT_DIR" ]] && check_dir_nonempty "$T7_CKPT_DIR" "T7 ckpt" || true
    [[ -d "$T6_CKPT_DIR" ]] && check_dir_nonempty "$T6_CKPT_DIR" "T6 ckpt" || true
    exit $([[ "$ok" -eq 1 ]] && echo 0 || echo 1)
fi

[[ "$FROM_PHASE" -le 0 && "$TO_PHASE" -ge 0 ]] && phase_0
[[ "$FROM_PHASE" -le 1 && "$TO_PHASE" -ge 1 ]] && phase_1
[[ "$FROM_PHASE" -le 2 && "$TO_PHASE" -ge 2 ]] && phase_2
[[ "$FROM_PHASE" -le 3 && "$TO_PHASE" -ge 3 ]] && phase_3
[[ "$FROM_PHASE" -le 4 && "$TO_PHASE" -ge 4 ]] && phase_4

hdr "v1.6 pipeline done"
