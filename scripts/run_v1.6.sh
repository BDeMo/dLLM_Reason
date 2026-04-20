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
#   bash scripts/run_v1.6.sh              # defaults: T6 only, single GPU
#   bash scripts/run_v1.6.sh --mirror hf-mirror
#   bash scripts/run_v1.6.sh --t6_gpus 8  # multi-GPU teacher trace gen
#   bash scripts/run_v1.6.sh --teacher_size 3.5-9B --max_train 5000
#   bash scripts/run_v1.6.sh --smoke      # fast validation: 200 prompts
#   bash scripts/run_v1.6.sh --include_t7 # legacy: also run T7 (deprecated)
#   bash scripts/run_v1.6.sh --from_phase 1   # skip Phase 0 downloads
#   bash scripts/run_v1.6.sh --dry_run
#   bash scripts/run_v1.6.sh --check_only
#
# Scope change from earlier drafts:
#   v1.6 is T6-ONLY (canvas distill from AR teacher). T7 (self-distill)
#   moved to a future release. Pass --include_t7 to opt back in.
#
# Multi-GPU:
#   --t6_gpus N launches N local-Qwen instances for parallel teacher
#   trace generation (each on CUDA_VISIBLE_DEVICES=0..N-1). T6 SFT itself
#   remains single-GPU for now (LLaDA-8B fits on one A100 80GB at bs=4).
#
# Auto behavior:
#   If Phase 1 (T6) not in active range, Qwen download auto-skipped.
#   If no training phase in range, gsm8k download auto-skipped.

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
TO_PHASE=3                      # v1.6 stops at eval; T7 dropped for now
DRY_RUN=0
CHECK_ONLY=0
SMOKE=0
SKIP_QWEN=0
SKIP_GSM8K=0
T6_GPUS=1                       # single GPU default; set --t6_gpus 8 for shards
INCLUDE_T7=0                    # explicit opt-in to legacy T7 phase

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
        --skip_qwen)      SKIP_QWEN=1;         shift ;;
        --skip_gsm8k)     SKIP_GSM8K=1;        shift ;;
        --smoke)          SMOKE=1;             shift ;;
        --t6_gpus)        T6_GPUS="$2";        shift 2 ;;
        --include_t7)     INCLUDE_T7=1; TO_PHASE=4; shift ;;
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
T6_CKPT_DIR="$ROOT/runs/training/v16_t6"
T7_CKPT_DIR="$ROOT/runs/training/v16_t7"
EVAL_DIR="$ROOT/runs/validation/v16_eval"

# Apply --smoke preset (small, fast pipeline validation)
if [[ "$SMOKE" -eq 1 ]]; then
    MAX_TRAIN=${MAX_TRAIN:-200}
    [[ "$MAX_TRAIN" -gt 200 ]] && MAX_TRAIN=200
    T7_TEMPS="0.7"
    T7_N_SAMPLES=4
    T7_MAX_STEPS=500
    T6_MAX_STEPS=500
    T6_RETRIES=2
    echo "[V1.6] --smoke preset: MAX_TRAIN=$MAX_TRAIN  T7_N=$T7_N_SAMPLES  "
    echo "        T7_temps=$T7_TEMPS  T7_steps=$T7_MAX_STEPS  T6_steps=$T6_MAX_STEPS"
fi

# Auto-skip Qwen download if T6 is NOT in the active phase range (Phase 1).
# T7 (Phase 2) doesn't need Qwen, so skipping it saves ~8GB for T7-only runs.
if [[ "$FROM_PHASE" -gt 1 || "$TO_PHASE" -lt 1 ]] && [[ "$SKIP_QWEN" -eq 0 ]]; then
    SKIP_QWEN=1
    echo "[V1.6] T6 (Phase 1) not in active range \u2192 auto-skip Qwen download"
fi

# Auto-skip gsm8k download if neither T6 (Phase 1) nor T7 (Phase 2) run
if [[ "$FROM_PHASE" -gt 2 || "$TO_PHASE" -lt 1 ]] && [[ "$SKIP_GSM8K" -eq 0 ]]; then
    SKIP_GSM8K=1
    echo "[V1.6] Neither T6 nor T7 in active range \u2192 auto-skip gsm8k download"
fi

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
  T6              retries=$T6_RETRIES  steps=$T6_MAX_STEPS  gpus=$T6_GPUS
  INCLUDE_T7      = $INCLUDE_T7  (T7 N=$T7_N_SAMPLES steps=$T7_MAX_STEPS if enabled)
  SFT             bs=$BATCH_SIZE  grad_accum=$GRAD_ACCUM  (effective=$((BATCH_SIZE * GRAD_ACCUM)))
EOF

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 0 — Dependencies: Qwen teacher + gsm8k train data
# ═════════════════════════════════════════════════════════════════════════════
phase_0() {
    hdr "Phase 0 — Downloads + data prep"

    # 0a: Qwen teacher (only if T6 will run)
    if [[ "$SKIP_QWEN" -eq 1 ]]; then
        echo "[PHASE 0] skip Qwen download (T6 not in active phase range)"
    else
        run_or_dry "Download Qwen teacher ($TEACHER_SIZE)" \
            python scripts/download_qwen.py \
            --sizes "$TEACHER_SIZE" --mirror "$MIRROR"
    fi

    # 0b: gsm8k train (only if T6 or T7 will run)
    if [[ "$SKIP_GSM8K" -eq 1 ]]; then
        echo "[PHASE 0] skip gsm8k download (no training phase in active range)"
    else
        run_or_dry "Load gsm8k train prompts (max $MAX_TRAIN)" \
            python scripts/validate/load_gsm8k_train.py \
            --mirror "$MIRROR" --max_samples "$MAX_TRAIN"
    fi

    # 0c: verify what we downloaded
    echo
    echo "[CHECK Phase 0 outputs]"
    local ok=1
    if [[ "$SKIP_QWEN" -eq 0 ]]; then
        python scripts/download_qwen.py --sizes "$TEACHER_SIZE" --check_only \
            --min_weights_gb 1.0 || ok=0
    fi
    if [[ "$SKIP_GSM8K" -eq 0 ]]; then
        check_file "$GSM8K_TRAIN_JSON" "gsm8k train JSON" 1000 || ok=0
    fi
    [[ "$ok" -eq 1 ]] && echo "[PHASE 0] ✓ PASS" || { echo "[PHASE 0] ✗ FAIL"; exit 1; }
}

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — T6: AR-teacher canvas distill (main contribution)
# ═════════════════════════════════════════════════════════════════════════════
# T6 before T7 rationale:
#   1. T6 introduces NEW structural reasoning (canvas <SETUP>/<STEP>/<ANSWER>);
#      T7 only amplifies LLaDA's existing reasoning. Structure-first is
#      the cleaner pedagogical order (curriculum learning).
#   2. T7 cannot help on ceiling-5 prompts (LLaDA pass@8=0 there) → only
#      T6 can break those. So T6 is the necessary pipeline.
#   3. Cost parity now that teacher is local Qwen (not paid API), so the
#      'T7 is cheaper' rationale no longer applies.
#   4. Running T6 and T7 independently (no warm-start) gives cleaner
#      attribution: baseline / +T6 / +T7 / +T6+T7 four-way compare.
phase_1() {
    hdr "Phase 1 — T6 canvas distill (AR-teacher, main contribution)"

    # 1a: teacher trace generation (local Qwen3.5-4B)
    # Single-GPU path vs multi-GPU shard orchestrator
    if [[ "$T6_GPUS" -gt 1 ]]; then
        run_or_dry "T6 teacher trace generation ($T6_GPUS-GPU shard)" \
            bash scripts/run_t6_shards.sh \
            --gpus "$T6_GPUS" \
            --max_train "$MAX_TRAIN" \
            --teacher_ckpt "$TEACHER_CKPT" \
            --retries "$T6_RETRIES" \
            --scope_path "$GSM8K_TRAIN_JSON" --scope_group gsm8k
    else
        run_or_dry "T6 teacher trace generation (single-GPU)" \
            python scripts/validate/t6_teacher_trace.py \
            --scope_path "$GSM8K_TRAIN_JSON" --scope_group gsm8k \
            --n "$MAX_TRAIN" \
            --teacher local --local_model "$TEACHER_CKPT" \
            --retries_per_prompt "$T6_RETRIES" \
            --max_tokens 800 --temperature 0.0
    fi

    # Resolve latest T6 run dir
    T6_RUN_DIR=$(ls -dt "$ROOT"/runs/validation/t6_teacher_trace_* 2>/dev/null | head -1)
    T6_SFT_JSONL="$T6_RUN_DIR/t6_sft.jsonl"

    # 1b: T6 SFT from LLaDA baseline (NOT warm-starting from T7 anymore)
    run_or_dry "T6 SFT (from LLaDA baseline)" \
        python scripts/validate/t6t7_train.py \
        --jsonl_path "$T6_SFT_JSONL" \
        --run_name v16_t6 \
        --init_ckpt "GSAI-ML/LLaDA-8B-Instruct" \
        --max_steps "$T6_MAX_STEPS" \
        --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" \
        --lr 2e-5

    # 1c: verify
    echo
    echo "[CHECK Phase 1 outputs]"
    local ok=1
    check_file "$T6_SFT_JSONL" "T6 SFT JSONL" 100 || ok=0
    check_dir_nonempty "$T6_CKPT_DIR" "T6 ckpt dir" || ok=0
    [[ "$ok" -eq 1 ]] && echo "[PHASE 1] ✓ PASS" || { echo "[PHASE 1] ✗ FAIL"; exit 1; }
}

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — T7: self-distill on sampled-correct (independent add-on)
# ═════════════════════════════════════════════════════════════════════════════
# Independent SFT from baseline LLaDA (not warm-start from T6). This gives
# a clean 4-way attribution in Phase 3 eval: baseline / T6-only / T7-only /
# T6+T7 combined. T7 is treated as an optional add-on to the main T6 claim.
phase_2() {
    hdr "Phase 2 — T7 self-distill (independent add-on)"

    # 2a: require LLaDA serve for T7 sampling
    if [[ "$CHECK_ONLY" -eq 0 && "$DRY_RUN" -eq 0 ]]; then
        if ! curl -sf "$SERVER_URL/health" > /dev/null 2>&1; then
            echo "[PHASE 2] ERROR: LLaDA server not reachable at $SERVER_URL"
            echo "  Start: python scripts/serve.py --port 8000"
            exit 1
        fi
        echo "[PHASE 2] server OK at $SERVER_URL"
    fi

    # 2b: T7 data generation (sample + filter correct)
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

    # 2c: T7 SFT from LLaDA baseline (independent add-on, not warm-start)
    run_or_dry "T7 SFT (from LLaDA baseline, independent)" \
        python scripts/validate/t6t7_train.py \
        --jsonl_path "$T7_SFT_JSONL" \
        --run_name v16_t7 \
        --init_ckpt "GSAI-ML/LLaDA-8B-Instruct" \
        --max_steps "$T7_MAX_STEPS" \
        --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM" \
        --lr 2e-5

    # 2d: verify
    echo
    echo "[CHECK Phase 2 outputs]"
    local ok=1
    check_file "$T7_SFT_JSONL" "T7 SFT JSONL" 100 || ok=0
    check_dir_nonempty "$T7_CKPT_DIR" "T7 ckpt dir" || ok=0
    [[ "$ok" -eq 1 ]] && echo "[PHASE 2] ✓ PASS" || { echo "[PHASE 2] ✗ FAIL"; exit 1; }
}

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — Eval on held-out test set
# ═════════════════════════════════════════════════════════════════════════════
phase_3() {
    hdr "Phase 3 — Eval on held-out scope_fail + scope_ok"

    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    EVAL_DIR="$ROOT/runs/validation/v16_eval_${ts}"

    # Build list of ckpts: baseline + any existing stage dirs
    local args=("--out_dir" "$EVAL_DIR"
                "--gen_length" "$T7_GEN_LENGTH"
                "--block_length" "32"
                "--temperature" "0.0"
                "--ckpts"
                "baseline=GSAI-ML/LLaDA-8B-Instruct")
    if [[ -d "$T7_CKPT_DIR/hf" ]]; then
        args+=("t7_stage1=$T7_CKPT_DIR/hf")
    fi
    if [[ -d "$T6_CKPT_DIR/hf" ]]; then
        args+=("t6_stage2=$T6_CKPT_DIR/hf")
    fi

    run_or_dry "v1.6 eval on scope_fail + scope_ok" \
        python scripts/validate/v16_eval.py "${args[@]}"

    echo
    echo "[CHECK Phase 3 outputs]"
    local ok=1
    check_file "$EVAL_DIR/summary.json" "eval summary" 100 || ok=0
    check_file "$EVAL_DIR/comparison.md" "eval comparison md" 50 || ok=0
    [[ "$ok" -eq 1 ]] && echo "[PHASE 3] ✓ PASS" || { echo "[PHASE 3] ✗ FAIL"; exit 1; }
}

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — Archive
# ═════════════════════════════════════════════════════════════════════════════
phase_4() {
    hdr "Phase 4 — Archive results"

    # Point at the latest eval dir if Phase 3 ran in the same invocation
    local latest_eval
    latest_eval=$(ls -dt "$ROOT"/runs/validation/v16_eval_* 2>/dev/null | head -1)
    if [[ -z "$latest_eval" ]]; then
        echo "[PHASE 4] no v16_eval_* dir found; run Phase 3 first"
        return 1
    fi

    local archive_doc="$ROOT/docs/archive/finding_v1.6_selfdistill_canvas.zh.md"
    echo "[PHASE 4] stub archive doc @ $archive_doc"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY-RUN] would write archive doc from $latest_eval/summary.json"
        return 0
    fi

    mkdir -p "$(dirname "$archive_doc")"
    cat > "$archive_doc" <<EOF
# Finding v1.6 —— Self-Distill + AR-Teacher Canvas SFT

**日期**：$(date +%Y-%m-%d)
**前置 plan**：\`docs/plans/2026-04-19_v1.6_plan.zh.md\`
**Eval run**：\`${latest_eval#$ROOT/}\`

## 数字

请看同目录 \`comparison.md\`:

\`\`\`
$(cat "$latest_eval/comparison.md" 2>/dev/null || echo '(comparison.md missing — Phase 3 did not complete)')
\`\`\`

## 结论

TODO: fill in after reviewing numbers

## 相关

- Plan: \`docs/plans/2026-04-19_v1.6_plan.zh.md\`
- Related work: \`docs/plans/2026-04-19_related_work_review.zh.md\`
- Ablation index: \`docs/archive/ablation_index.zh.md\`
EOF

    echo "[PHASE 4] stub archive written: $archive_doc"
    echo "[PHASE 4] fill 结论 section manually after review"
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

# Phase 0 (data dependencies) always runs — its steps are idempotent
# (snapshot_download resumes, load_gsm8k_train skips if JSON exists) and
# subsequent phases need the data. SKIP_QWEN / SKIP_GSM8K still respected
# inside phase_0 for fine control. Use --from_phase 1+ if you've already
# verified downloads elsewhere and want to start from T6.
phase_0
[[ "$FROM_PHASE" -le 1 && "$TO_PHASE" -ge 1 ]] && phase_1
if [[ "$INCLUDE_T7" -eq 1 ]]; then
    [[ "$FROM_PHASE" -le 2 && "$TO_PHASE" -ge 2 ]] && phase_2
fi
[[ "$FROM_PHASE" -le 3 && "$TO_PHASE" -ge 3 ]] && phase_3
[[ "$FROM_PHASE" -le 4 && "$TO_PHASE" -ge 4 ]] && phase_4

hdr "v1.6 pipeline done"
