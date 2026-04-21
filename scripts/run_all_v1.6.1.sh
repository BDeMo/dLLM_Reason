#!/usr/bin/env bash
# run_all_v1.6.1.sh — End-to-end v1.6.1 pipeline, one command.
#
# Pipeline:
#   0. Download Qwen teacher + gsm8k train  (idempotent: resume if present)
#   1. Start a single-GPU LLaDA serve for scope regen + T7/eval HTTP
#   2. Regenerate scope_fail_prompts.json + scope_ok_prompts.json from
#      gsm8k test under CANONICAL config (T=0, bl=32, g=128)
#   3. T6 teacher trace on gsm8k TRAIN (multi-GPU shard, multi-output
#      collection: diverse valid traces per prompt under T>0)
#   4. T6 SFT on the (auto-cleaned) t6_sft.jsonl
#   5. Eval baseline vs T6-trained on the FRESH scope (canonical config)
#   6. Archive stub in docs/archive/
#
# Designed for a single overnight / multi-hour run on 8×A100. Every
# phase is idempotent and resume-safe; re-running after an interruption
# picks up where it stopped.
#
# Usage:
#   bash scripts/run_all_v1.6.1.sh
#   bash scripts/run_all_v1.6.1.sh --mirror hf-mirror
#   bash scripts/run_all_v1.6.1.sh --teacher_size 3-8B --max_train 2000
#   bash scripts/run_all_v1.6.1.sh --from_phase 3   # skip downloads / scope
#   bash scripts/run_all_v1.6.1.sh --dry_run
#
# Local-first behavior (no --offline flag needed):
#   Phase 0 invokes the project's registered downloaders:
#     - scripts/download_models.py --models llada-instruct
#         → checkpoints/llada-instruct/
#     - scripts/download_datasets.py --datasets gsm8k
#         → datasets/gsm8k/{train,test}/  (save_to_disk format)
#     - scripts/download_qwen.py --sizes <T>
#         → checkpoints/Qwen__Qwen<T>/
#   All three are idempotent: skip if files present + verified.
#
#   Subsequent phases (regen scope, T6 trace, SFT, eval) all use the
#   project's local-first resolvers (resolve_model_path, resolve_dataset)
#   so they hit the registered local paths automatically. HF is only
#   contacted when a local path is missing AND Phase 0 was skipped.
#
#   Mirror is consulted only if a download is actually needed (Phase 0
#   first run, or when files were manually deleted).
#
# Multi-GPU summary (all phases now parallelizable):
#   --scope_gpus N   Phase 2 scope regen (default 8). Each GPU runs its
#                    own serve + client; ~8× speedup.
#   --t6_gpus N      Phase 3 teacher trace (default 8); same shard pattern.
#   --sft_gpus N     Phase 4 T6 SFT via torchrun + DDP (default 8).
#                    Effective batch = batch_size × grad_accum × N.
#   --eval_gpus N    Phase 5 parallel ckpt eval (default 1 = serial).
#                    >1 only helps if multiple ckpts to eval; each ckpt
#                    pinned to one GPU.
#
# Target wall-time on 8×A100 with all flags ≥ 4:
#   Phase 0 ~10 min, Phase 2 ~15 min, Phase 3 ~30 min, Phase 4 ~15 min
#   (was ~1.5h single-GPU), Phase 5 ~15 min, Phase 6 seconds. Total ~1.5h.
#
# Defaults target the "big run" profile the user described:
#   - Qwen3-8B teacher (compatible with stable transformers 4.x)
#   - 2000 gsm8k train prompts
#   - Multi-output teacher (retries=5, temperature=0.7) → diverse traces
#   - 8-GPU shard for teacher trace gen
#   - Single-GPU SFT (LLaDA-8B fits A100-80GB at bs=4)

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MIRROR="default"
TEACHER_SIZE="3-8B"                # Qwen3-8B (no transformers 5 required)
TEACHER_CKPT=""                    # auto from TEACHER_SIZE
MAX_TRAIN=2000
MAX_SCOPE_PROMPTS=""               # "" = full gsm8k test (1319)
T6_GPUS=8
T6_BATCH_SIZE=8
SCOPE_GPUS=8                        # multi-GPU scope regen (Phase 2). 1 = single
SFT_GPUS=8                          # multi-GPU T6 SFT via torchrun DDP. 1 = single
EVAL_GPUS=1                         # parallel ckpt eval (Phase 5). 1 = serial
T6_RETRIES=5                       # ↑ from 3: more diverse attempts/prompt
T6_TEMPERATURE=0.7                 # > 0 so retries give different outputs
T6_MAX_STEPS=2000
T6_BATCH_SIZE_SFT=1                 # per-rank batch
T6_GRAD_ACCUM=16                    # effective batch = 1 × 16 × SFT_GPUS
T6_LR=2e-5
T6_PARALLEL="fsdp"                  # fsdp (default for 8B) | ddp
T6_USE_LORA=0                       # 1 = LoRA adapters; base weights frozen
T6_LORA_R=16
T6_LORA_ALPHA=32
EVAL_GEN_LENGTH=128                # MUST match scope canonical
EVAL_BLOCK_LENGTH=32
EVAL_TEMPERATURE=0
SERVER_PORT=8000
FROM_PHASE=0
TO_PHASE=6
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mirror)             MIRROR="$2"; shift 2 ;;
        --teacher_size)       TEACHER_SIZE="$2"; shift 2 ;;
        --teacher_ckpt)       TEACHER_CKPT="$2"; shift 2 ;;
        --max_train)          MAX_TRAIN="$2"; shift 2 ;;
        --max_scope_prompts)  MAX_SCOPE_PROMPTS="$2"; shift 2 ;;
        --t6_gpus)            T6_GPUS="$2"; shift 2 ;;
        --t6_batch_size)      T6_BATCH_SIZE="$2"; shift 2 ;;
        --scope_gpus)         SCOPE_GPUS="$2"; shift 2 ;;
        --sft_gpus)           SFT_GPUS="$2"; shift 2 ;;
        --eval_gpus)          EVAL_GPUS="$2"; shift 2 ;;
        --t6_retries)         T6_RETRIES="$2"; shift 2 ;;
        --t6_temperature)     T6_TEMPERATURE="$2"; shift 2 ;;
        --t6_max_steps)       T6_MAX_STEPS="$2"; shift 2 ;;
        --t6_lr)              T6_LR="$2"; shift 2 ;;
        --t6_parallel)        T6_PARALLEL="$2"; shift 2 ;;
        --t6_use_lora)        T6_USE_LORA=1; shift ;;
        --t6_lora_r)          T6_LORA_R="$2"; shift 2 ;;
        --t6_lora_alpha)      T6_LORA_ALPHA="$2"; shift 2 ;;
        --server_port)        SERVER_PORT="$2"; shift 2 ;;
        --from_phase)         FROM_PHASE="$2"; shift 2 ;;
        --to_phase)           TO_PHASE="$2"; shift 2 ;;
        --dry_run)            DRY_RUN=1; shift ;;
        -h|--help)            grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)                    echo "[ALL] unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

[[ -z "$TEACHER_CKPT" ]] && \
    TEACHER_CKPT="$ROOT/checkpoints/Qwen__Qwen${TEACHER_SIZE}"

GSM8K_TRAIN="$ROOT/runs/validation/gsm8k_train_prompts.json"
SCOPE_FAIL="$ROOT/runs/validation/scope_fail_prompts.json"
SCOPE_OK="$ROOT/runs/validation/scope_ok_prompts.json"
T6_CKPT_DIR="$ROOT/runs/training/v161_t6"
SERVER_URL="http://127.0.0.1:$SERVER_PORT"

hdr() {
    echo
    echo "══════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "══════════════════════════════════════════════════════════════"
}

run_or_dry() {
    local desc="$1"; shift
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY] $desc"
        echo "      $*"
        return 0
    fi
    echo "[RUN] $desc"
    echo "      $*"
    eval "$@"
}

check_file() {
    local p="$1" desc="$2" min="${3:-1}"
    if [[ ! -f "$p" ]]; then
        echo "  ✗ missing: $desc ($p)"; return 1
    fi
    local sz
    sz=$(stat -c%s "$p" 2>/dev/null || stat -f%z "$p" 2>/dev/null || echo 0)
    if [[ "$sz" -lt "$min" ]]; then
        echo "  ✗ too small ($sz < $min): $desc"; return 1
    fi
    echo "  ✓ $desc ($sz bytes)"; return 0
}

# ── Banner ────────────────────────────────────────────────────────────────────
hdr "v1.6.1 Pipeline — Run All"
cat <<EOF
  ROOT              = $ROOT
  MIRROR            = $MIRROR  (used only if local cache miss)
  TEACHER_SIZE      = $TEACHER_SIZE
  TEACHER_CKPT      = $TEACHER_CKPT
  MAX_TRAIN (T6 src)= $MAX_TRAIN (gsm8k train)
  MAX_SCOPE         = ${MAX_SCOPE_PROMPTS:-"all 1319"} (gsm8k test, for scope regen)
  SCOPE_GPUS        = $SCOPE_GPUS  (Phase 2; 1 = single via Phase-1 serve)
  SFT_GPUS          = $SFT_GPUS  (Phase 4 $T6_PARALLEL via torchrun; 1 = single)
  T6_USE_LORA       = $T6_USE_LORA  (r=$T6_LORA_R, α=$T6_LORA_ALPHA)
  EVAL_GPUS         = $EVAL_GPUS  (Phase 5 parallel ckpts; 1 = serial)
  T6_GPUS           = $T6_GPUS
  T6_BATCH_SIZE     = $T6_BATCH_SIZE  (HF pipeline batching per shard)
  T6_RETRIES        = $T6_RETRIES  (v1.6.1: multi-output via T>0 + retries)
  T6_TEMPERATURE    = $T6_TEMPERATURE  (v1.6.1: >0 for diversity)
  T6_MAX_STEPS      = $T6_MAX_STEPS
  EVAL config       = T=$EVAL_TEMPERATURE  bl=$EVAL_BLOCK_LENGTH  g=$EVAL_GEN_LENGTH  (canonical)
  SERVER_URL        = $SERVER_URL
  PHASES            = $FROM_PHASE ... $TO_PHASE
  DRY_RUN           = $DRY_RUN
EOF

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 0 — Downloads
# ═════════════════════════════════════════════════════════════════════════════
phase_0() {
    hdr "Phase 0 — Materialize registered local data (idempotent)"

    # Build mirror URL kwarg only if MIRROR is a non-default value.
    # "default" / "" → no flag; let downloader use HF default
    # "hf-mirror" / "modelscope" → canonical URL
    # http(s)://... → pass through
    local mirror_flag=()
    case "$MIRROR" in
        ""|default)         ;;
        hf-mirror)          mirror_flag=(--mirror "https://hf-mirror.com") ;;
        modelscope)         mirror_flag=(--mirror "https://www.modelscope.cn") ;;
        http://*|https://*) mirror_flag=(--mirror "$MIRROR") ;;
        *)                  mirror_flag=(--mirror "$MIRROR") ;;
    esac

    # 0a: LLaDA via the project-registered downloader. Saves to
    #     checkpoints/llada-instruct/ where resolve_model_path will find it.
    #     Idempotent (skip if files present + verified).
    run_or_dry "Download LLaDA-Instruct (registered → checkpoints/llada-instruct/)" \
        python scripts/download_models.py \
        --models llada-instruct "${mirror_flag[@]}"

    # 0b: GSM8K via project-registered downloader. Saves both splits to
    #     datasets/gsm8k/{train,test}/ via save_to_disk so resolve_dataset
    #     can later load_from_disk without ANY HF call.
    run_or_dry "Download GSM8K (registered → datasets/gsm8k/{train,test}/)" \
        python scripts/download_datasets.py \
        --datasets gsm8k "${mirror_flag[@]}"

    # 0c: Qwen teacher via our v1.6 downloader (saves to
    #     checkpoints/Qwen__<name>/). Not in MODEL_REGISTRY (yet),
    #     but absolute path works fine for t6_teacher_trace.
    run_or_dry "Download Qwen teacher ($TEACHER_SIZE → checkpoints/Qwen__...)" \
        python scripts/download_qwen.py \
        --sizes "$TEACHER_SIZE" --mirror "$MIRROR"

    # 0d: Convert downloaded gsm8k train into our scope JSON shape (still
    #     needed because t6_teacher_trace consumes that JSON, not the HF
    #     Dataset object). After this, runs/validation/gsm8k_train_prompts.json
    #     references local data only.
    run_or_dry "Materialize gsm8k_train_prompts.json from local datasets/" \
        python scripts/validate/load_gsm8k_train.py \
        --max_samples "$MAX_TRAIN"

    echo
    echo "[CHECK Phase 0]"
    local ok=1
    python scripts/download_qwen.py --sizes "$TEACHER_SIZE" --check_only \
        --min_weights_gb 1.0 || ok=0
    check_file "$GSM8K_TRAIN" "gsm8k_train_prompts.json" 1000 || ok=0
    if [[ -d "$ROOT/datasets/gsm8k/train" ]] && [[ -d "$ROOT/datasets/gsm8k/test" ]]; then
        echo "  ✓ datasets/gsm8k/train/ + test/ registered"
    else
        echo "  ✗ datasets/gsm8k/{train,test}/ missing — resolve_dataset will fall back to HF"
    fi
    if [[ -d "$ROOT/checkpoints/llada-instruct" ]]; then
        echo "  ✓ checkpoints/llada-instruct/ registered"
    else
        echo "  ✗ checkpoints/llada-instruct/ missing — LLaDAWrapper will fall back to HF"
    fi
    [[ "$ok" -eq 1 ]] || { echo "[PHASE 0] ✗ FAIL"; exit 1; }
    echo "[PHASE 0] ✓ PASS"
}

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — Start LLaDA serve (for scope regen + eval)
# ═════════════════════════════════════════════════════════════════════════════
SERVE_PID=""
cleanup_serve() {
    if [[ -n "$SERVE_PID" ]]; then
        echo "[ALL] cleanup: kill serve pid $SERVE_PID"
        kill "$SERVE_PID" 2>/dev/null || true
    fi
}
trap cleanup_serve EXIT INT TERM

phase_1() {
    hdr "Phase 1 — Start LLaDA serve on port $SERVER_PORT"

    # If scope regen will use its own multi-GPU shards (each spinning its
    # own serve), this Phase-1 serve is redundant. Skip.
    if [[ "$SCOPE_GPUS" -gt 1 ]]; then
        echo "[PHASE 1] SCOPE_GPUS=$SCOPE_GPUS > 1 \u2192 Phase-2 spins its own serves, skipping Phase-1 serve"
        return 0
    fi

    # Reuse if already up
    if curl -sf "$SERVER_URL/health" > /dev/null 2>&1; then
        echo "[PHASE 1] serve already running at $SERVER_URL, reusing"
        return 0
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY] would launch: CUDA_VISIBLE_DEVICES=0 python scripts/serve.py --port $SERVER_PORT"
        return 0
    fi

    mkdir -p tmp
    CUDA_VISIBLE_DEVICES=0 nohup python scripts/serve.py \
        --model_id GSAI-ML/LLaDA-8B-Instruct \
        --port "$SERVER_PORT" --host 127.0.0.1 \
        > tmp/v161_serve.log 2>&1 &
    SERVE_PID=$!
    echo "[PHASE 1] serve pid=$SERVE_PID, log=tmp/v161_serve.log"

    local waited=0
    until curl -sf "$SERVER_URL/health" > /dev/null 2>&1; do
        sleep 5; waited=$((waited + 5))
        if [[ "$waited" -ge 600 ]]; then
            echo "[PHASE 1] ✗ serve not ready after 10 min"; exit 1
        fi
    done
    echo "[PHASE 1] ✓ serve ready after ${waited}s"
}

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — Regenerate scope_fail + scope_ok from gsm8k test
# ═════════════════════════════════════════════════════════════════════════════
phase_2() {
    hdr "Phase 2 — Regenerate scope ($SCOPE_GPUS-GPU, gsm8k test → scope_fail/ok)"

    # gsm8k test now comes from datasets/gsm8k/test/ (Phase 0 materialized
    # via download_datasets.py). resolve_dataset will load from disk; no
    # HF call needed even on flaky network.

    if [[ "$SCOPE_GPUS" -gt 1 ]]; then
        local regen_args=(
            --gpus "$SCOPE_GPUS"
            --base_port "$((SERVER_PORT + 100))"
        )
        [[ -n "$MAX_SCOPE_PROMPTS" ]] && \
            regen_args+=(--max_prompts "$MAX_SCOPE_PROMPTS")
        run_or_dry "Regenerate scope ($SCOPE_GPUS-GPU shard, local-first)" \
            bash scripts/run_regen_scope_shards.sh "${regen_args[@]}"
    else
        local regen_args=(
            --server_url "$SERVER_URL"
            --gen_length 128
            --block_length 32
            --temperature 0
        )
        [[ -n "$MAX_SCOPE_PROMPTS" ]] && \
            regen_args+=(--max_prompts "$MAX_SCOPE_PROMPTS")
        run_or_dry "Regenerate scope (single-GPU, local-first)" \
            python scripts/validate/regen_scope.py "${regen_args[@]}"
    fi

    echo
    echo "[CHECK Phase 2]"
    local ok=1
    check_file "$SCOPE_FAIL" "scope_fail" 500 || ok=0
    check_file "$SCOPE_OK"   "scope_ok"   500 || ok=0
    [[ "$ok" -eq 1 ]] || { echo "[PHASE 2] ✗ FAIL"; exit 1; }
    echo "[PHASE 2] ✓ PASS"
}

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — T6 teacher trace (multi-GPU, multi-output)
# ═════════════════════════════════════════════════════════════════════════════
phase_3() {
    hdr "Phase 3 — T6 teacher trace ($T6_GPUS GPU, T=$T6_TEMPERATURE, retries=$T6_RETRIES)"

    local rt6_args=(
        --gpus "$T6_GPUS"
        --max_train "$MAX_TRAIN"
        --teacher_ckpt "$TEACHER_CKPT"
        --retries "$T6_RETRIES"
        --batch_size "$T6_BATCH_SIZE"
        --scope_path "$GSM8K_TRAIN"
        --scope_group gsm8k
    )
    run_or_dry "T6 teacher trace ($T6_GPUS-GPU shard)" \
        bash scripts/run_t6_shards.sh "${rt6_args[@]}"

    # Resolve the most recent t6_teacher_trace dir for the SFT phase
    T6_RUN_DIR=$(ls -dt "$ROOT"/runs/validation/t6_teacher_trace_* 2>/dev/null | head -1)
    T6_SFT_JSONL="$T6_RUN_DIR/t6_sft.jsonl"

    echo
    echo "[CHECK Phase 3]"
    local ok=1
    check_file "$T6_SFT_JSONL" "t6_sft.jsonl" 100 || ok=0
    [[ "$ok" -eq 1 ]] || { echo "[PHASE 3] ✗ FAIL"; exit 1; }
    echo "[PHASE 3] ✓ PASS  (run dir: $T6_RUN_DIR)"
}

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — T6 SFT (single-GPU, skip if ckpt exists)
# ═════════════════════════════════════════════════════════════════════════════
phase_4() {
    hdr "Phase 4 — T6 SFT ($T6_MAX_STEPS steps, bs=$T6_BATCH_SIZE_SFT × accum $T6_GRAD_ACCUM)"

    # Resolve T6 JSONL (may have been set by Phase 3 or by --from_phase 4)
    if [[ -z "${T6_SFT_JSONL:-}" ]]; then
        T6_RUN_DIR=$(ls -dt "$ROOT"/runs/validation/t6_teacher_trace_* 2>/dev/null | head -1)
        T6_SFT_JSONL="$T6_RUN_DIR/t6_sft.jsonl"
    fi

    if [[ -f "$T6_CKPT_DIR/hf/config.json" ]]; then
        echo "[PHASE 4] T6 ckpt exists at $T6_CKPT_DIR/hf; skipping SFT"
    else
        local launcher="python"
        if [[ "$SFT_GPUS" -gt 1 ]]; then
            launcher="torchrun --standalone --nproc_per_node=$SFT_GPUS"
            echo "[PHASE 4] parallel=$T6_PARALLEL on $SFT_GPUS GPUs via torchrun"
        fi
        local lora_flags=()
        if [[ "$T6_USE_LORA" -eq 1 ]]; then
            lora_flags=(--use_lora --lora_r "$T6_LORA_R" --lora_alpha "$T6_LORA_ALPHA")
            echo "[PHASE 4] LoRA ON  r=$T6_LORA_R α=$T6_LORA_ALPHA"
        fi
        run_or_dry "T6 SFT (gpus=$SFT_GPUS parallel=$T6_PARALLEL lora=$T6_USE_LORA)" \
            $launcher scripts/validate/t6t7_train.py \
            --jsonl_path "$T6_SFT_JSONL" \
            --run_name v161_t6 \
            --init_ckpt "GSAI-ML/LLaDA-8B-Instruct" \
            --max_steps "$T6_MAX_STEPS" \
            --batch_size "$T6_BATCH_SIZE_SFT" --grad_accum_steps "$T6_GRAD_ACCUM" \
            --lr "$T6_LR" \
            --max_seq_len 768 \
            --parallel "$T6_PARALLEL" \
            "${lora_flags[@]}"
    fi

    echo
    echo "[CHECK Phase 4]"
    if [[ -d "$T6_CKPT_DIR/hf" ]]; then
        echo "  ✓ T6 ckpt ($T6_CKPT_DIR/hf)"
        echo "[PHASE 4] ✓ PASS"
    else
        echo "  ✗ missing T6 ckpt dir"; echo "[PHASE 4] ✗ FAIL"; exit 1
    fi
}

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 5 — Eval baseline vs T6 (canonical config)
# ═════════════════════════════════════════════════════════════════════════════
phase_5() {
    hdr "Phase 5 — Eval baseline vs T6 (canonical config g=$EVAL_GEN_LENGTH)"

    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    local EVAL_DIR="$ROOT/runs/validation/v161_eval_${ts}"

    # Build ckpt list
    local CKPTS=("baseline=GSAI-ML/LLaDA-8B-Instruct")
    [[ -d "$T6_CKPT_DIR/hf" ]] && CKPTS+=("t6=$T6_CKPT_DIR/hf")

    if [[ "$EVAL_GPUS" -gt 1 ]] && [[ "${#CKPTS[@]}" -gt 1 ]]; then
        # Parallel: one ckpt per GPU, each writes its own out_dir, merge after
        echo "[PHASE 5] EVAL_GPUS=$EVAL_GPUS \u2192 parallel ckpt eval"
        mkdir -p "$EVAL_DIR/per_prompt"
        local PIDS=()
        local g=0
        for ck in "${CKPTS[@]}"; do
            local label="${ck%%=*}"
            local SUB="$EVAL_DIR/${label}_only"
            run_or_dry "eval $label on GPU $g" \
                "CUDA_VISIBLE_DEVICES=$g python scripts/validate/v16_eval.py \
                 --ckpts $ck --out_dir $SUB \
                 --gen_length $EVAL_GEN_LENGTH \
                 --block_length $EVAL_BLOCK_LENGTH \
                 --temperature $EVAL_TEMPERATURE &"
            PIDS+=($!)
            g=$((g + 1))
            [[ "$g" -ge "$EVAL_GPUS" ]] && g=0   # cycle
        done
        if [[ "$DRY_RUN" -eq 0 ]]; then
            echo "[PHASE 5] waiting on parallel evals: ${PIDS[*]}"
            for p in "${PIDS[@]}"; do wait "$p"; done
            # Merge per-ckpt summaries into EVAL_DIR
            python - <<PY
import json, glob, pathlib
out = pathlib.Path("$EVAL_DIR")
all_stats = []
for s in sorted(glob.glob(str(out / "*_only" / "summary.json"))):
    d = json.load(open(s))
    all_stats.extend(d.get("ckpts", []))
(out / "summary.json").write_text(
    json.dumps({"ckpts": all_stats, "config": {
        "gen_length": $EVAL_GEN_LENGTH, "block_length": $EVAL_BLOCK_LENGTH,
        "temperature": $EVAL_TEMPERATURE,
    }}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
# Re-render comparison.md
lines = ["# v1.6.1 Eval Comparison (parallel)\n",
         "| Label | fail pass@1 | ok pass@1 | FAIL18 rescued | ceiling broken |",
         "|---|---|---|---|---|"]
for s in all_stats:
    lines.append(
        f"| {s['label']} | {s['fail_pass@1']:.2%} ({s['fail_correct']}/{s['n_fail']}) "
        f"| {s['ok_pass@1']:.2%} ({s['ok_correct']}/{s['n_ok']}) "
        f"| {s['fail18_rescued_count']}/18 \`{s['fail18_rescued']}\` "
        f"| {s['ceiling_broken_count']}/5 \`{s['ceiling_broken']}\` |"
    )
(out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
print(f"[merge] {len(all_stats)} ckpts merged \u2192 {out}/comparison.md")
PY
        fi
    else
        # Serial path (single GPU or only 1 ckpt)
        local eval_args=(
            --out_dir "$EVAL_DIR"
            --gen_length "$EVAL_GEN_LENGTH"
            --block_length "$EVAL_BLOCK_LENGTH"
            --temperature "$EVAL_TEMPERATURE"
            --ckpts "${CKPTS[@]}"
        )
        run_or_dry "v1.6.1 eval (serial)" \
            python scripts/validate/v16_eval.py "${eval_args[@]}"
    fi

    echo
    echo "[CHECK Phase 5]"
    check_file "$EVAL_DIR/comparison.md" "comparison.md" 50 && \
        echo "[PHASE 5] ✓ PASS (see $EVAL_DIR/comparison.md)" || \
        { echo "[PHASE 5] ✗ FAIL"; exit 1; }
}

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 6 — Archive stub
# ═════════════════════════════════════════════════════════════════════════════
phase_6() {
    hdr "Phase 6 — Archive stub to docs/archive/"

    local latest_eval
    latest_eval=$(ls -dt "$ROOT"/runs/validation/v161_eval_* 2>/dev/null | head -1)
    if [[ -z "$latest_eval" ]]; then
        echo "[PHASE 6] skip: no v161_eval_* dir"; return
    fi

    local archive_doc="$ROOT/docs/archive/finding_v1.6.1_canvas_distill.zh.md"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY] would write archive stub from $latest_eval"
        return
    fi
    mkdir -p "$(dirname "$archive_doc")"
    cat > "$archive_doc" <<EOF
# Finding v1.6.1 — Canvas Distill (multi-output teacher)

**日期**：$(date +%Y-%m-%d)
**前置 plan**：\`docs/plans/2026-04-19_v1.6_plan.zh.md\`
**Eval run**：\`${latest_eval#$ROOT/}\`

## 数字

\`\`\`
$(cat "$latest_eval/comparison.md" 2>/dev/null || echo '(comparison.md missing)')
\`\`\`

## 配置

- Teacher: \`$TEACHER_CKPT\`
- MAX_TRAIN: $MAX_TRAIN
- T6 retries=$T6_RETRIES  temperature=$T6_TEMPERATURE  (multi-output)
- SFT steps=$T6_MAX_STEPS  lr=$T6_LR
- Eval: g=$EVAL_GEN_LENGTH  bl=$EVAL_BLOCK_LENGTH  T=$EVAL_TEMPERATURE

## 结论

TODO: fill in after reviewing numbers
EOF
    echo "[PHASE 6] stub → $archive_doc"
}

# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════
[[ "$FROM_PHASE" -le 0 && "$TO_PHASE" -ge 0 ]] && phase_0
[[ "$FROM_PHASE" -le 1 && "$TO_PHASE" -ge 1 ]] && phase_1
[[ "$FROM_PHASE" -le 2 && "$TO_PHASE" -ge 2 ]] && phase_2
[[ "$FROM_PHASE" -le 3 && "$TO_PHASE" -ge 3 ]] && phase_3
[[ "$FROM_PHASE" -le 4 && "$TO_PHASE" -ge 4 ]] && phase_4
[[ "$FROM_PHASE" -le 5 && "$TO_PHASE" -ge 5 ]] && phase_5
[[ "$FROM_PHASE" -le 6 && "$TO_PHASE" -ge 6 ]] && phase_6

hdr "v1.6.1 pipeline done"
