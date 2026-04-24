#!/usr/bin/env bash
# t6_lora_ablate.sh — 2D ablation over LoRA rank × training duration.
#
# Design: LoRA adapters of different rank are DIFFERENT models (different
# parameter tensors), so we CAN'T share a single training across ranks.
# But within one rank, training is deterministic, so we share across
# epoch points via --hf_export_at_steps (same trick as t6_ablate.sh).
#
# Grid: ranks = {1, 2, 4, 8, 16} × epochs = {0.5, 1, 2, 4}.
#   → 5 trainings × multi-export = 5 × 4 = 20 ckpts
# Eval each ckpt vs baseline-constant (0/N_fail and N_ok/N_ok, no re-test).
#
# Cost:
#   naive (5×4 independent): 20 × train + 20 × eval
#   here:                    5  × train (to max epoch) + 20 × eval
#   → 75% training-time saving.
#
# Epoch → step: 1 epoch ≈ 1350/8 = 169 steps.
#
# Usage:
#   bash scripts/t6_lora_ablate.sh                       # default grid
#   bash scripts/t6_lora_ablate.sh --ranks 1,4,16
#   bash scripts/t6_lora_ablate.sh --epochs 0.5,1,2,4,8
#   bash scripts/t6_lora_ablate.sh --ranks 8 --epochs 2  # single cell
#   bash scripts/t6_lora_ablate.sh --dry_run

set -euo pipefail

DRY_RUN=0
TRAIN_N=1350
SAMPLES_PER_STEP=8
STEPS_PER_EPOCH=$(( TRAIN_N / SAMPLES_PER_STEP ))   # = 169
SFT_GPUS=8
T6_LR=2e-5
T6_BATCH_SIZE_SFT=1
T6_GRAD_ACCUM=16
EVAL_GEN_LENGTH=128
EVAL_BLOCK_LENGTH=32
EVAL_TEMPERATURE=0
EVAL_GPUS=8                     # parallel eval across this many GPUs
GPU_CSV=""                      # explicit CSV list (e.g. 0,2,4,6)
AUTO_GPUS=0                     # pick least-busy GPUs via nvidia-smi
BASELINE_CKPT="GSAI-ML/LLaDA-8B-Instruct"

RANKS_CSV="1,2,4,8,16"
EPOCHS_CSV="0.5,1,2,4"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ranks)     RANKS_CSV="$2"; shift 2 ;;
        --epochs)    EPOCHS_CSV="$2"; shift 2 ;;
        --eval_gpus) EVAL_GPUS="$2"; shift 2 ;;
        --gpus)      GPU_CSV="$2"; shift 2 ;;
        --auto_gpus) AUTO_GPUS=1; shift ;;
        --dry_run)   DRY_RUN=1; shift ;;
        -h|--help)   grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[LORA-ABL] unknown arg: $1" >&2; exit 1 ;;
    esac
done

IFS=',' read -r -a RANKS  <<< "$RANKS_CSV"
IFS=',' read -r -a EPOCHS <<< "$EPOCHS_CSV"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

ABL_DIR="$ROOT/runs/validation/t6_lora_ablate"
mkdir -p "$ABL_DIR"
LOG_DIR="$ROOT/runs/ablate_logs"
mkdir -p "$LOG_DIR"

TS_ALL=$(date +%Y%m%d_%H%M%S)

# epochs → target steps (integer, sorted unique)
TARGET_STEPS_CSV=$(python - <<PY
spe = int("$STEPS_PER_EPOCH")
epochs = [$(IFS=,; echo "${EPOCHS[*]}")]
steps = sorted(set(max(1, round(float(e) * spe)) for e in epochs))
print(",".join(str(s) for s in steps))
PY
)
MAX_STEP=$(python - <<PY
print(max(int(s) for s in "$TARGET_STEPS_CSV".split(",")))
PY
)

MANIFEST="$ABL_DIR/manifest_${TS_ALL}.txt"
{
    echo "ablation started: $TS_ALL"
    echo "RANKS             = ${RANKS[*]}"
    echo "EPOCHS            = ${EPOCHS[*]}"
    echo "TARGET_STEPS      = $TARGET_STEPS_CSV"
    echo "MAX_STEP          = $MAX_STEP"
    echo "STEPS_PER_EPOCH   = $STEPS_PER_EPOCH"
    echo "DRY_RUN           = $DRY_RUN"
} > "$MANIFEST"

echo "[LORA-ABL] ==========================================================="
echo "[LORA-ABL]   LoRA rank × epoch ablation"
echo "[LORA-ABL]   ranks           = ${RANKS[*]}"
echo "[LORA-ABL]   epochs          = ${EPOCHS[*]}"
echo "[LORA-ABL]   target steps    = $TARGET_STEPS_CSV  (train once to $MAX_STEP)"
echo "[LORA-ABL]   total cells     = $(( ${#RANKS[@]} * ${#EPOCHS[@]} ))"
echo "[LORA-ABL]   output dir      = $ABL_DIR"
echo "[LORA-ABL] ==========================================================="

# Resolve T6 SFT data
if [[ -z "${T6_SFT_JSONL:-}" ]]; then
    T6_RUN_DIR=$(ls -dt "$ROOT"/runs/validation/t6_teacher_trace_* 2>/dev/null | head -1)
    T6_SFT_JSONL="$T6_RUN_DIR/t6_sft.jsonl"
fi
[[ ! -f "$T6_SFT_JSONL" ]] && { echo "[LORA-ABL] ERROR: $T6_SFT_JSONL missing" >&2; exit 1; }
echo "[LORA-ABL] data: $T6_SFT_JSONL"

# Resolve GPU constraint FIRST — applies to all 3 phases:
#   Phase A  (training)  → torchrun sees only these N cards, nproc=N
#   Phase A.5 (merge)    → uses GPUS_ARR[0] instead of hardcoded cuda:0
#   Phase B  (eval)      → round-robin GPUS_ARR
# Whole pipeline stays on user's chosen cards.
if [[ -n "$GPU_CSV" ]]; then
    IFS=',' read -r -a GPUS_ARR <<< "$GPU_CSV"
    SFT_GPUS="${#GPUS_ARR[@]}"    # training shrinks to match
    EVAL_GPUS="${#GPUS_ARR[@]}"
    echo "[LORA-ABL] --gpus=$GPU_CSV constrains WHOLE pipeline (train+merge+eval)"
    echo "[LORA-ABL]   SFT_GPUS  = $SFT_GPUS  (nproc_per_node for torchrun)"
    echo "[LORA-ABL]   EVAL_GPUS = $EVAL_GPUS"
elif [[ "$AUTO_GPUS" -eq 1 ]]; then
    source "$SCRIPT_DIR/_select_gpus.sh"
    # Pick enough for eval-parallel; training will use the same set
    SEL=$(select_free_gpus "$EVAL_GPUS")
    IFS=',' read -r -a GPUS_ARR <<< "$SEL"
    SFT_GPUS="${#GPUS_ARR[@]}"
    echo "[LORA-ABL] auto-selected GPUs = $SEL  (train+merge+eval)"
    echo "[LORA-ABL]   SFT_GPUS  = $SFT_GPUS"
    echo "[LORA-ABL]   EVAL_GPUS = $EVAL_GPUS"
else
    GPUS_ARR=(); for i in $(seq 0 $((EVAL_GPUS - 1))); do GPUS_ARR+=("$i"); done
fi

# CVD string for phase-prefix so torchrun / python see only the chosen cards
CVD=$(IFS=,; echo "${GPUS_ARR[*]}")

LAUNCH="torchrun --standalone --nproc_per_node=$SFT_GPUS"
[[ "$SFT_GPUS" -le 1 ]] && LAUNCH="python"

# ── Phase A: one training per rank ───────────────────────────────────────
for R in "${RANKS[@]}"; do
    RUN_NAME="v161_t6_lora_r${R}"
    TRAIN_DIR="$ROOT/runs/training/$RUN_NAME"
    LOG="$LOG_DIR/lora_r${R}_train_${TS_ALL}.log"
    echo
    echo "[LORA-ABL] ===== TRAIN  rank=$R  max_steps=$MAX_STEP  ====="
    echo "[LORA-ABL]   log: $LOG"

    if [[ -d "$TRAIN_DIR" && "$DRY_RUN" -eq 0 ]]; then
        echo "[LORA-ABL]   wiping $TRAIN_DIR"
        rm -rf "$TRAIN_DIR"
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[LORA-ABL]   (dry-run, skip)"
        continue
    fi

    if ! CUDA_VISIBLE_DEVICES="$CVD" $LAUNCH scripts/validate/t6t7_train.py \
            --jsonl_path "$T6_SFT_JSONL" \
            --run_name "$RUN_NAME" \
            --init_ckpt "$BASELINE_CKPT" \
            --max_steps "$MAX_STEP" \
            --batch_size "$T6_BATCH_SIZE_SFT" \
            --grad_accum_steps "$T6_GRAD_ACCUM" \
            --lr "$T6_LR" \
            --max_seq_len 768 \
            --parallel ddp \
            --use_lora --lora_r "$R" \
            --hf_export_at_steps "$TARGET_STEPS_CSV" \
            > "$LOG" 2>&1; then
        echo "[LORA-ABL] ✗ rank=$R training FAILED — see $LOG"
        echo "rank=$R TRAIN_FAILED" >> "$MANIFEST"
        continue
    fi
    echo "rank=$R train ok" >> "$MANIFEST"
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[LORA-ABL] dry-run done"; exit 0
fi

# ── Phase A.5: merge each adapter → HF ckpt (one-time, for v16_eval) ─────
# Intermediate hf_step_<N>/ contains only the LoRA adapter (tiny,
# non-destructive to training). v16_eval doesn't speak peft, so we merge
# once here into hf_step_<N>_merged/ and point eval there.
echo
echo "[LORA-ABL] ===== MERGE  adapter → HF ckpt (one-time) ====="
IFS=',' read -r -a STEPS_ARR <<< "$TARGET_STEPS_CSV"

for R in "${RANKS[@]}"; do
    TRAIN_DIR="$ROOT/runs/training/v161_t6_lora_r${R}"
    for S in "${STEPS_ARR[@]}"; do
        ADAPTER_DIR="$TRAIN_DIR/hf_step_${S}"
        MERGED_DIR="$TRAIN_DIR/hf_step_${S}_merged"
        if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ]]; then
            echo "[LORA-ABL] ✗ no adapter at $ADAPTER_DIR (step $S not exported?)"
            continue
        fi
        if [[ -f "$MERGED_DIR/config.json" ]]; then
            echo "[LORA-ABL]   already merged: $MERGED_DIR  (skip)"
            continue
        fi
        echo "[LORA-ABL]   merge r=$R step=$S  →  $MERGED_DIR"
        CUDA_VISIBLE_DEVICES=${GPUS_ARR[0]} python - <<PY
import sys, shutil
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import torch

adapter = Path("$ADAPTER_DIR")
merged  = Path("$MERGED_DIR")
base_id_or_path = "$BASELINE_CKPT"
merged.mkdir(parents=True, exist_ok=True)

# Resolve base to local path FIRST (offline-safe). Project convention:
# checkpoints/llada-instruct/ is materialised by scripts/download_models.py
# in Phase 0. Falling through to HF id requires network/cache.
base_local = Path("checkpoints/llada-instruct")
if base_local.is_dir() and (base_local / "config.json").exists():
    base_path = str(base_local)
else:
    base_path = base_id_or_path  # HF id fallback (requires cache or net)

base = AutoModel.from_pretrained(
    base_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    device_map={"": 0},
)
m = PeftModel.from_pretrained(base, adapter)
m = m.merge_and_unload()
m.save_pretrained(merged, safe_serialization=True)
tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
tok.save_pretrained(merged)

# copy trust_remote_code files from base local dir
if base_local.is_dir():
    for nm in ("modeling_llada.py", "configuration_llada.py", "tokenization_llada.py"):
        src = base_local / nm
        if src.is_file():
            shutil.copy2(src, merged / nm)
print(f"merged → {merged}")
PY
    done
done
wait

# ── Phase B: eval each (rank × step) cell in parallel on $EVAL_GPUS ──────
echo
echo "[LORA-ABL] ===== EVAL  (parallel on $EVAL_GPUS GPUs) ====="

declare -a PIDS=()
declare -a PID_LABELS=()
g=0
for R in "${RANKS[@]}"; do
    TRAIN_DIR="$ROOT/runs/training/v161_t6_lora_r${R}"
    for S in "${STEPS_ARR[@]}"; do
        HF_CKPT="$TRAIN_DIR/hf_step_${S}_merged"
        [[ ! -f "$HF_CKPT/config.json" ]] && {
            echo "[LORA-ABL] ✗ missing $HF_CKPT (merge failed?)"
            echo "r=$R step=$S MISSING_MERGED" >> "$MANIFEST"
            continue
        }
        EVAL_OUT="$ABL_DIR/r${R}_step${S}"
        EVAL_LOG="$LOG_DIR/lora_r${R}_eval_step${S}_${TS_ALL}.log"
        GPU="${GPUS_ARR[$g]}"
        echo "[LORA-ABL]   launching r=$R step=$S on GPU $GPU (slot $g) → $EVAL_OUT"

        CUDA_VISIBLE_DEVICES=$GPU python scripts/validate/v16_eval.py \
            --ckpts "t6=$HF_CKPT" \
            --out_dir "$EVAL_OUT" \
            --gen_length "$EVAL_GEN_LENGTH" \
            --block_length "$EVAL_BLOCK_LENGTH" \
            --temperature "$EVAL_TEMPERATURE" \
            > "$EVAL_LOG" 2>&1 &
        PIDS+=($!)
        PID_LABELS+=("r=$R step=$S")
        g=$(( (g + 1) % EVAL_GPUS ))
        if [[ "${#PIDS[@]}" -ge "$EVAL_GPUS" ]]; then
            wait "${PIDS[0]}" || echo "[LORA-ABL] ✗ ${PID_LABELS[0]} FAILED"
            PIDS=("${PIDS[@]:1}")
            PID_LABELS=("${PID_LABELS[@]:1}")
        fi
    done
done
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || echo "[LORA-ABL] ✗ ${PID_LABELS[$i]} FAILED"
done

# stash metadata on completed cells
for R in "${RANKS[@]}"; do
    for S in "${STEPS_ARR[@]}"; do
        EVAL_OUT="$ABL_DIR/r${R}_step${S}"
        [[ ! -f "$EVAL_OUT/summary.json" ]] && {
            echo "r=$R step=$S EVAL_NO_OUTPUT" >> "$MANIFEST"; continue; }
        python - <<PY
import json
from pathlib import Path
spe = int("$STEPS_PER_EPOCH")
p = Path("$EVAL_OUT/ablate_meta.json")
# _ts stamp lets the aggregator filter out cells from prior runs
p.write_text(json.dumps(
    {"rank": int("$R"), "step": int("$S"), "epoch": int("$S")/spe,
     "_ts": "$TS_ALL"}, indent=2))
PY
        echo "r=$R step=$S ok → $EVAL_OUT" >> "$MANIFEST"
    done
done

# ── Phase C: aggregate ───────────────────────────────────────────────────
echo
echo "[LORA-ABL] aggregating summary..."
python - <<PY
import json
from pathlib import Path

abl = Path("$ABL_DIR")
spe = int("$STEPS_PER_EPOCH")

rows = []
this_ts = "$TS_ALL"
for d in abl.glob("r*_step*"):
    meta_p = d / "ablate_meta.json"
    if not meta_p.exists(): continue
    meta = json.load(open(meta_p))
    # filter to THIS run's cells only
    if meta.get("_ts") != this_ts: continue
    sj = d / "summary.json"
    if not sj.exists():
        rows.append((meta, None)); continue
    data = json.load(open(sj))
    t6 = next((c for c in data.get("ckpts", []) if c["label"] == "t6"), None)
    rows.append((meta, t6))

# group by (rank, step) — sort by rank then step
rows.sort(key=lambda r: (r[0].get("rank", 1e9), r[0].get("step", 1e9)))

lines = [
    "# T6 LoRA rank × epoch ablation",
    "",
    f"1 epoch ≈ {spe} steps.  baseline is definitionally 0/N_fail, N_ok/N_ok.",
    "",
    "| rank | epoch | step | fail rescued | ok retention | Δ fail | Δ ok | net | FAIL18 | ceiling-5 |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for meta, t6 in rows:
    r = meta.get("rank", "?")
    s = meta.get("step", "?")
    e = f"{meta.get('epoch', 0):.2f}" if t6 else "?"
    if t6 is None:
        lines.append(f"| {r} | {e} | {s} | — | — | — | — | — | — | — |")
        continue
    # baseline synthesized
    n_fail = t6["n_fail"]; n_ok = t6["n_ok"]
    fail_r = t6["fail_correct"]; ok_r = t6["ok_correct"]
    d_fail = fail_r - 0
    d_ok   = ok_r - n_ok
    net    = d_fail + d_ok
    lines.append(
        f"| {r} | {e} | {s} "
        f"| {fail_r}/{n_fail} ({t6['fail_pass@1']:.1%}) "
        f"| {ok_r}/{n_ok} ({t6['ok_pass@1']:.1%}) "
        f"| +{d_fail} | {d_ok:+d} | {net:+d} "
        f"| {t6['fail18_rescued_count']}/18 "
        f"| {t6['ceiling_broken_count']}/5 |"
    )
lines += [
    "",
    "**Expected shape**: small r (1-4) → less forgetting but less capacity.",
    "  Bigger r (8-16) → more capacity, approaching full-SFT forgetting.",
    "  Longer epoch → more rescue but also more forgetting (within each rank).",
    "**Target**: max net, ok retention ≥ 95%.",
    "",
]
out_md = abl / f"summary_${TS_ALL}.md"
out_md.write_text("\n".join(lines), encoding="utf-8")
(abl / "summary.md").write_text("\n".join(lines), encoding="utf-8")
print(open(out_md).read())
print(f"\n[LORA-ABL] summary → {out_md}")
PY

echo "[LORA-ABL] done. → $ABL_DIR"
