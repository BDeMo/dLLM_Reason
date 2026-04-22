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
BASELINE_CKPT="GSAI-ML/LLaDA-8B-Instruct"

RANKS_CSV="1,2,4,8,16"
EPOCHS_CSV="0.5,1,2,4"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ranks)    RANKS_CSV="$2"; shift 2 ;;
        --epochs)   EPOCHS_CSV="$2"; shift 2 ;;
        --dry_run)  DRY_RUN=1; shift ;;
        -h|--help)  grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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

    if ! $LAUNCH scripts/validate/t6t7_train.py \
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

# ── Phase B: eval each (rank × step) cell ────────────────────────────────
echo
echo "[LORA-ABL] ===== EVAL  each cell (t6 only; baseline is constant) ====="
IFS=',' read -r -a STEPS_ARR <<< "$TARGET_STEPS_CSV"

for R in "${RANKS[@]}"; do
    TRAIN_DIR="$ROOT/runs/training/v161_t6_lora_r${R}"
    for S in "${STEPS_ARR[@]}"; do
        HF_CKPT="$TRAIN_DIR/hf_step_${S}"
        [[ ! -f "$HF_CKPT/config.json" ]] && {
            echo "[LORA-ABL] ✗ missing $HF_CKPT"
            echo "r=$R step=$S MISSING_HF" >> "$MANIFEST"
            continue
        }
        EVAL_OUT="$ABL_DIR/r${R}_step${S}"
        EVAL_LOG="$LOG_DIR/lora_r${R}_eval_step${S}_${TS_ALL}.log"
        echo "[LORA-ABL]   eval r=$R step=$S → $EVAL_OUT"

        if ! python scripts/validate/v16_eval.py \
                --ckpts "t6=$HF_CKPT" \
                --out_dir "$EVAL_OUT" \
                --gen_length "$EVAL_GEN_LENGTH" \
                --block_length "$EVAL_BLOCK_LENGTH" \
                --temperature "$EVAL_TEMPERATURE" \
                > "$EVAL_LOG" 2>&1; then
            echo "[LORA-ABL] ✗ r=$R step=$S eval FAILED"
            echo "r=$R step=$S EVAL_FAILED" >> "$MANIFEST"
            continue
        fi
        python - <<PY
import json
from pathlib import Path
spe = int("$STEPS_PER_EPOCH")
p = Path("$EVAL_OUT/ablate_meta.json")
p.write_text(json.dumps(
    {"rank": int("$R"), "step": int("$S"), "epoch": int("$S")/spe},
    indent=2))
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
for d in abl.glob("r*_step*"):
    meta_p = d / "ablate_meta.json"
    if not meta_p.exists(): continue
    meta = json.load(open(meta_p))
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
