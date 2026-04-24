#!/usr/bin/env bash
# t6_ablate.sh — T6 full-SFT ablation over training duration.
#
# KEY DESIGN (first principles): training is a deterministic sequence
# θ_0 → θ_1 → ... so θ at step N is identical whether reached fresh or
# mid-run. Running N independent trainings for N epoch values is N× waste.
# Instead: run ONE training to max(epochs), emit HF ckpts at each target
# step via --hf_export_at_steps, then eval each ckpt.
#
# Cost comparison (default 0.5/1/2/4 epochs):
#   naive: (0.5+1+2+4) × train + 4 × eval = 7.5× training
#   here:              4     × train (== max(target)) + 4 × eval
#   → ~45% training-time saving.
#
# v1.6.1 full-SFT at default ~12 epochs overfits: fail +26.6% but ok -27%.
# Sweep shorter durations to find the fail/ok Pareto sweet spot.
#
# Epoch → step: 1 epoch ≈ 1350/8 = 169 steps.
#
# Default: epochs {0.5, 1, 2, 4} → steps {85, 169, 338, 676}.
#
# Flow:
#   1. compute target_steps = [round(E × 169) for E in EPOCHS]
#   2. ONE torchrun t6t7_train.py --t6_max_steps max(target_steps)
#                                 --hf_export_at_steps <csv>
#      → runs/training/v161_t6_ablate/hf_step_<N>/ for each N
#   3. for each hf_step_<N>: run v16_eval.py vs baseline
#   4. aggregate summary.md
#
# Usage:
#   bash scripts/t6_ablate.sh                   # default 0.5 1 2 4
#   bash scripts/t6_ablate.sh 0.25 0.5 1 2 4 8
#   bash scripts/t6_ablate.sh --dry_run         # plan only, no train/eval

set -euo pipefail

DRY_RUN=0
TRAIN_N=1350
SAMPLES_PER_STEP=8              # bs 1 × world 8
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

EPOCHS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --eval_gpus) EVAL_GPUS="$2"; shift 2 ;;
        --gpus)      GPU_CSV="$2"; shift 2 ;;
        --auto_gpus) AUTO_GPUS=1; shift ;;
        --dry_run)   DRY_RUN=1; shift ;;
        -h|--help)   grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) EPOCHS+=("$1"); shift ;;
    esac
done
[[ "${#EPOCHS[@]}" -eq 0 ]] && EPOCHS=(0.5 1 2 4)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Resolve which GPU indices to dispatch to:
#   --gpus 0,2,4        → explicit list, EVAL_GPUS := len
#   --auto_gpus         → query nvidia-smi for EVAL_GPUS least-busy
#   (default)           → sequential 0..EVAL_GPUS-1
if [[ -n "$GPU_CSV" ]]; then
    IFS=',' read -r -a GPUS_ARR <<< "$GPU_CSV"
    EVAL_GPUS="${#GPUS_ARR[@]}"
elif [[ "$AUTO_GPUS" -eq 1 ]]; then
    source "$SCRIPT_DIR/_select_gpus.sh"
    SEL=$(select_free_gpus "$EVAL_GPUS")
    IFS=',' read -r -a GPUS_ARR <<< "$SEL"
    echo "[ABL] auto-selected GPUs: $SEL"
else
    GPUS_ARR=(); for i in $(seq 0 $((EVAL_GPUS - 1))); do GPUS_ARR+=("$i"); done
fi

ABL_DIR="$ROOT/runs/validation/t6_ablate"
TRAIN_DIR_NAME="v161_t6_ablate"
TRAIN_DIR="$ROOT/runs/training/$TRAIN_DIR_NAME"
mkdir -p "$ABL_DIR"
LOG_DIR="$ROOT/runs/ablate_logs"
mkdir -p "$LOG_DIR"

TS_ALL=$(date +%Y%m%d_%H%M%S)

# ── compute target steps from epochs ────────────────────────────────────
python_target_steps() {
python - <<PY
spe = int("$STEPS_PER_EPOCH")
epochs = [$(IFS=,; echo "${EPOCHS[*]}")]
steps = sorted(set(max(1, round(float(e) * spe)) for e in epochs))
print(",".join(str(s) for s in steps))
PY
}
TARGET_STEPS_CSV=$(python_target_steps)
MAX_STEP=$(python - <<PY
print(max(int(s) for s in "$TARGET_STEPS_CSV".split(",")))
PY
)

MANIFEST="$ABL_DIR/manifest_${TS_ALL}.txt"
{
    echo "ablation started: $TS_ALL"
    echo "EPOCHS            = ${EPOCHS[*]}"
    echo "TARGET_STEPS      = $TARGET_STEPS_CSV"
    echo "MAX_STEP (train to)= $MAX_STEP"
    echo "STEPS_PER_EPOCH   = $STEPS_PER_EPOCH"
    echo "DRY_RUN           = $DRY_RUN"
} > "$MANIFEST"

echo "[ABL] ============================================================"
echo "[ABL]   T6 full-SFT epoch ablation (single-training + multi-export)"
echo "[ABL]   epochs          = ${EPOCHS[*]}"
echo "[ABL]   target steps    = $TARGET_STEPS_CSV"
echo "[ABL]   train once to   = $MAX_STEP steps (max of targets)"
echo "[ABL]   output dir      = $ABL_DIR"
echo "[ABL] ============================================================"

# Resolve T6 SFT data (same as run_all_v1.6.1.sh Phase 4)
if [[ -z "${T6_SFT_JSONL:-}" ]]; then
    T6_RUN_DIR=$(ls -dt "$ROOT"/runs/validation/t6_teacher_trace_* 2>/dev/null | head -1)
    T6_SFT_JSONL="$T6_RUN_DIR/t6_sft.jsonl"
fi
if [[ ! -f "$T6_SFT_JSONL" ]]; then
    echo "[ABL] ERROR: T6_SFT_JSONL not found: $T6_SFT_JSONL" >&2
    echo "[ABL]   run Phase 3 first (teacher trace) or set env T6_SFT_JSONL" >&2
    exit 1
fi
echo "[ABL] T6 SFT data: $T6_SFT_JSONL"

# ── 1. ONE training run ─────────────────────────────────────────────────
echo
echo "[ABL] ===== Phase: single training run (max_steps=$MAX_STEP) ====="
TRAIN_LOG="$LOG_DIR/ablate_train_${TS_ALL}.log"

# wipe prior ablation run (we want a clean trajectory from θ_0)
if [[ -d "$TRAIN_DIR" && "$DRY_RUN" -eq 0 ]]; then
    echo "[ABL] wiping $TRAIN_DIR"
    rm -rf "$TRAIN_DIR"
fi

dry_flag=()
[[ "$DRY_RUN" -eq 1 ]] && dry_flag=(echo DRY)

echo "[ABL] launching torchrun  log: $TRAIN_LOG"
LAUNCH="torchrun --standalone --nproc_per_node=$SFT_GPUS"
[[ "$SFT_GPUS" -le 1 ]] && LAUNCH="python"

if ! "${dry_flag[@]}" $LAUNCH scripts/validate/t6t7_train.py \
        --jsonl_path "$T6_SFT_JSONL" \
        --run_name "$TRAIN_DIR_NAME" \
        --init_ckpt "$BASELINE_CKPT" \
        --max_steps "$MAX_STEP" \
        --batch_size "$T6_BATCH_SIZE_SFT" \
        --grad_accum_steps "$T6_GRAD_ACCUM" \
        --lr "$T6_LR" \
        --max_seq_len 768 \
        --parallel fsdp \
        --hf_export_at_steps "$TARGET_STEPS_CSV" \
        > "$TRAIN_LOG" 2>&1; then
    echo "[ABL] ✗ training FAILED — see $TRAIN_LOG"; exit 1
fi

[[ "$DRY_RUN" -eq 1 ]] && { echo "[ABL] dry: skip eval/aggregate"; exit 0; }

# ── 2. eval each hf_step_<N> — ONLY t6, baseline is definitionally 0/N_fail
# and N_ok/N_ok by how scope_fail / scope_ok were constructed (prompts
# baseline failed / succeeded under this same canonical config). Re-testing
# baseline each time is ~10-15 min × N of pure waste.
echo
echo "[ABL] ===== Phase: eval each exported ckpt (parallel on $EVAL_GPUS GPUs) ====="

IFS=',' read -r -a STEPS_ARR <<< "$TARGET_STEPS_CSV"
declare -a PIDS=()
declare -a PID_LABELS=()
g=0
for S in "${STEPS_ARR[@]}"; do
    HF_CKPT="$TRAIN_DIR/hf_step_${S}"
    if [[ ! -f "$HF_CKPT/config.json" ]]; then
        echo "[ABL] ✗ missing $HF_CKPT — step $S not exported; skipping"
        echo "step=$S MISSING_HF" >> "$MANIFEST"
        continue
    fi
    EVAL_OUT="$ABL_DIR/step_${S}"
    EVAL_LOG="$LOG_DIR/ablate_eval_step${S}_${TS_ALL}.log"
    GPU="${GPUS_ARR[$g]}"
    echo "[ABL]   launching eval step=$S on GPU $GPU (slot $g)  → $EVAL_OUT"
    CUDA_VISIBLE_DEVICES=$GPU python scripts/validate/v16_eval.py \
        --ckpts "t6=$HF_CKPT" \
        --out_dir "$EVAL_OUT" \
        --gen_length "$EVAL_GEN_LENGTH" \
        --block_length "$EVAL_BLOCK_LENGTH" \
        --temperature "$EVAL_TEMPERATURE" \
        > "$EVAL_LOG" 2>&1 &
    PIDS+=($!)
    PID_LABELS+=("step=$S")
    g=$(( (g + 1) % EVAL_GPUS ))
    # throttle when all GPU slots busy
    if [[ "${#PIDS[@]}" -ge "$EVAL_GPUS" ]]; then
        wait "${PIDS[0]}" || echo "[ABL] ✗ ${PID_LABELS[0]} FAILED"
        PIDS=("${PIDS[@]:1}")
        PID_LABELS=("${PID_LABELS[@]:1}")
    fi
done
# drain remaining
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || echo "[ABL] ✗ ${PID_LABELS[$i]} FAILED"
done

# stash metadata on completed runs
for S in "${STEPS_ARR[@]}"; do
    EVAL_OUT="$ABL_DIR/step_${S}"
    [[ ! -f "$EVAL_OUT/summary.json" ]] && {
        echo "step=$S EVAL_NO_OUTPUT" >> "$MANIFEST"; continue; }
    python - <<PY
import json
from pathlib import Path
p = Path("$EVAL_OUT/ablate_meta.json")
spe = int("$STEPS_PER_EPOCH")
s = int("$S")
# stamp this run's TS so the aggregator can filter out stale cells
# left over from prior ablations with different epoch sets
p.write_text(json.dumps({"step": s, "epoch": s / spe, "_ts": "$TS_ALL"}, indent=2))
PY
    echo "step=$S ok → $EVAL_OUT" >> "$MANIFEST"
done

# ── 3. aggregate summary ────────────────────────────────────────────────
echo
echo "[ABL] aggregating summary..."
python - <<PY
import json
from pathlib import Path

abl = Path("$ABL_DIR")
spe = int("$STEPS_PER_EPOCH")

rows = []
this_ts = "$TS_ALL"
for d in abl.glob("step_*"):
    meta_p = d / "ablate_meta.json"
    if not meta_p.exists(): continue
    meta = json.load(open(meta_p))
    # filter to THIS run's cells — without this, prior ablations with
    # different epoch sets would pollute the summary
    if meta.get("_ts") != this_ts: continue
    sj = d / "summary.json"
    if not sj.exists():
        rows.append((meta, None, None)); continue
    data = json.load(open(sj))
    ckpts = data.get("ckpts", [])
    t6   = next((c for c in ckpts if c["label"] == "t6"), None)
    # Baseline is definitionally 0/n_fail and n_ok/n_ok — synthesize it
    # from the t6 row's split sizes rather than re-testing (deterministic
    # by construction of scope_fail/scope_ok).
    if t6 is not None:
        base = {
            "label": "baseline",
            "n_fail": t6["n_fail"], "n_ok": t6["n_ok"],
            "fail_correct": 0, "ok_correct": t6["n_ok"],
            "fail_pass@1": 0.0, "ok_pass@1": 1.0,
        }
    else:
        base = None
    rows.append((meta, base, t6))
rows.sort(key=lambda r: r[0].get("step", 1e9))

lines = [
    "# T6 full-SFT ablation — single-training + multi-ckpt-export",
    "",
    f"1 epoch ≈ {spe} steps.  Single training run to max(target_steps), with",
    "HF exports at each target step (deterministic mid-run checkpoints).",
    "",
    "| step | epoch | fail rescued | ok retention | Δ fail | Δ ok | net | FAIL18 | ceiling-5 |",
    "|---|---|---|---|---|---|---|---|---|",
]
for meta, base, t6 in rows:
    s = meta.get("step", "?")
    e = f"{meta.get('epoch', 0):.2f}" if base else "?"
    if base is None or t6 is None:
        lines.append(f"| {s} | {e} | — | — | — | — | — | — | — |")
        continue
    fail_r = t6["fail_correct"]; ok_r = t6["ok_correct"]
    d_fail = fail_r - base["fail_correct"]
    d_ok   = ok_r   - base["ok_correct"]
    net    = d_fail + d_ok
    lines.append(
        f"| {s} | {e} "
        f"| {fail_r}/{t6['n_fail']} ({t6['fail_pass@1']:.1%}) "
        f"| {ok_r}/{t6['n_ok']} ({t6['ok_pass@1']:.1%}) "
        f"| +{d_fail} | {d_ok:+d} | {net:+d} "
        f"| {t6['fail18_rescued_count']}/18 "
        f"| {t6['ceiling_broken_count']}/5 |"
    )
lines += [
    "",
    "**Target**: max net positive, ideally fail rescue ≥ 15% with ok retention ≥ 95%.",
    "",
]
out_md = abl / f"summary_${TS_ALL}.md"
out_md.write_text("\n".join(lines), encoding="utf-8")
(abl / "summary.md").write_text("\n".join(lines), encoding="utf-8")
print(open(out_md).read())
print(f"\n[ABL] summary → {out_md}")
PY

echo "[ABL] done. All artefacts under $ABL_DIR"
