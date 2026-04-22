#!/usr/bin/env bash
# t6_ablate.sh — 2D ablation over T6 SFT training duration (epochs) ×
# training mode (full SFT vs LoRA). Finds the fail-rescue vs ok-retention
# Pareto sweet spot against catastrophic forgetting.
#
# Context: v1.6.1 T6 SFT at default max_steps=2000 on a 1350-sample
# train split (effective batch = 1 × 16 × 8 = 128; backward every 8
# samples → ~12 epochs) overfits: fail +26.6% but ok -27%, net -179.
#
# Two regimes compared at every epoch value:
#   - full : all 8B params trainable (FSDP FULL_SHARD). Max capacity but
#            strongest catastrophic forgetting.
#   - lora : base frozen, only LoRA adapters trained (DDP, since optim
#            state is tiny). Implicit regularizer — forgetting drops a
#            lot; expected Pareto better for small-data SFT.
#
# Epoch ↔ step conversion:
#   1 step  = 1 forward + 1 backward (grad_accum'd)
#   samples = steps × batch_size × world_size   (= steps × 8)
#   epoch   = steps × 8 / 1350
#   → STEPS_PER_EPOCH = 1350 / 8 = 169 (rounded)
#
# Default: epochs {0.5, 1, 2, 4} × modes {full, lora} = 8 runs.
#
# For each (E, mode) in EPOCHS × MODES:
#   1. compute S = round(E × STEPS_PER_EPOCH)
#   2. wipe runs/training/v161_t6
#   3. run Phase 4 (SFT --t6_max_steps S [--t6_use_lora]) + Phase 5 (eval)
#   4. rename eval dir → runs/validation/ablate_t6_epochs/epoch_<E>_<mode>
#   5. collect summary.json
# Final: write runs/validation/ablate_t6_epochs/summary.md
#
# Usage:
#   bash scripts/t6_ablate.sh                       # default 0.5 1 2 4 × {full,lora}
#   bash scripts/t6_ablate.sh --mode full 0.5 1 2   # full-SFT only
#   bash scripts/t6_ablate.sh --mode lora 1 2 4 8   # LoRA only
#   bash scripts/t6_ablate.sh --mode both 0.25 0.5 1 2 4 8
#   bash scripts/t6_ablate.sh --dry_run 1

set -euo pipefail

MODE="both"                     # full | lora | both
DRY_RUN=0
TRAIN_N=1350                    # from v1.6.1 t6_sft.jsonl split
SAMPLES_PER_STEP=8              # batch_size 1 × world_size 8
STEPS_PER_EPOCH=$(( TRAIN_N / SAMPLES_PER_STEP ))   # = 169

EPOCHS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)     MODE="$2"; shift 2 ;;
        --dry_run)  DRY_RUN=1;  shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) EPOCHS+=("$1"); shift ;;
    esac
done
[[ "${#EPOCHS[@]}" -eq 0 ]] && EPOCHS=(0.5 1 2 4)

case "$MODE" in
    full) MODES=(full) ;;
    lora) MODES=(lora) ;;
    both) MODES=(full lora) ;;
    *) echo "[ABL] invalid --mode: $MODE (want full|lora|both)" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

ABL_DIR="$ROOT/runs/validation/ablate_t6_epochs"
mkdir -p "$ABL_DIR"
LOG_DIR="$ROOT/runs/ablate_logs"
mkdir -p "$LOG_DIR"

TS_ALL=$(date +%Y%m%d_%H%M%S)
MANIFEST="$ABL_DIR/manifest_${TS_ALL}.txt"
{
    echo "ablation started: $TS_ALL"
    echo "EPOCHS           = ${EPOCHS[*]}"
    echo "MODES            = ${MODES[*]}"
    echo "STEPS_PER_EPOCH  = $STEPS_PER_EPOCH  (train=$TRAIN_N, samples/step=$SAMPLES_PER_STEP)"
    echo "DRY_RUN          = $DRY_RUN"
} > "$MANIFEST"

echo "[ABL] ============================================================"
echo "[ABL]   T6 2D ablation:  epochs=(${EPOCHS[*]})  modes=(${MODES[*]})"
echo "[ABL]   steps_per_epoch = $STEPS_PER_EPOCH"
echo "[ABL]   total runs = $(( ${#EPOCHS[@]} * ${#MODES[@]} ))"
echo "[ABL]   output dir: $ABL_DIR"
echo "[ABL] ============================================================"

T6_CKPT_DIR="$ROOT/runs/training/v161_t6"

epoch_to_steps() {
    local e="$1"
    python - <<PY
e = float("$e")
spe = int("$STEPS_PER_EPOCH")
print(max(1, round(e * spe)))
PY
}

run_count=0
total=$(( ${#EPOCHS[@]} * ${#MODES[@]} ))
for E in "${EPOCHS[@]}"; do
    S=$(epoch_to_steps "$E")
    LABEL="$(python -c "e=float('$E'); print(f'{e:.2f}'.rstrip('0').rstrip('.').replace('.', 'p'))")"

    for MODE_I in "${MODES[@]}"; do
        run_count=$(( run_count + 1 ))
        echo
        echo "[ABL] ===== [$run_count/$total] epoch=$E ($S steps) mode=$MODE_I ====="
        LOG="$LOG_DIR/ablate_epoch_${LABEL}_${MODE_I}_${TS_ALL}.log"

        if [[ -d "$T6_CKPT_DIR" && "$DRY_RUN" -eq 0 ]]; then
            echo "[ABL] wiping $T6_CKPT_DIR"
            rm -rf "$T6_CKPT_DIR"
        fi

        EXTRA_FLAGS=()
        if [[ "$MODE_I" == "lora" ]]; then
            EXTRA_FLAGS+=(--t6_use_lora --t6_parallel ddp)
        fi
        # full → default (fsdp), no extra flags

        dry_flag=()
        [[ "$DRY_RUN" -eq 1 ]] && dry_flag=(--dry_run)

        echo "[ABL] Phase 4+5  log: $LOG"
        if ! bash scripts/run_all_v1.6.1.sh \
                --from_phase 4 --to_phase 5 \
                --t6_max_steps "$S" \
                "${EXTRA_FLAGS[@]}" \
                "${dry_flag[@]}" \
                > "$LOG" 2>&1; then
            echo "[ABL] ✗ epoch=$E mode=$MODE_I FAILED — see $LOG"
            echo "epoch=$E steps=$S mode=$MODE_I FAILED" >> "$MANIFEST"
            continue
        fi

        [[ "$DRY_RUN" -eq 1 ]] && { echo "[ABL] dry: skip rename/collect"; continue; }

        latest_eval=$(ls -dt "$ROOT"/runs/validation/v161_eval_* 2>/dev/null | head -1)
        if [[ -z "$latest_eval" || ! -d "$latest_eval" ]]; then
            echo "[ABL] ✗ epoch=$E mode=$MODE_I: no v161_eval_* dir produced"
            echo "epoch=$E steps=$S mode=$MODE_I NO_EVAL_DIR" >> "$MANIFEST"
            continue
        fi
        DEST="$ABL_DIR/epoch_${LABEL}_${MODE_I}"
        rm -rf "$DEST"
        mv "$latest_eval" "$DEST"
        echo "{\"epoch\": $E, \"steps\": $S, \"mode\": \"$MODE_I\"}" \
            > "$DEST/ablate_meta.json"
        echo "[ABL]   epoch=$E steps=$S mode=$MODE_I → $DEST"
        echo "epoch=$E steps=$S mode=$MODE_I ok → $DEST" >> "$MANIFEST"
    done
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[ABL] dry-run done"; exit 0
fi

echo
echo "[ABL] aggregating summary..."
python - <<PY
import json
from pathlib import Path

abl = Path("$ABL_DIR")
spe = int("$STEPS_PER_EPOCH")

def load_run(d):
    meta_p = d / "ablate_meta.json"
    meta = json.load(open(meta_p)) if meta_p.exists() else {}
    sj = d / "summary.json"
    if not sj.exists():
        return meta, None, None
    data = json.load(open(sj))
    ckpts = data.get("ckpts", [])
    base = next((c for c in ckpts if c["label"] == "baseline"), None)
    t6   = next((c for c in ckpts if c["label"] == "t6"), None)
    return meta, base, t6

rows = []
for d in abl.glob("epoch_*"):
    meta, base, t6 = load_run(d)
    rows.append((d.name, meta, base, t6))
# sort by (mode, epoch) so all full-SFT rows come together, then LoRA
rows.sort(key=lambda r: (r[1].get("mode", "z"), r[1].get("epoch", 1e9))
          if r[1] else ("z", 1e9))

lines = [
    "# T6 SFT ablation — epoch × mode",
    "",
    "Fail-rescue vs ok-retention Pareto sweep over training duration and mode.",
    f"Physical: 1 epoch ≈ {spe} steps (train split = 1350, backward every 8 samples).",
    "",
    "- **full**: all 8B params trainable (FSDP FULL_SHARD). Max capacity; strongest forgetting.",
    "- **lora**: base frozen, LoRA adapters only (DDP). Regularizes against forgetting.",
    "",
    "| mode | epoch | steps | fail rescued | ok retention | Δ fail | Δ ok | net | FAIL18 | ceiling-5 |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for name, meta, base, t6 in rows:
    mode = meta.get("mode", "?")
    e = meta.get("epoch", "?")
    s = meta.get("steps", "?")
    if base is None or t6 is None:
        lines.append(f"| {mode} | {e} | {s} | — | — | — | — | — | — | — |")
        continue
    fail_r = t6["fail_correct"]; ok_r = t6["ok_correct"]
    fail_b = base["fail_correct"]; ok_b = base["ok_correct"]
    d_fail = fail_r - fail_b
    d_ok   = ok_r   - ok_b
    net    = d_fail + d_ok
    lines.append(
        f"| {mode} | {e} | {s} "
        f"| {fail_r}/{t6['n_fail']} ({t6['fail_pass@1']:.1%}) "
        f"| {ok_r}/{t6['n_ok']} ({t6['ok_pass@1']:.1%}) "
        f"| +{d_fail} | {d_ok:+d} | {net:+d} "
        f"| {t6['fail18_rescued_count']}/18 "
        f"| {t6['ceiling_broken_count']}/5 |"
    )
lines += [
    "",
    "**Target**: max net positive, ideally fail rescue ≥ 15% with ok retention ≥ 95%.",
    "Expected: LoRA dominates full-SFT on the Pareto (less forgetting at comparable rescue).",
    "",
]
out_md = abl / f"summary_${TS_ALL}.md"
out_md.write_text("\n".join(lines), encoding="utf-8")
(abl / "summary.md").write_text("\n".join(lines), encoding="utf-8")
print(open(out_md).read())
print(f"\n[ABL] summary → {out_md}")
PY

echo "[ABL] done. All artefacts under $ABL_DIR"
