#!/usr/bin/env bash
# t6_ablate.sh — T6 full-SFT ablation over training duration (epochs).
#
# v1.6.1 default max_steps=2000 on a 1350-sample train split (~12 epochs)
# overfits: fail +26.6% but ok -27%, net -179. Sweep shorter durations
# to find the fail/ok Pareto sweet spot.
#
# Epoch → step: 1 epoch ≈ 1350/8 = 169 steps (train=1350, bs=1 × world=8
# = 8 samples per backward).
#
# Default: epochs {0.5, 1, 2, 4} → steps {85, 169, 338, 676}.
#
# For each E:
#   1. wipe runs/training/v161_t6
#   2. run Phase 4 (SFT --t6_max_steps S) + Phase 5 (eval)
#   3. rename eval dir → runs/validation/t6_ablate/epoch_<E>/
# Final: runs/validation/t6_ablate/summary.md
#
# Usage:
#   bash scripts/t6_ablate.sh                     # default 0.5 1 2 4
#   bash scripts/t6_ablate.sh 0.25 0.5 1 2 4 8
#   bash scripts/t6_ablate.sh --dry_run 1
#
# (LoRA rank ablation: planned follow-up — run this first, decide whether
#  full-SFT at the best epoch is good enough before adding the LoRA axis.)

set -euo pipefail

DRY_RUN=0
TRAIN_N=1350
SAMPLES_PER_STEP=8              # bs 1 × world 8
STEPS_PER_EPOCH=$(( TRAIN_N / SAMPLES_PER_STEP ))   # = 169

EPOCHS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry_run)  DRY_RUN=1; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) EPOCHS+=("$1"); shift ;;
    esac
done
[[ "${#EPOCHS[@]}" -eq 0 ]] && EPOCHS=(0.5 1 2 4)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

ABL_DIR="$ROOT/runs/validation/t6_ablate"
mkdir -p "$ABL_DIR"
LOG_DIR="$ROOT/runs/ablate_logs"
mkdir -p "$LOG_DIR"

TS_ALL=$(date +%Y%m%d_%H%M%S)
MANIFEST="$ABL_DIR/manifest_${TS_ALL}.txt"
{
    echo "ablation started: $TS_ALL"
    echo "EPOCHS           = ${EPOCHS[*]}"
    echo "STEPS_PER_EPOCH  = $STEPS_PER_EPOCH"
    echo "DRY_RUN          = $DRY_RUN"
} > "$MANIFEST"

echo "[ABL] ============================================================"
echo "[ABL]   T6 full-SFT epoch sweep"
echo "[ABL]   EPOCHS          = ${EPOCHS[*]}"
echo "[ABL]   steps_per_epoch = $STEPS_PER_EPOCH"
echo "[ABL]   total runs      = ${#EPOCHS[@]}"
echo "[ABL]   output dir      = $ABL_DIR"
echo "[ABL] ============================================================"

T6_CKPT_DIR="$ROOT/runs/training/v161_t6"

epoch_to_steps() {
    python - <<PY
e = float("$1")
spe = int("$STEPS_PER_EPOCH")
print(max(1, round(e * spe)))
PY
}

dec_label() {
    python -c "e=float('$1'); print(f'{e:.2f}'.rstrip('0').rstrip('.').replace('.', 'p'))"
}

idx=0
total=${#EPOCHS[@]}
for E in "${EPOCHS[@]}"; do
    idx=$(( idx + 1 ))
    S=$(epoch_to_steps "$E")
    LAB=$(dec_label "$E")
    LABEL="epoch_${LAB}"
    LOG="$LOG_DIR/${LABEL}_${TS_ALL}.log"
    echo
    echo "[ABL] ===== [$idx/$total] epoch=$E  ($S steps) ====="

    if [[ -d "$T6_CKPT_DIR" && "$DRY_RUN" -eq 0 ]]; then
        echo "[ABL] wiping $T6_CKPT_DIR"
        rm -rf "$T6_CKPT_DIR"
    fi

    dry_flag=()
    [[ "$DRY_RUN" -eq 1 ]] && dry_flag=(--dry_run)

    echo "[ABL] Phase 4+5  log: $LOG"
    if ! bash scripts/run_all_v1.6.1.sh \
            --from_phase 4 --to_phase 5 \
            --t6_max_steps "$S" \
            "${dry_flag[@]}" \
            > "$LOG" 2>&1; then
        echo "[ABL] ✗ epoch=$E FAILED — see $LOG"
        echo "epoch=$E steps=$S FAILED" >> "$MANIFEST"
        continue
    fi
    [[ "$DRY_RUN" -eq 1 ]] && { echo "[ABL] dry: skip rename/collect"; continue; }

    latest_eval=$(ls -dt "$ROOT"/runs/validation/v161_eval_* 2>/dev/null | head -1)
    if [[ -z "$latest_eval" || ! -d "$latest_eval" ]]; then
        echo "[ABL] ✗ epoch=$E: no v161_eval_* dir produced"
        echo "epoch=$E steps=$S NO_EVAL_DIR" >> "$MANIFEST"
        continue
    fi
    DEST="$ABL_DIR/$LABEL"
    rm -rf "$DEST"
    mv "$latest_eval" "$DEST"
    echo "{\"epoch\": $E, \"steps\": $S}" > "$DEST/ablate_meta.json"
    echo "[ABL]   $LABEL → $DEST"
    echo "epoch=$E steps=$S ok → $DEST" >> "$MANIFEST"
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

rows = []
for d in abl.glob("epoch_*"):
    meta_p = d / "ablate_meta.json"
    if not meta_p.exists(): continue
    meta = json.load(open(meta_p))
    sj = d / "summary.json"
    if not sj.exists():
        rows.append((meta, None, None)); continue
    data = json.load(open(sj))
    ckpts = data.get("ckpts", [])
    base = next((c for c in ckpts if c["label"] == "baseline"), None)
    t6   = next((c for c in ckpts if c["label"] == "t6"), None)
    rows.append((meta, base, t6))
rows.sort(key=lambda r: r[0].get("epoch", 1e9))

lines = [
    "# T6 full-SFT epoch ablation",
    "",
    f"1 epoch ≈ {spe} steps (train=1350, bs=1 × world=8 → 8 samples/step).",
    "",
    "| epoch | steps | fail rescued | ok retention | Δ fail | Δ ok | net | FAIL18 | ceiling-5 |",
    "|---|---|---|---|---|---|---|---|---|",
]
for meta, base, t6 in rows:
    e = meta.get("epoch", "?")
    s = meta.get("steps", "?")
    if base is None or t6 is None:
        lines.append(f"| {e} | {s} | — | — | — | — | — | — | — |")
        continue
    fail_r = t6["fail_correct"]; ok_r = t6["ok_correct"]
    d_fail = fail_r - base["fail_correct"]
    d_ok   = ok_r   - base["ok_correct"]
    net    = d_fail + d_ok
    lines.append(
        f"| {e} | {s} "
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
