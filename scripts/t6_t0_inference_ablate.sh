#!/usr/bin/env bash
# t6_t0_inference_ablate.sh — T=0 (greedy) inference-hyperparameter
# ablation on a T6-trained ckpt. Replicates A4 (block_length) + A6
# (gen_length) findings but under the trained model instead of vanilla
# LLaDA. Complements t6_decode_ablate (which varies temperature/N but
# fixes block/gen at canonical).
#
# Canonical baseline: gen=128 bl=32 T=0 → already captured by
# t6_ablate's own summary. This script sweeps:
#   - block_length ∈ {16, 32, 64}   (A4 axis)
#   - gen_length   ∈ {128, 192, 256} (A6 axis)
#
# Grid: 3 × 3 = 9 cells, T=0, pass@1 on full scope (331 fail + 988 ok).
# Parallel across GPUs (one cell per GPU).
#
# Usage:
#   bash scripts/t6_t0_inference_ablate.sh \
#       --ckpt runs/training/v161_t6_ablate/hf_step_336
#   bash scripts/t6_t0_inference_ablate.sh \
#       --ckpt .../hf_step_336 --block_lengths 16,32 --gen_lengths 128,192
#   bash scripts/t6_t0_inference_ablate.sh --ckpt ... --auto_gpus

set -euo pipefail

CKPT=""
BLOCK_CSV="16,32,64"
GEN_CSV="128,192,256"
TEMPERATURE=0
N_FAIL=331
N_OK=988
EVAL_GPUS=8
GPU_CSV=""
AUTO_GPUS=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)           CKPT="$2"; shift 2 ;;
        --block_lengths)  BLOCK_CSV="$2"; shift 2 ;;
        --gen_lengths)    GEN_CSV="$2"; shift 2 ;;
        --temperature)    TEMPERATURE="$2"; shift 2 ;;
        --n_fail)         N_FAIL="$2"; shift 2 ;;
        --n_ok)           N_OK="$2"; shift 2 ;;
        --eval_gpus)      EVAL_GPUS="$2"; shift 2 ;;
        --gpus)           GPU_CSV="$2"; shift 2 ;;
        --auto_gpus)      AUTO_GPUS=1; shift ;;
        --dry_run)        DRY_RUN=1; shift ;;
        -h|--help)        grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[T0-ABL] unknown arg: $1" >&2; exit 1 ;;
    esac
done
[[ -z "$CKPT" ]] && { echo "[T0-ABL] --ckpt required" >&2; exit 2; }
[[ ! -f "$CKPT/config.json" ]] && { echo "[T0-ABL] missing $CKPT/config.json" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

IFS=',' read -r -a BLOCKS <<< "$BLOCK_CSV"
IFS=',' read -r -a GENS   <<< "$GEN_CSV"
N_CELLS=$(( ${#BLOCKS[@]} * ${#GENS[@]} ))

if [[ "$EVAL_GPUS" -gt "$N_CELLS" ]]; then EVAL_GPUS=$N_CELLS; fi

if [[ -n "$GPU_CSV" ]]; then
    IFS=',' read -r -a GPUS_ARR <<< "$GPU_CSV"
    if [[ "${#GPUS_ARR[@]}" -gt "$N_CELLS" ]]; then
        GPUS_ARR=("${GPUS_ARR[@]:0:$N_CELLS}")
    fi
    EVAL_GPUS="${#GPUS_ARR[@]}"
elif [[ "$AUTO_GPUS" -eq 1 ]]; then
    source "$SCRIPT_DIR/_select_gpus.sh"
    SEL=$(select_free_gpus "$EVAL_GPUS")
    IFS=',' read -r -a GPUS_ARR <<< "$SEL"
    echo "[T0-ABL]   auto-GPUs = $SEL"
else
    GPUS_ARR=(); for i in $(seq 0 $((EVAL_GPUS - 1))); do GPUS_ARR+=("$i"); done
fi

CKPT_LABEL=$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")
OUT_BASE="$ROOT/runs/validation/t6_t0_inference_ablate/$CKPT_LABEL"
mkdir -p "$OUT_BASE"
LOG_DIR="$ROOT/runs/ablate_logs"
mkdir -p "$LOG_DIR"

TS=$(date +%Y%m%d_%H%M%S)

echo "[T0-ABL] ============================================================"
echo "[T0-ABL]   ckpt         = $CKPT"
echo "[T0-ABL]   block_lens   = ${BLOCKS[*]}"
echo "[T0-ABL]   gen_lens     = ${GENS[*]}"
echo "[T0-ABL]   T            = $TEMPERATURE"
echo "[T0-ABL]   n_fail/ok    = $N_FAIL / $N_OK"
echo "[T0-ABL]   n_cells      = $N_CELLS   eval_gpus = $EVAL_GPUS"
echo "[T0-ABL]   out          = $OUT_BASE"
echo "[T0-ABL] ============================================================"

declare -a PIDS=() PID_LABELS=()
g=0
for BL in "${BLOCKS[@]}"; do
    for GL in "${GENS[@]}"; do
        CELL="bl${BL}_gl${GL}"
        RUN_DIR="$OUT_BASE/${CELL}_${TS}"
        LOG="$LOG_DIR/t0_ablate_${CKPT_LABEL}_${CELL}_${TS}.log"
        GPU="${GPUS_ARR[$g]}"
        echo "[T0-ABL]   launch $CELL on GPU $GPU → $RUN_DIR"
        [[ "$DRY_RUN" -eq 1 ]] && continue

        CUDA_VISIBLE_DEVICES=$GPU python scripts/validate/v16_eval.py \
            --ckpts "t6=$CKPT" \
            --out_dir "$RUN_DIR" \
            --gen_length "$GL" \
            --block_length "$BL" \
            --temperature "$TEMPERATURE" \
            > "$LOG" 2>&1 &
        PIDS+=($!); PID_LABELS+=("$CELL")
        g=$(( (g + 1) % EVAL_GPUS ))
        if [[ "${#PIDS[@]}" -ge "$EVAL_GPUS" ]]; then
            wait "${PIDS[0]}" || echo "[T0-ABL] ✗ ${PID_LABELS[0]} FAILED"
            PIDS=("${PIDS[@]:1}"); PID_LABELS=("${PID_LABELS[@]:1}")
        fi
    done
done
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || echo "[T0-ABL] ✗ ${PID_LABELS[$i]} FAILED"
done

[[ "$DRY_RUN" -eq 1 ]] && exit 0

echo
echo "[T0-ABL] aggregating..."
python - <<PY
import json, re
from pathlib import Path
base = Path("$OUT_BASE")
rows = []
for sj in sorted(base.glob(f"*_${TS}/summary.json")):
    v = json.loads(sj.read_text(encoding="utf-8"))
    t6 = next((c for c in v.get("ckpts", []) if c["label"] == "t6"), None)
    if not t6: continue
    m = re.match(r"bl(\d+)_gl(\d+)_", sj.parent.name)
    bl, gl = (int(m[1]), int(m[2])) if m else (None, None)
    rows.append((bl, gl, t6))
rows.sort(key=lambda r: (r[0], r[1]))

lines = [
    "# T=0 inference-knob ablation",
    "",
    "| bl | gl | fail rescued | ok retention | Δ fail | Δ ok | net | FAIL18 | ceiling-5 |",
    "|---|---|---|---|---|---|---|---|---|",
]
for bl, gl, t6 in rows:
    fr = t6["fail_correct"]; okr = t6["ok_correct"]
    d_fail = fr; d_ok = okr - t6["n_ok"]
    lines.append(
        f"| {bl} | {gl} "
        f"| {fr}/{t6['n_fail']} ({t6['fail_pass@1']:.1%}) "
        f"| {okr}/{t6['n_ok']} ({t6['ok_pass@1']:.1%}) "
        f"| +{d_fail} | {d_ok:+d} | {d_fail+d_ok:+d} "
        f"| {t6['fail18_rescued_count']}/18 "
        f"| {t6['ceiling_broken_count']}/5 |"
    )
out = base / f"summary_${TS}.md"
out.write_text("\n".join(lines), encoding="utf-8")
(base / "summary.md").write_text("\n".join(lines), encoding="utf-8")
print(open(out).read())
print(f"\n[T0-ABL] summary → {out}")
PY

echo "[T0-ABL] done. → $OUT_BASE"
