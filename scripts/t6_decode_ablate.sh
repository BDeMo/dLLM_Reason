#!/usr/bin/env bash
# t6_decode_ablate.sh — decoding strategy ablation on a fixed T6 ckpt.
#
# Goal: keep ok_pass@N at 100% while maximising fail_pass@N.
#
# Sweeps h3_passN at multiple (temperature, n_samples) configs on
# FULL scope (n_fail=331 n_ok=988) — not the subsampled 30+30 that
# t6_passN/h3_passN defaults to. Parallelises across GPUs by launching
# one (T, N) cell per GPU.
#
# Why full scope here: subsample 30+30 has ±9% binomial noise per cell,
# hiding small decoding-strategy improvements. Full scope narrows CI
# enough to rank strategies reliably.
#
# Cost budget (8×A100):
#   1319 prompts × N samples × 1 T × ~5s/gen → ~14h for N=8 on 1 GPU
#   3 temps × 1 N parallel on 3 GPUs → ~14h wall (each GPU runs 1 cell)
#   2 N × 3 temps on 6 GPUs → ~14h (N doubling doesn't hurt wall-time
#   since each GPU still runs its one cell serially through all prompts)
#
# Usage:
#   # Default: full-SFT best ckpt (step 336 = 2 epoch), T × N grid
#   bash scripts/t6_decode_ablate.sh \
#     --ckpt runs/training/v161_t6_ablate/hf_step_336
#
#   # LoRA r=1 ep=4 best
#   bash scripts/t6_decode_ablate.sh \
#     --ckpt runs/training/v161_t6_lora_r1/hf_step_672_merged
#
#   # Custom grid
#   bash scripts/t6_decode_ablate.sh --ckpt <path> \
#     --temps 0.3 0.7 --n_samples_list 4 8
#
#   # Subset for smoke test (use hardset from t6_hardset, or tiny)
#   bash scripts/t6_decode_ablate.sh --ckpt <path> --n_fail 50 --n_ok 100

set -euo pipefail

CKPT=""
TEMPS=(0.3 0.7 1.0)
N_SAMPLES_LIST=(8)
GEN_LENGTH=128
BLOCK_LENGTH=32
STEPS_=128
N_FAIL=331                   # FULL scope_fail
N_OK=988                     # FULL scope_ok
EVAL_GPUS=8
GPU_CSV=""
AUTO_GPUS=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)            CKPT="$2"; shift 2 ;;
        --temps)           shift; TEMPS=(); while [[ $# -gt 0 && "$1" != --* ]]; do TEMPS+=("$1"); shift; done ;;
        --n_samples_list)  shift; N_SAMPLES_LIST=(); while [[ $# -gt 0 && "$1" != --* ]]; do N_SAMPLES_LIST+=("$1"); shift; done ;;
        --gen_length)      GEN_LENGTH="$2"; shift 2 ;;
        --block_length)    BLOCK_LENGTH="$2"; shift 2 ;;
        --steps)           STEPS_="$2"; shift 2 ;;
        --n_fail)          N_FAIL="$2"; shift 2 ;;
        --n_ok)            N_OK="$2"; shift 2 ;;
        --eval_gpus)       EVAL_GPUS="$2"; shift 2 ;;
        --gpus)            GPU_CSV="$2"; shift 2 ;;
        --auto_gpus)       AUTO_GPUS=1; shift ;;
        --dry_run)         DRY_RUN=1; shift ;;
        -h|--help)         grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[DEC-ABL] unknown arg: $1" >&2; exit 1 ;;
    esac
done
[[ -z "$CKPT" ]] && { echo "[DEC-ABL] ERROR: --ckpt <path> required" >&2; exit 2; }
[[ ! -f "$CKPT/config.json" ]] && { echo "[DEC-ABL] ERROR: $CKPT/config.json missing" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

CKPT_LABEL=$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")
OUT_BASE="$ROOT/runs/validation/t6_decode_ablate/$CKPT_LABEL"
mkdir -p "$OUT_BASE"
LOG_DIR="$ROOT/runs/ablate_logs"
mkdir -p "$LOG_DIR"

TS=$(date +%Y%m%d_%H%M%S)

echo "[DEC-ABL] ============================================================"
echo "[DEC-ABL]   ckpt        = $CKPT"
echo "[DEC-ABL]   temps       = ${TEMPS[*]}"
echo "[DEC-ABL]   n_samples   = ${N_SAMPLES_LIST[*]}"
echo "[DEC-ABL]   n_fail/ok   = $N_FAIL / $N_OK  (full=331/988)"
echo "[DEC-ABL]   gen/bl/steps= $GEN_LENGTH / $BLOCK_LENGTH / $STEPS_"
# Cap parallelism to actual cell count — no point reserving 8 GPUs for
# 3 cells (wastes auto_gpus picks that other work could use).
N_CELLS=$(( ${#TEMPS[@]} * ${#N_SAMPLES_LIST[@]} ))
if [[ "$EVAL_GPUS" -gt "$N_CELLS" ]]; then
    echo "[DEC-ABL]   cap EVAL_GPUS $EVAL_GPUS → $N_CELLS (n_cells = $N_CELLS)"
    EVAL_GPUS=$N_CELLS
fi
echo "[DEC-ABL]   eval_gpus   = $EVAL_GPUS  (for $N_CELLS cells)"

# Resolve GPU indices
if [[ -n "$GPU_CSV" ]]; then
    IFS=',' read -r -a GPUS_ARR <<< "$GPU_CSV"
    # explicit list — user chose; don't cap beyond cell count but warn
    if [[ "${#GPUS_ARR[@]}" -gt "$N_CELLS" ]]; then
        GPUS_ARR=("${GPUS_ARR[@]:0:$N_CELLS}")
        echo "[DEC-ABL]   trimmed --gpus to first $N_CELLS (matches cells)"
    fi
    EVAL_GPUS="${#GPUS_ARR[@]}"
elif [[ "$AUTO_GPUS" -eq 1 ]]; then
    source "$SCRIPT_DIR/_select_gpus.sh"
    SEL=$(select_free_gpus "$EVAL_GPUS")
    IFS=',' read -r -a GPUS_ARR <<< "$SEL"
    echo "[DEC-ABL]   auto-GPUs   = $SEL"
else
    GPUS_ARR=(); for i in $(seq 0 $((EVAL_GPUS - 1))); do GPUS_ARR+=("$i"); done
fi

echo "[DEC-ABL]   out base    = $OUT_BASE"
echo "[DEC-ABL] ============================================================"

# ── launch one (T, N) cell per GPU, throttle at EVAL_GPUS ───────────────
declare -a PIDS=()
declare -a PID_LABELS=()
g=0
for N in "${N_SAMPLES_LIST[@]}"; do
    for T in "${TEMPS[@]}"; do
        CELL="T${T}_N${N}"
        RUN_DIR="$OUT_BASE/${CELL}_${TS}"
        LOG="$LOG_DIR/dec_ablate_${CKPT_LABEL}_${CELL}_${TS}.log"

        GPU="${GPUS_ARR[$g]}"
        echo "[DEC-ABL]   launch T=$T N=$N on GPU $GPU (slot $g) → $RUN_DIR"
        [[ "$DRY_RUN" -eq 1 ]] && continue

        CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python -u \
            scripts/validate/h3_passN_at_temperature.py \
            --model "$CKPT" \
            --run_dir "$RUN_DIR" \
            --n_samples "$N" \
            --gen_length "$GEN_LENGTH" \
            --block_length "$BLOCK_LENGTH" \
            --steps "$STEPS_" \
            --temps "$T" \
            --n_fail "$N_FAIL" --n_ok "$N_OK" \
            > "$LOG" 2>&1 &
        PIDS+=($!)
        PID_LABELS+=("$CELL")
        g=$(( (g + 1) % EVAL_GPUS ))
        if [[ "${#PIDS[@]}" -ge "$EVAL_GPUS" ]]; then
            wait "${PIDS[0]}" || echo "[DEC-ABL] ✗ ${PID_LABELS[0]} FAILED"
            PIDS=("${PIDS[@]:1}")
            PID_LABELS=("${PID_LABELS[@]:1}")
        fi
    done
done
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || echo "[DEC-ABL] ✗ ${PID_LABELS[$i]} FAILED"
done

[[ "$DRY_RUN" -eq 1 ]] && { echo "[DEC-ABL] dry-run done"; exit 0; }

# ── aggregate Pareto: ok_pass@k must be 100% AND fail_pass@k max ────────
echo
echo "[DEC-ABL] aggregating Pareto table..."
python - <<PY
import json
from pathlib import Path

base = Path("$OUT_BASE")
rows = []
for sj in sorted(base.glob(f"*_${TS}/summary.json")):
    v = json.loads(sj.read_text(encoding="utf-8"))
    fs = v.get("fail_stats", {})
    os_ = v.get("ok_stats", {})
    # each cell was run with a single T, so single key
    for T in fs:
        f = fs[T]; o = os_.get(T, {})
        rows.append({
            "cell": sj.parent.name,
            "T": float(T),
            "n_samples": v["config"].get("n_samples"),
            "n_fail": v["n_fail"], "n_ok": v["n_ok"],
            "fail@1": f.get("pass@1"), "fail@4": f.get("pass@4"), "fail@N": f.get("pass@8"),
            "ok@1":   o.get("pass@1"), "ok@4":   o.get("pass@4"), "ok@N":   o.get("pass@8"),
        })
rows.sort(key=lambda r: (r["T"], r["n_samples"]))

def pct(x):
    try: return f"{float(x):.1%}"
    except: return "—"

lines = [
    f"# Decoding strategy ablation — {Path('$CKPT').name}",
    "",
    f"Full scope: n_fail={rows[0]['n_fail'] if rows else '?'}, "
    f"n_ok={rows[0]['n_ok'] if rows else '?'}.",
    "",
    "| T | N | fail@1 | fail@4 | fail@N | ok@1 | ok@4 | **ok@N** | ok=100%? |",
    "|---|---|---|---|---|---|---|---|---|",
]
for r in rows:
    ok_n = r["ok@N"] or 0
    mark = "✅" if ok_n >= 0.999 else ("🟡" if ok_n >= 0.99 else "❌")
    lines.append(
        f"| {r['T']} | {r['n_samples']} "
        f"| {pct(r['fail@1'])} | {pct(r['fail@4'])} | {pct(r['fail@N'])} "
        f"| {pct(r['ok@1'])} | {pct(r['ok@4'])} | **{pct(r['ok@N'])}** "
        f"| {mark} |"
    )

# Pick best-under-constraint: ok_pass@N ≥ 99.9% (allow rounding), max fail@N
qualifying = [r for r in rows if (r["ok@N"] or 0) >= 0.999]
lines += ["", "## Pareto pick", ""]
if qualifying:
    best = max(qualifying, key=lambda r: r["fail@N"] or 0)
    lines.append(
        f"- **Best at ok=100%**: T={best['T']}, N={best['n_samples']} "
        f"→ fail@N = {pct(best['fail@N'])}, "
        f"(vs baseline fail=0% canonical T=0 p@1)"
    )
else:
    best_any = max(rows, key=lambda r: (r["ok@N"] or 0, r["fail@N"] or 0))
    lines.append(
        f"- No cell hit ok=100% — closest: T={best_any['T']} N={best_any['n_samples']} "
        f"ok@N={pct(best_any['ok@N'])}, fail@N={pct(best_any['fail@N'])}"
    )

out = base / f"summary_${TS}.md"
out.write_text("\n".join(lines), encoding="utf-8")
(base / "summary.md").write_text("\n".join(lines), encoding="utf-8")
print(open(out).read())
print(f"\n[DEC-ABL] summary → {out}")
PY

echo "[DEC-ABL] done. → $OUT_BASE"
