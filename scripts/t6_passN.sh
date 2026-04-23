#!/usr/bin/env bash
# t6_passN.sh — pass@N eval on a T6-trained ckpt under T>0 sampling.
#
# Complements the canonical T=0 pass@1 ablation (scripts/t6_ablate.sh):
# measures the diversity/capacity frontier of the trained model under
# stochastic decoding. Uses scripts/validate/h3_passN_at_temperature.py
# which subsamples scope_fail / scope_ok and reports pass@k for k≤N at
# each temperature.
#
# Usage:
#   # Auto-pick the best ckpt from t6_ablate: pass --auto
#   bash scripts/t6_passN.sh --auto
#
#   # Specific ckpt (any HF-format dir with config.json):
#   bash scripts/t6_passN.sh --ckpt runs/training/v161_t6_ablate/hf_step_169
#
#   # Override N / temps / scope subset size:
#   bash scripts/t6_passN.sh --ckpt <path> --n_samples 16 --temps 0.3 0.7 1.0
#   bash scripts/t6_passN.sh --ckpt <path> --n_fail 60 --n_ok 60
#
#   # Sweep ALL checkpoints in a t6_ablate training run:
#   bash scripts/t6_passN.sh --sweep runs/training/v161_t6_ablate
#
#   # Plan only:
#   bash scripts/t6_passN.sh --auto --dry_run

set -euo pipefail

CKPT=""
AUTO=0
SWEEP_DIR=""
N_SAMPLES=8
GEN_LENGTH=128
BLOCK_LENGTH=32
STEPS_=128                      # MDLM steps (= gen_length by canonical)
TEMPS=(0.3 0.7 1.0)
N_FAIL=30
N_OK=30
EVAL_GPUS=8                     # parallel ckpts across this many GPUs
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)         CKPT="$2"; shift 2 ;;
        --auto)         AUTO=1; shift ;;
        --sweep)        SWEEP_DIR="$2"; shift 2 ;;
        --n_samples)    N_SAMPLES="$2"; shift 2 ;;
        --gen_length)   GEN_LENGTH="$2"; shift 2 ;;
        --block_length) BLOCK_LENGTH="$2"; shift 2 ;;
        --steps)        STEPS_="$2"; shift 2 ;;
        --temps)        shift; TEMPS=(); while [[ $# -gt 0 && "$1" != --* ]]; do TEMPS+=("$1"); shift; done ;;
        --n_fail)       N_FAIL="$2"; shift 2 ;;
        --n_ok)         N_OK="$2"; shift 2 ;;
        --eval_gpus)    EVAL_GPUS="$2"; shift 2 ;;
        --dry_run)      DRY_RUN=1; shift ;;
        -h|--help)      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[PASSN] unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Resolve ckpts to eval
CKPT_LIST=()
if [[ -n "$SWEEP_DIR" ]]; then
    for d in "$SWEEP_DIR"/hf_step_*; do
        [[ -f "$d/config.json" ]] && CKPT_LIST+=("$d")
    done
elif [[ "$AUTO" -eq 1 ]]; then
    # auto = latest t6_ablate training, pick whichever step had the best
    # net score from the most recent summary.md. If no summary, default to
    # the final hf/ ckpt.
    LATEST_TRAIN=$(ls -dt "$ROOT"/runs/training/v161_t6_ablate* 2>/dev/null | head -1)
    [[ -z "$LATEST_TRAIN" ]] && { echo "[PASSN] --auto: no v161_t6_ablate dir"; exit 1; }
    SUMMARY="$ROOT/runs/validation/t6_ablate/summary.md"
    if [[ -f "$SUMMARY" ]]; then
        BEST_STEP=$(python - <<PY
import re
from pathlib import Path
best_step = None; best_net = -10**9
for line in open("$SUMMARY"):
    m = re.match(r"\| (\d+) \|.*\| ([+-]?\d+) \| ([+-]?\d+) \| ([+-]?\d+) \|", line)
    if m:
        step, df, do, net = int(m[1]), int(m[2]), int(m[3]), int(m[4])
        if net > best_net:
            best_net, best_step = net, step
print(best_step or "")
PY
)
        if [[ -n "$BEST_STEP" ]]; then
            CKPT="$LATEST_TRAIN/hf_step_${BEST_STEP}"
            echo "[PASSN] auto-picked best-net ckpt: step=$BEST_STEP → $CKPT"
        fi
    fi
    [[ -z "$CKPT" ]] && CKPT="$LATEST_TRAIN/hf"
    CKPT_LIST+=("$CKPT")
elif [[ -n "$CKPT" ]]; then
    CKPT_LIST+=("$CKPT")
else
    echo "[PASSN] ERROR: pass --ckpt <path> | --auto | --sweep <train_dir>" >&2
    exit 2
fi

OUT_BASE="$ROOT/runs/validation/t6_passN"
mkdir -p "$OUT_BASE"
TS=$(date +%Y%m%d_%H%M%S)

echo "[PASSN] ============================================================"
echo "[PASSN]   ckpts to eval : ${#CKPT_LIST[@]}"
for c in "${CKPT_LIST[@]}"; do echo "[PASSN]     $c"; done
echo "[PASSN]   N_SAMPLES     = $N_SAMPLES"
echo "[PASSN]   TEMPS         = ${TEMPS[*]}"
echo "[PASSN]   n_fail / n_ok = $N_FAIL / $N_OK"
echo "[PASSN]   gen/block/steps= $GEN_LENGTH / $BLOCK_LENGTH / $STEPS_"
echo "[PASSN] ============================================================"

echo "[PASSN]   EVAL_GPUS      = $EVAL_GPUS (ckpts run in parallel)"
echo

declare -a PIDS=()
declare -a PID_LABELS=()
g=0
for CK in "${CKPT_LIST[@]}"; do
    [[ ! -f "$CK/config.json" ]] && { echo "[PASSN] skip (no config.json): $CK"; continue; }
    LABEL=$(basename "$CK")
    PARENT=$(basename "$(dirname "$CK")")
    RUN_DIR="$OUT_BASE/${PARENT}_${LABEL}_${TS}"
    LOG="$OUT_BASE/${PARENT}_${LABEL}_${TS}.log"

    echo "[PASSN] launching on GPU $g: ckpt=$CK → $RUN_DIR"
    [[ "$DRY_RUN" -eq 1 ]] && { echo "[PASSN]   (dry-run, skip)"; continue; }

    CUDA_VISIBLE_DEVICES=$g PYTHONUNBUFFERED=1 python -u \
        scripts/validate/h3_passN_at_temperature.py \
        --model "$CK" \
        --run_dir "$RUN_DIR" \
        --n_samples "$N_SAMPLES" \
        --gen_length "$GEN_LENGTH" \
        --block_length "$BLOCK_LENGTH" \
        --steps "$STEPS_" \
        --temps "${TEMPS[@]}" \
        --n_fail "$N_FAIL" \
        --n_ok "$N_OK" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
    PID_LABELS+=("${PARENT}/${LABEL}")
    g=$(( (g + 1) % EVAL_GPUS ))
    if [[ "${#PIDS[@]}" -ge "$EVAL_GPUS" ]]; then
        wait "${PIDS[0]}" || echo "[PASSN] ✗ ${PID_LABELS[0]} FAILED"
        PIDS=("${PIDS[@]:1}")
        PID_LABELS=("${PID_LABELS[@]:1}")
    fi
done
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || echo "[PASSN] ✗ ${PID_LABELS[$i]} FAILED"
done

# ── aggregate summary ─────────────────────────────────────────────────────
if [[ "$DRY_RUN" -eq 1 ]]; then exit 0; fi

echo
echo "[PASSN] aggregating summary..."
python - <<PY
import json
from pathlib import Path

base = Path("$OUT_BASE")
rows = []
# find run_dirs created this TS only (avoid mixing with prior runs)
for rd in sorted(base.glob(f"*_${TS}")):
    vj = rd / "verdict.json"
    if not vj.exists(): continue
    v = json.load(open(vj))
    # h3_passN verdict.json structure (see compute_verdict in
    # h3_passN_at_temperature.py:106-114):
    #   {"fail_stats": {temp_str: {pass@1, pass@4, pass@8, n}},
    #    "ok_stats":   {temp_str: {pass@1, pass@4, pass@8, n}},
    #    "fail_pass@8_max":..., "ok_pass@8_max":..., "verdict":...}
    rows.append((rd.name, v))

if not rows:
    print("[PASSN] no verdict.json produced — check logs.")
    raise SystemExit(0)

# temps from first row (all runs share temps within one invocation)
any_v = rows[0][1]
temps = sorted(any_v.get("fail_stats", {}).keys(), key=float)

def fmt_pct(x):
    try: return f"{float(x):.1%}"
    except (TypeError, ValueError): return "—"

lines = [
    "# T6 pass@N eval",
    "",
    f"N_samples={int('$N_SAMPLES')}  n_fail={int('$N_FAIL')}  n_ok={int('$N_OK')}",
    "",
    "h3_passN reports pass@k for k ∈ {1, 4, 8}. pass@8 means pass@N (= n_samples) — "
    "the column name is hard-coded in upstream regardless of N.",
    "",
    "| ckpt | temp | fail p@1 | fail p@4 | fail p@8 | ok p@1 | ok p@4 | ok p@8 |",
    "|---|---|---|---|---|---|---|---|",
]
for name, v in rows:
    fs = v.get("fail_stats", {})
    os_ = v.get("ok_stats", {})
    for T in temps:
        f = fs.get(T, {}) or fs.get(str(T), {})
        o = os_.get(T, {}) or os_.get(str(T), {})
        lines.append(
            f"| {name} | {T} "
            f"| {fmt_pct(f.get('pass@1'))} "
            f"| {fmt_pct(f.get('pass@4'))} "
            f"| {fmt_pct(f.get('pass@8'))} "
            f"| {fmt_pct(o.get('pass@1'))} "
            f"| {fmt_pct(o.get('pass@4'))} "
            f"| {fmt_pct(o.get('pass@8'))} |"
        )
    lines.append(f"| {name} | (max) | "
                 f"— | — | {fmt_pct(v.get('fail_pass@8_max'))} "
                 f"| — | — | {fmt_pct(v.get('ok_pass@8_max'))} |")

out = base / f"summary_${TS}.md"
out.write_text("\n".join(lines), encoding="utf-8")
(base / "summary.md").write_text("\n".join(lines), encoding="utf-8")
print(open(out).read())
print(f"\n[PASSN] summary → {out}")
PY

echo "[PASSN] done. → $OUT_BASE"
