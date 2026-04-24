#!/usr/bin/env bash
# t6_cleanup.sh — reclaim disk from LoRA ablation intermediate artefacts.
#
# After t6_lora_ablate finishes, runs/training/v161_t6_lora_r*/ contains:
#   hf_step_<S>/            adapter only (~30MB)     KEEP
#   hf_step_<S>_merged/     merged HF ckpt (~16GB)   DELETE after eval
#   step_*.pt               optimizer+model ckpt     DELETE (best.pt OK)
#   hf/                     final merged ckpt        DELETE (can re-merge)
#
# Eval outputs in runs/validation/t6_lora_ablate/r*_step*/ stay.
# Adapter + summary.json are sufficient to re-eval any cell.
#
# Usage:
#   bash scripts/t6_cleanup.sh --dry_run    # see what would be deleted
#   bash scripts/t6_cleanup.sh              # actually delete
#   bash scripts/t6_cleanup.sh --keep_best  # also keep best.pt per rank

set -euo pipefail

DRY_RUN=0
KEEP_BEST=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry_run)   DRY_RUN=1; shift ;;
        --keep_best) KEEP_BEST=1; shift ;;
        -h|--help)   grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

total=0
do_rm() {
    local p="$1"
    if [[ ! -e "$p" ]]; then return; fi
    local sz
    sz=$(du -sb "$p" 2>/dev/null | awk '{print $1}')
    total=$((total + sz))
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  [dry] rm -rf  $(du -sh "$p" | cut -f1)  $p"
    else
        rm -rf "$p"
        echo "  rm'd          $p"
    fi
}

echo "[CLEANUP] scanning runs/training/v161_t6_lora_r*/ ..."
for d in "$ROOT"/runs/training/v161_t6_lora_r*; do
    [[ -d "$d" ]] || continue
    echo "[CLEANUP] $(basename "$d"):"
    # merged intermediate ckpts
    for md in "$d"/hf_step_*_merged; do
        [[ -d "$md" ]] && do_rm "$md"
    done
    # final merged (can be rebuilt from final adapter + base)
    [[ -d "$d/hf" ]] && do_rm "$d/hf"
    # optimizer ckpt .pt files
    for pt in "$d"/step_*.pt "$d"/best.pt; do
        [[ ! -f "$pt" ]] && continue
        if [[ "$KEEP_BEST" -eq 1 && "$(basename "$pt")" == "best.pt" ]]; then
            echo "  keep   $(du -sh "$pt" | cut -f1)  $pt"
            continue
        fi
        do_rm "$pt"
    done
done

echo "[CLEANUP] scanning runs/training/v161_t6_ablate/ ..."
d="$ROOT/runs/training/v161_t6_ablate"
if [[ -d "$d" ]]; then
    for pt in "$d"/step_*.pt "$d"/best.pt; do
        [[ ! -f "$pt" ]] && continue
        if [[ "$KEEP_BEST" -eq 1 && "$(basename "$pt")" == "best.pt" ]]; then continue; fi
        do_rm "$pt"
    done
fi

human=$(python -c "
n = $total
for u in ['B','KB','MB','GB','TB']:
    if n < 1024: print(f'{n:.1f}{u}'); break
    n /= 1024
")
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[CLEANUP] would free: $human"
else
    echo "[CLEANUP] freed: $human"
fi
