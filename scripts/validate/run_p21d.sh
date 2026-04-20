#!/usr/bin/env bash
# P2.1.d launcher —— GSM8K training-set leakage probe.
#
# 先跑 5 条 broken-by-answer 的定性验证，再全扫 60 条。
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN_DIR="${RUN_DIR:-runs/validation/a5_prompt_template_20260415_191434}"

if [ ! -d "$RUN_DIR/per_prompt" ]; then
    echo "[p21d] RUN_DIR invalid (missing per_prompt/): $RUN_DIR" >&2
    echo "       Override with: RUN_DIR=<path> bash scripts/validate/run_p21d.sh" >&2
    exit 1
fi

echo "========================================================"
echo "[p21d] Phase 1: broken-by-answer 5 条定性验证"
echo "========================================================"
python scripts/validate/p21d_gsm8k_leakage.py \
    --run_dir "$RUN_DIR" \
    --only_idx 2,17,22,24,57 \
    --ngram 8 \
    --topk 3 \
    --min_hits 3 \
    --out "runs/validation/p21d_leakage_broken5.json"

echo
echo "========================================================"
echo "[p21d] Phase 2: 全扫 60 条（base_correct + fail 污染率）"
echo "========================================================"
python scripts/validate/p21d_gsm8k_leakage.py \
    --run_dir "$RUN_DIR" \
    --ngram 8 \
    --topk 3 \
    --min_hits 3 \
    --out "runs/validation/p21d_leakage_full60.json"

echo
echo "[p21d] done."
echo "  - runs/validation/p21d_leakage_broken5.json   (5 条定性)"
echo "  - runs/validation/p21d_leakage_full60.json    (60 条污染率)"
