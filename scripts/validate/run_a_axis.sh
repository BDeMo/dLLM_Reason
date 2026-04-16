#!/usr/bin/env bash
# 顺序跑 A 轴探索实验 A3 → A4 → A5（走 FastAPI server，不直接占 GPU）
#
# 前置：
#   1. scripts/serve.py 已启动在 DLLM_SERVER_URL (默认 http://localhost:8000)
#   2. h0_forensics.py 已跑过，runs/validation/scope_fail_prompts.json 存在
#
# Usage:
#   bash scripts/validate/run_a_axis.sh
#   DLLM_SERVER_URL=http://1.2.3.4:8000 bash scripts/validate/run_a_axis.sh
#   bash scripts/validate/run_a_axis.sh --resume
#   bash scripts/validate/run_a_axis.sh --n 30        # 先跑 30 条试水
#
# 输出：
#   runs/validation/a3_span_revise_<ts>/
#   runs/validation/a4_block_rerank_<ts>/
#   runs/validation/a5_prompt_template_<ts>/
#   docs/archive/hypotheses.md 底部结论板自动回填 A3/A4/A5 行
#
# 估时（llada-instruct, H100/A100, 137 prompts，server-side）：
#   A3: 2 gen/prompt × 137     = 274 gen   ~25min
#   A4: 5 layout/prompt × 137  = 685 gen   ~1h
#   A5: 4 template/prompt × 137 = 548 gen  ~50min
#   合计 ~2.5h
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

SERVER_URL="${DLLM_SERVER_URL:-http://localhost:8000}"
echo "[run_a_axis] server_url = ${SERVER_URL}"

# Quick health probe — if the server isn't up, fail fast.
if ! curl -sf "${SERVER_URL}/info" > /dev/null; then
    echo "[run_a_axis] ERROR: FastAPI server not reachable at ${SERVER_URL}"
    echo "             Start it first:  python scripts/serve.py --model_id <path>"
    exit 1
fi

EXTRA_ARGS="$*"  # 透传 --resume / --n / --run_dir 等

if [ ! -f "runs/validation/scope_fail_prompts.json" ]; then
    echo "[run_a_axis] ERROR: runs/validation/scope_fail_prompts.json 不存在"
    echo "             先跑: python scripts/validate/h0_forensics.py"
    exit 1
fi

echo "════════════════════════════════════════════════════════════"
echo "[run_a_axis] A3: span-level revise (window mean-conf)"
echo "════════════════════════════════════════════════════════════"
python scripts/validate/a3_span_revise.py --n 137 --server_url "${SERVER_URL}" $EXTRA_ARGS

echo ""
echo "════════════════════════════════════════════════════════════"
echo "[run_a_axis] A4: block-layout rerank (bl8/16/32/64 + short_then_long)"
echo "════════════════════════════════════════════════════════════"
python scripts/validate/a4_block_rerank.py --n 137 --server_url "${SERVER_URL}" $EXTRA_ARGS

echo ""
echo "════════════════════════════════════════════════════════════"
echo "[run_a_axis] A5: prompt-template rerank (baseline / cot_plain / cot_step / answer)"
echo "════════════════════════════════════════════════════════════"
python scripts/validate/a5_prompt_template.py --n 137 --server_url "${SERVER_URL}" $EXTRA_ARGS

echo ""
echo "════════════════════════════════════════════════════════════"
echo "[run_a_axis] Aggregate verdicts → docs/archive/hypotheses.{md,zh.md}"
echo "════════════════════════════════════════════════════════════"
python scripts/validate/aggregate_verdicts.py

echo ""
echo "[run_a_axis] Done. A 轴 verdicts:"
grep -E "^\| A[345] " docs/archive/hypotheses.md || true
echo ""
echo "[run_a_axis] Pack & download:"
echo "  tar czf a_axis_results.tar.gz runs/validation/a{3,4,5}_*/ docs/archive/"
