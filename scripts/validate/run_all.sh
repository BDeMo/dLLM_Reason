#!/usr/bin/env bash
# 一键全跑 H0/H1/H2/H3 验证套件（服务器 GPU 环境）
#
# Usage:
#   bash scripts/validate/run_all.sh
#
# 输出：
#   runs/validation/scope_fail_prompts.json
#   runs/validation/h1_remask_<ts>/{config,per_prompt/,progress.jsonl,summary}.json
#   runs/validation/h2_order_content_<ts>/...
#   runs/validation/h3_passN_<ts>/...
#   docs/archive/hypotheses.md 底部结论板自动更新
#
# 断点 resume：重跑同一条命令加 --resume --run_dir <path>
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

echo "════════════════════════════════════════════════════════════"
echo "[run_all] H0: scope generation (no GPU)"
echo "════════════════════════════════════════════════════════════"
python scripts/validate/h0_forensics.py

echo ""
echo "════════════════════════════════════════════════════════════"
echo "[run_all] H1: remask rescue (137 fail prompts, 2 generations each)"
echo "════════════════════════════════════════════════════════════"
python scripts/validate/h1_remask_rescue.py --n 137

echo ""
echo "════════════════════════════════════════════════════════════"
echo "[run_all] H3: pass@N at temperature (fail 30 + ok 30, × 3T × 8 samples)"
echo "════════════════════════════════════════════════════════════"
python scripts/validate/h3_passN_at_temperature.py --n_fail 30 --n_ok 30 --n_samples 8

echo ""
echo "════════════════════════════════════════════════════════════"
echo "[run_all] H2: order vs content variance (20 prompts)"
echo "════════════════════════════════════════════════════════════"
python scripts/validate/h2_order_vs_content.py --n 20

echo ""
echo "════════════════════════════════════════════════════════════"
echo "[run_all] Aggregate verdicts → docs/archive/hypotheses.md"
echo "════════════════════════════════════════════════════════════"
python scripts/validate/aggregate_verdicts.py

echo ""
echo "[run_all] Done. Pack & download:"
echo "  tar czf validation_results.tar.gz runs/validation/ docs/archive/hypotheses.md"
