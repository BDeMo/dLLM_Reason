#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  运行所有 training 模块的集成测试
#
#  Usage:
#    bash tests/scripts/run_all_training_tests.sh
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PASSED=0
FAILED=0

run_test() {
    local name="$1"
    local script="$2"
    shift 2
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if bash "$SCRIPT_DIR/$script" "$@"; then
        PASSED=$((PASSED + 1))
    else
        echo "❌ FAILED: $name"
        FAILED=$((FAILED + 1))
    fi
}

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║        dLLM-Reason Training Modules — Full Test Suite    ║"
echo "╚═══════════════════════════════════════════════════════════╝"

# 1. pytest 单元测试
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Running: pytest unit tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if python -m pytest tests/test_training_modules.py -v --tb=short; then
    PASSED=$((PASSED + 1))
else
    echo "❌ FAILED: pytest"
    FAILED=$((FAILED + 1))
fi

# 2. PUMA progressive masking
run_test "PUMA Progressive Masking" test_progressive_train.sh

# 3. Supervised Planner (ranking)
run_test "Supervised Planner (ranking)" test_supervised_planner.sh ranking

# 4. Supervised Planner (regression)
run_test "Supervised Planner (regression)" test_supervised_planner.sh regression

# 5. KL-regularised RL (all modes)
run_test "KL-Regularised RL" test_kl_regularised_rl.sh all

# ── Summary ──────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Results: $PASSED passed, $FAILED failed"
echo "═══════════════════════════════════════════════════════════"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
echo "✅ All training module tests passed!"
