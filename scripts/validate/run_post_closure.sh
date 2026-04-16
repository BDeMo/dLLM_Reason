#!/usr/bin/env bash
# Post-closure 一键全跑 —— P0 + P3 + P7 服务器端，加 P4/P5/P6 离线分析
#
# 顺序:
#   1. A3/A4/A5 resume 到 N=137  (P0)
#   2. A6 gen_length 新 run       (P7)
#   3. A4x5 6-cell joint 新 run   (P3)
#   4. aggregate verdicts
#   5. a4x5_overlap (用最新 A4/A5 run)
#   6. P4 cot-broken 特征分析    (用最新 A5 run)
#   7. P6 A4-rescue 特征分析     (用最新 A4/A5 run)
#   8. P5 H3 cross-ref           (如果 H3 run 存在)
#
# 用法:
#   LLADA_URL=http://localhost:8000 N=137 bash scripts/validate/run_post_closure.sh
#
# 环境变量:
#   LLADA_URL  server 地址 (默认 http://localhost:8000)
#   N          fail prompt 数量 (默认 137)
#   SKIP_P0    设为 1 跳过 A3/A4/A5 resume
#   SKIP_A6    设为 1 跳过 A6
#   SKIP_JOINT 设为 1 跳过 A4x5 joint

set -euo pipefail

cd "$(dirname "$0")/../.."

SERVER="${LLADA_URL:-http://localhost:8000}"
N="${N:-137}"
RUNS="runs/validation"

echo "=========================================================="
echo "[post-closure] server = $SERVER · N = $N"
echo "[post-closure] start  = $(date -Iseconds)"
echo "=========================================================="
echo

# helper: 取某 prefix 最新的 run dir
latest_run() {
    ls -1d "$RUNS"/$1 2>/dev/null | sort | tail -1
}

# ---------- Phase 1: 服务器端 (P0) ----------
if [ "${SKIP_P0:-0}" != "1" ]; then
    echo "## [P0] A3 → N=$N (resume)"
    python scripts/validate/a3_span_revise.py --n "$N" --resume \
        --server_url "$SERVER"
    echo

    echo "## [P0] A4 → N=$N (resume)"
    python scripts/validate/a4_block_rerank.py --n "$N" --resume \
        --server_url "$SERVER"
    echo

    echo "## [P0] A5 → N=$N (resume)"
    python scripts/validate/a5_prompt_template.py --n "$N" --resume \
        --server_url "$SERVER"
    echo
else
    echo "[post-closure] SKIP_P0=1 跳过 A3/A4/A5 resume"
fi

# ---------- Phase 2: 新 run (P7 + P3) ----------
if [ "${SKIP_A6:-0}" != "1" ]; then
    echo "## [P7] A6 gen_length sweep → N=$N (新 run)"
    python scripts/validate/a6_gen_length.py --n "$N" \
        --server_url "$SERVER"
    echo
else
    echo "[post-closure] SKIP_A6=1 跳过 A6"
fi

if [ "${SKIP_JOINT:-0}" != "1" ]; then
    echo "## [P3] A4 × A5 joint 6-cell → N=$N (新 run)"
    python scripts/validate/a4x5_joint.py --n "$N" \
        --server_url "$SERVER"
    echo
else
    echo "[post-closure] SKIP_JOINT=1 跳过 A4x5 joint"
fi

# ---------- Phase 3: 汇总 + 离线分析 ----------
echo "## [aggregate] 汇总 verdict 表"
python scripts/validate/aggregate_verdicts.py || true
echo

A4_RUN=$(latest_run "a4_block_rerank_*")
A5_RUN=$(latest_run "a5_prompt_template_*")
A6_RUN=$(latest_run "a6_gen_length_*")
JOINT_RUN=$(latest_run "a4x5_joint_*")
H3_RUN=$(latest_run "h3_passN_*")

echo "[post-closure] 最新 run dirs:"
echo "   A4    = ${A4_RUN:-<none>}"
echo "   A5    = ${A5_RUN:-<none>}"
echo "   A6    = ${A6_RUN:-<none>}"
echo "   joint = ${JOINT_RUN:-<none>}"
echo "   H3    = ${H3_RUN:-<none>}"
echo

if [ -n "$A4_RUN" ] && [ -n "$A5_RUN" ]; then
    echo "## [overlap] A4 × A5 overlap (覆盖最新 run)"
    python scripts/validate/a4x5_overlap.py \
        --a4_run "$A4_RUN" \
        --a5_run "$A5_RUN" \
        --out "$RUNS/a4x5_overlap_$(basename $A4_RUN | sed 's/a4_block_rerank_//')_$(basename $A5_RUN | sed 's/a5_prompt_template_//').json" \
        || echo "[post-closure] overlap 失败（不致命，继续）"
    echo

    echo "## [P4] cot-broken pattern 分析"
    python scripts/validate/p4_cot_broken_pattern.py --a5_run "$A5_RUN" \
        || echo "[post-closure] P4 失败（不致命）"
    echo

    echo "## [P6] A4-rescue features 分析"
    python scripts/validate/p6_a4_rescue_features.py \
        --a4_run "$A4_RUN" --a5_run "$A5_RUN" \
        || echo "[post-closure] P6 失败（不致命）"
    echo
else
    echo "[post-closure] A4 或 A5 run 缺失，跳过 overlap / P4 / P6"
fi

if [ -n "$H3_RUN" ] && [ -n "$A4_RUN" ] && [ -n "$A5_RUN" ]; then
    echo "## [P5] H3 × (A4 ∪ A5) cross-ref"
    python scripts/validate/p5_h3_crossref.py \
        --h3_run "$H3_RUN" \
        --a4_run "$A4_RUN" \
        --a5_run "$A5_RUN" \
        || echo "[post-closure] P5 失败（不致命）"
    echo
else
    echo "[post-closure] H3 run 不存在，跳过 P5（H3 跑完后再单独跑一次）"
fi

echo "=========================================================="
echo "[post-closure] done = $(date -Iseconds)"
echo "=========================================================="
echo
echo "下一步本机检视:"
echo "  - $RUNS/a4_block_rerank_*/summary.json"
echo "  - $RUNS/a5_prompt_template_*/summary.json"
echo "  - $RUNS/a6_gen_length_*/summary.json"
echo "  - $RUNS/a4x5_joint_*/summary.json"
echo "  - $RUNS/a4x5_overlap_*.json"
echo "  - $RUNS/p4_cot_broken_pattern_*.json"
echo "  - $RUNS/p6_a4_rescue_features.json"
echo "  - docs/archive/hypotheses.zh.md (verdict 表已更新)"
