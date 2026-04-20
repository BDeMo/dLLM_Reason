#!/usr/bin/env bash
# P2.1.e launcher —— dump A5 full outputs for broken-by-answer 5 条.
#
# 要求 LLaDA server 在 ${LLADA_URL:-http://localhost:8000} 跑着。
set -euo pipefail

cd "$(dirname "$0")/../.."

SERVER="${LLADA_URL:-http://localhost:8000}"

echo "[p21e] server = $SERVER"

echo "========================================================"
echo "[p21e] Phase 1: 原 A5 参数 (gen_length=128) —— 复现 A5 run 的输出"
echo "========================================================"
python scripts/validate/p21e_dump_full.py \
    --only_idx 2,17,22,24,57 \
    --gen_length 128 \
    --steps 128 \
    --block_length 32 \
    --server_url "$SERVER" \
    --out "runs/validation/p21e_full_broken5_g128.json"

echo
echo "========================================================"
echo "[p21e] Phase 2: gen_length=256 —— 看完整输出是否被截断导致 tail 错位"
echo "========================================================"
python scripts/validate/p21e_dump_full.py \
    --only_idx 2,17,22,24,57 \
    --gen_length 256 \
    --steps 256 \
    --block_length 32 \
    --server_url "$SERVER" \
    --out "runs/validation/p21e_full_broken5_g256.json"

echo
echo "[p21e] done. 两个 JSON 给 claude 读:"
echo "  - runs/validation/p21e_full_broken5_g128.json"
echo "  - runs/validation/p21e_full_broken5_g256.json"
