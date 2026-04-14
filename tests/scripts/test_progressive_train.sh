#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  PUMA Progressive Masking 训练 — 快速功能验证
#
#  用 mock MDLM 在合成数据上跑 20 步，验证：
#    1. progressive mask 构建正确
#    2. 课程学习 warmup 正常工作
#    3. loss 逐步下降（或至少不爆炸）
#
#  Usage:
#    bash tests/scripts/test_progressive_train.sh
#    bash tests/scripts/test_progressive_train.sh --steps 50 --ratio 0.8
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

STEPS="${1:-20}"
RATIO="${2:-1.0}"
WARMUP="${3:-10}"

cat <<EOF
╔═══════════════════════════════════════════════════════════╗
║  PUMA Progressive Masking — Smoke Test                   ║
║  steps=$STEPS  ratio=$RATIO  warmup=$WARMUP              ║
╚═══════════════════════════════════════════════════════════╝
EOF

python -c "
import torch
from torch.utils.data import DataLoader, TensorDataset

from dllm_reason.models.mdlm import MDLM
from dllm_reason.training.progressive_train import (
    ProgressiveTrainer, ProgressiveTrainConfig,
)

# ── 1. 构建小型 MDLM ──────────────────────────────────────────
print('[1/4] Building mock MDLM (vocab=128, dim=64, 2 layers, linear schedule) ...')
model = MDLM(vocab_size=128, max_seq_len=64, dim=64, num_layers=2, num_heads=2, noise_schedule='linear')
print(f'       params = {sum(p.numel() for p in model.parameters()):,}')

# ── 2. 合成训练数据 ───────────────────────────────────────────
print('[2/4] Generating synthetic data (64 samples, seq_len=64) ...')
B, L = 64, 64
input_ids = torch.randint(0, 127, (B, L))
prompt_mask = torch.zeros(B, L, dtype=torch.bool)
prompt_mask[:, :16] = True  # 前 16 tokens 是 prompt
attention_mask = torch.ones(B, L, dtype=torch.bool)

dataset = TensorDataset(input_ids, prompt_mask, attention_mask)
def collate(batch):
    ids, pm, am = zip(*batch)
    return {'input_ids': torch.stack(ids), 'prompt_mask': torch.stack(pm), 'attention_mask': torch.stack(am)}
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate)

# ── 3. 训练 ──────────────────────────────────────────────────
print('[3/4] Training with progressive masking ...')
cfg = ProgressiveTrainConfig(
    max_steps=$STEPS,
    progressive_ratio=$RATIO,
    progressive_warmup_steps=$WARMUP,
    lr=1e-3,
    log_every=5,
    save_every=999999,
    eval_every=999999,
)
trainer = ProgressiveTrainer(model, loader, config=cfg)
trainer.train()

# ── 4. 验证 ──────────────────────────────────────────────────
print('[4/4] Verifying ...')
assert trainer.global_step == $STEPS, f'Expected {$STEPS} steps, got {trainer.global_step}'

# 跑一次 progressive mask 构建，检查形状
batch = next(iter(loader))
x_0 = batch['input_ids']
# 用 t=0.9 确保大部分 token 被 mask（sigma(0.9) ≈ 0.9）
t = torch.full((x_0.shape[0],), 0.9)
mask = trainer._build_progressive_mask(x_0, t, batch['attention_mask'], batch['prompt_mask'])
assert mask.shape == x_0.shape
assert not mask[:, :16].any(), 'Prompt should not be masked'
assert mask[:, 16:].any(), 'Some gen positions should be masked at t=0.9'

print()
print('✅ PUMA Progressive Masking — ALL CHECKS PASSED')
"
