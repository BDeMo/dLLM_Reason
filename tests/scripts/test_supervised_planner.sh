#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  Supervised Planner (Oracle Distillation) — 快速功能验证
#
#  用 mock MDLM 验证完整流程：
#    1. collect_oracle_order — oracle 顺序收集
#    2. SupervisedPlannerTrainer.train() — 监督训练
#    3. PlannerScheduler — planner 驱动的 unmasking
#    4. save/load planner — 权重持久化
#
#  Usage:
#    bash tests/scripts/test_supervised_planner.sh
#    bash tests/scripts/test_supervised_planner.sh ranking    # loss type
#    bash tests/scripts/test_supervised_planner.sh regression
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

LOSS_TYPE="${1:-ranking}"

cat <<EOF
╔═══════════════════════════════════════════════════════════╗
║  Supervised Planner (Oracle Distillation) — Smoke Test   ║
║  loss_type=$LOSS_TYPE                                     ║
╚═══════════════════════════════════════════════════════════╝
EOF

python -c "
import os, tempfile
import torch
from torch.utils.data import DataLoader, TensorDataset

from dllm_reason.models.mdlm import MDLM
from dllm_reason.training.supervised_planner import (
    collect_oracle_order, collect_oracle_dataset,
    OracleConfig, SupervisedPlannerTrainer, SupervisedPlannerConfig,
    PlannerScheduler,
)

# ── 1. 构建小型 MDLM ──────────────────────────────────────────
print('[1/6] Building mock MDLM (linear schedule) ...')
model = MDLM(vocab_size=128, max_seq_len=64, dim=64, num_layers=2, num_heads=2, noise_schedule='linear')

# ── 2. 合成数据 ──────────────────────────────────────────────
print('[2/6] Generating synthetic data (32 samples) ...')
B, L = 32, 64
input_ids = torch.randint(0, 127, (B, L))
prompt_mask = torch.zeros(B, L, dtype=torch.bool)
prompt_mask[:, :16] = True
attention_mask = torch.ones(B, L, dtype=torch.bool)
dataset = TensorDataset(input_ids, prompt_mask, attention_mask)
def collate(batch):
    ids, pm, am = zip(*batch)
    return {'input_ids': torch.stack(ids), 'prompt_mask': torch.stack(pm), 'attention_mask': torch.stack(am)}
loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate)

# ── 3. Oracle order 收集 ─────────────────────────────────────
print('[3/6] Collecting oracle unmasking orders ...')
batch = next(iter(loader))
x_0, pm = batch['input_ids'], batch['prompt_mask']
oracle_cfg = OracleConfig(num_steps=8, num_noise_levels=3)
oracle = collect_oracle_order(model, x_0, pm, config=oracle_cfg)
assert oracle.shape == x_0.shape, f'Shape mismatch: {oracle.shape} vs {x_0.shape}'
assert (oracle[:, :16] == -1).all(), 'Prompt should be -1'
gen = oracle[:, 16:]
assert (gen[gen >= 0] < 8).all(), 'Order values should be < num_steps'
print(f'       Oracle order sample [pos 16..31]: {oracle[0, 16:32].tolist()}')

# ── 4. 监督训练 ─────────────────────────────────────────────
print('[4/6] Training supervised planner (loss=$LOSS_TYPE) ...')
cfg = SupervisedPlannerConfig(
    num_epochs=2,
    batch_size=8,
    oracle_num_steps=8,
    oracle_noise_levels=3,
    oracle_max_samples=32,
    loss_type='$LOSS_TYPE',
    log_every=5,
    lr=1e-3,
)
trainer = SupervisedPlannerTrainer(model, loader, config=cfg)
trainer.train()

# ── 5. PlannerScheduler 测试 ─────────────────────────────────
print('[5/6] Testing PlannerScheduler ...')
scheduler = PlannerScheduler(trainer.get_planner())
current_mask = torch.ones(2, 64, dtype=torch.bool)
current_mask[:, :16] = False
is_unmasked = ~current_mask
logits = torch.randn(2, 64, 128)
confs = torch.rand(2, 64)

positions = scheduler.select_positions(
    step=0, total_steps=8,
    current_mask=current_mask, is_unmasked=is_unmasked,
    logits=logits, confidences=confs, n_to_select=4,
)
assert positions.shape == (2, 64)
assert not positions[:, :16].any(), 'Should not select prompt positions'
for b in range(2):
    assert positions[b].sum() <= 4, f'Selected too many positions: {positions[b].sum()}'
print(f'       Selected positions (sample 0): {positions[0].nonzero(as_tuple=True)[0].tolist()}')

# ── 6. Save / Load ──────────────────────────────────────────
print('[6/6] Testing save/load ...')
with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, 'planner.pt')
    trainer.save_planner(path)
    assert os.path.exists(path), 'Save failed'

    # 重新加载
    trainer.load_planner(path)
    # 验证加载后仍然能 forward
    out = trainer.get_planner()(torch.rand(1, 64))
    assert out.shape == (1, 64)

print()
print('✅ Supervised Planner — ALL CHECKS PASSED')
"
