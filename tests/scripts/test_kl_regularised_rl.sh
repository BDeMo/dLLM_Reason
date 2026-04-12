#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  KL-Regularised UnmaskingPolicyRL — 快速功能验证
#
#  验证三种 reference policy：
#    1. uniform  — 均匀分布 baseline
#    2. dag      — DAG-guided reference
#    3. pretrained — 预训练 planner 作为 reference
#
#  Usage:
#    bash tests/scripts/test_kl_regularised_rl.sh
#    bash tests/scripts/test_kl_regularised_rl.sh uniform
#    bash tests/scripts/test_kl_regularised_rl.sh dag
#    bash tests/scripts/test_kl_regularised_rl.sh pretrained
#    bash tests/scripts/test_kl_regularised_rl.sh all
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

MODE="${1:-all}"

cat <<EOF
╔═══════════════════════════════════════════════════════════╗
║  KL-Regularised UnmaskingPolicyRL — Smoke Test           ║
║  mode=$MODE                                               ║
╚═══════════════════════════════════════════════════════════╝
EOF

python -c "
import os, sys, tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dllm_reason.models.base import DiffusionLM, DiffusionOutput
from dllm_reason.training.rl_train import (
    UnmaskingPolicyRL, UnmaskingPolicyConfig, UnmaskingPolicyNet,
)

MODE = '$MODE'

# ── Mock model ───────────────────────────────────────────────
class MockDLLM(DiffusionLM):
    def __init__(self):
        super().__init__(vocab_size=64, max_seq_len=32, mask_token_id=63)
        self._proj = nn.Linear(64, 64)
    def forward(self, x_t, t, attention_mask=None):
        oh = F.one_hot(x_t.clamp(0, 63), 64).float()
        return DiffusionOutput(logits=self._proj(oh), loss=None, confidences=None)
    def noise_input(self, x_0, t):
        sigma = t[:, None].expand_as(x_0)
        mask = torch.rand_like(sigma.float()) < sigma
        return torch.where(mask, self.mask_token_id, x_0)
    def compute_loss(self, x_0, attention_mask=None):
        return torch.tensor(0.0)

model = MockDLLM()
model.eval()

# ── 数据 ────────────────────────────────────────────────────
B, L = 16, 32
input_ids = torch.randint(0, 63, (B, L))
prompt_mask = torch.zeros(B, L, dtype=torch.bool)
prompt_mask[:, :8] = True
dataset = TensorDataset(input_ids, prompt_mask)
def collate(batch):
    ids, pm = zip(*batch)
    return {'input_ids': torch.stack(ids), 'prompt_mask': torch.stack(pm)}
loader = DataLoader(dataset, batch_size=4, collate_fn=collate)

def reward_fn(seq, batch):
    return (seq != 63).float().mean().item()

ITERS, GROUPS, STEPS = 3, 2, 4

# ── Test 1: Uniform reference ───────────────────────────────
if MODE in ('all', 'uniform'):
    print('[1/3] KL with uniform reference (kl_coeff=0.01) ...')
    cfg = UnmaskingPolicyConfig(
        num_iterations=ITERS, group_size=GROUPS, num_steps=STEPS,
        kl_coeff=0.01, kl_ref_type='uniform', log_every=1,
    )
    trainer = UnmaskingPolicyRL(model, reward_fn, loader, cfg)
    trainer.train()
    print('      ✓ uniform reference OK')

# ── Test 2: DAG reference ───────────────────────────────────
if MODE in ('all', 'dag'):
    print('[2/3] KL with DAG reference (kl_coeff=0.01) ...')
    from dllm_reason.graph.dag import TokenDAG
    dag = TokenDAG.linear_chain(L)
    cfg = UnmaskingPolicyConfig(
        num_iterations=ITERS, group_size=GROUPS, num_steps=STEPS,
        kl_coeff=0.01, kl_ref_type='dag', log_every=1,
    )
    trainer = UnmaskingPolicyRL(model, reward_fn, loader, cfg, dag=dag)
    trainer.train()
    print('      ✓ DAG reference OK')

# ── Test 3: Pretrained planner reference ────────────────────
if MODE in ('all', 'pretrained'):
    print('[3/3] KL with pretrained planner reference ...')
    with tempfile.TemporaryDirectory() as tmpdir:
        # 保存一个 dummy planner
        planner = UnmaskingPolicyNet(d_model=64, n_heads=4)
        ref_path = os.path.join(tmpdir, 'ref_planner.pt')
        torch.save(planner.state_dict(), ref_path)

        cfg = UnmaskingPolicyConfig(
            num_iterations=ITERS, group_size=GROUPS, num_steps=STEPS,
            kl_coeff=0.01, kl_ref_type='pretrained',
            ref_policy_path=ref_path, log_every=1,
        )
        trainer = UnmaskingPolicyRL(model, reward_fn, loader, cfg)
        trainer.train()
    print('      ✓ pretrained reference OK')

# ── Bonus: 对比 vanilla vs KL ───────────────────────────────
if MODE == 'all':
    print()
    print('[Bonus] Comparing vanilla REINFORCE (kl=0) vs KL-regularised (kl=0.05) ...')
    for label, kl in [('vanilla', 0.0), ('KL-reg', 0.05)]:
        cfg = UnmaskingPolicyConfig(
            num_iterations=5, group_size=2, num_steps=4,
            kl_coeff=kl, kl_ref_type='uniform', log_every=5,
        )
        t = UnmaskingPolicyRL(model, reward_fn, loader, cfg)
        t.train()

print()
print('✅ KL-Regularised RL — ALL CHECKS PASSED')
"
