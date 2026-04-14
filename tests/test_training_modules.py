"""Tests for the three new training modules:

1. ProgressiveTrainer (PUMA progressive masking)
2. SupervisedPlannerTrainer + PlannerScheduler (oracle distillation)
3. UnmaskingPolicyRL with KL regularisation

All tests use mock models — no GPU required.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dllm_reason.models.base import DiffusionLM, DiffusionOutput


# ── Mock Model ─────────────────────���─────────────────────────���──────────────

SEQ_LEN = 32
VOCAB_SIZE = 64
MASK_ID = 63
BATCH_SIZE = 4


class MockDiffusionLM(DiffusionLM):
    """Minimal DiffusionLM for testing (no real transformer)."""

    def __init__(self):
        super().__init__(
            vocab_size=VOCAB_SIZE,
            max_seq_len=SEQ_LEN,
            mask_token_id=MASK_ID,
        )
        self._proj = nn.Linear(VOCAB_SIZE, VOCAB_SIZE)

    def forward(self, x_t, t, attention_mask=None):
        one_hot = F.one_hot(x_t.clamp(0, VOCAB_SIZE - 1), VOCAB_SIZE).float()
        logits = self._proj(one_hot)
        return DiffusionOutput(logits=logits, loss=None, confidences=None)

    def noise_input(self, x_0, t):
        sigma = t[:, None].expand_as(x_0)
        mask = torch.rand_like(sigma.float()) < sigma
        return torch.where(mask, self.mask_token_id, x_0)

    def compute_loss(self, x_0, attention_mask=None):
        B, L = x_0.shape
        t = torch.rand(B, device=x_0.device).clamp(1e-5, 1.0 - 1e-5)
        x_t = self.noise_input(x_0, t)
        output = self.forward(x_t, t, attention_mask)
        log_probs = F.log_softmax(output.logits, dim=-1)
        nll = -log_probs.gather(-1, x_0.unsqueeze(-1)).squeeze(-1)
        is_masked = (x_t == self.mask_token_id)
        return (nll * is_masked.float()).sum(-1).div(
            is_masked.float().sum(-1).clamp(min=1)
        ).mean()


@pytest.fixture
def model():
    m = MockDiffusionLM()
    m.eval()
    return m


@pytest.fixture
def train_loader():
    """DataLoader yielding batches with input_ids and prompt_mask."""
    B, L = 16, SEQ_LEN
    input_ids = torch.randint(0, VOCAB_SIZE - 1, (B, L))
    prompt_mask = torch.zeros(B, L, dtype=torch.bool)
    prompt_mask[:, :8] = True  # First 8 tokens are prompt
    attention_mask = torch.ones(B, L, dtype=torch.bool)
    dataset = TensorDataset(input_ids, prompt_mask, attention_mask)

    def collate(batch):
        ids, pm, am = zip(*batch)
        return {
            "input_ids": torch.stack(ids),
            "prompt_mask": torch.stack(pm),
            "attention_mask": torch.stack(am),
        }

    return DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate)


# ═════════════════════════════════════════════════════���═════════════════════════
#  Test 1: PUMA Progressive Masking
# ═══════════════════════════════════════════════════════════════════════════════

class TestProgressiveTrainer:

    def test_config_defaults(self):
        from dllm_reason.training.progressive_train import ProgressiveTrainConfig
        cfg = ProgressiveTrainConfig()
        assert cfg.progressive_ratio == 1.0
        assert cfg.progressive_warmup_steps == 2000
        assert cfg.loss_on_answer_only is True

    def test_progressive_mask_construction(self, model, train_loader):
        from dllm_reason.training.progressive_train import (
            ProgressiveTrainer, ProgressiveTrainConfig,
        )
        cfg = ProgressiveTrainConfig(max_steps=1, progressive_ratio=1.0)
        trainer = ProgressiveTrainer(model, train_loader, config=cfg)

        batch = next(iter(train_loader))
        x_0 = batch["input_ids"]
        prompt_mask = batch["prompt_mask"]
        t = torch.full((x_0.shape[0],), 0.5)

        mask = trainer._build_progressive_mask(x_0, t, None, prompt_mask)

        # Should be (B, L) bool
        assert mask.shape == x_0.shape
        assert mask.dtype == torch.bool
        # Prompt positions should NOT be masked
        assert not mask[:, :8].any(), "Prompt positions should not be masked"
        # Some generation positions should be masked
        assert mask[:, 8:].any(), "Some gen positions should be masked"

    def test_loss_computation(self, model, train_loader):
        from dllm_reason.training.progressive_train import (
            ProgressiveTrainer, ProgressiveTrainConfig,
        )
        cfg = ProgressiveTrainConfig(max_steps=1, progressive_ratio=0.5)
        trainer = ProgressiveTrainer(model, train_loader, config=cfg)

        batch = next(iter(train_loader))
        x_0 = batch["input_ids"]
        prompt_mask = batch["prompt_mask"]
        attention_mask = batch["attention_mask"]

        loss = trainer._compute_progressive_loss(x_0, attention_mask, prompt_mask)

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() > 0, "Loss should be positive"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_curriculum_warmup(self, model, train_loader):
        from dllm_reason.training.progressive_train import (
            ProgressiveTrainer, ProgressiveTrainConfig,
        )
        cfg = ProgressiveTrainConfig(
            max_steps=4,
            progressive_ratio=1.0,
            progressive_warmup_steps=4,
            log_every=2,
        )
        trainer = ProgressiveTrainer(model, train_loader, config=cfg)
        # Should run without error — curriculum ramps from 0→1 over 4 steps
        trainer.train()
        assert trainer.global_step == 4

    def test_pure_random_mode(self, model, train_loader):
        """progressive_ratio=0 should behave like standard training."""
        from dllm_reason.training.progressive_train import (
            ProgressiveTrainer, ProgressiveTrainConfig,
        )
        cfg = ProgressiveTrainConfig(
            max_steps=2, progressive_ratio=0.0, log_every=1
        )
        trainer = ProgressiveTrainer(model, train_loader, config=cfg)
        trainer.train()
        assert trainer.global_step == 2


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 2: Supervised Planner (Oracle Distillation)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSupervisedPlanner:

    def test_oracle_order_collection(self, model, train_loader):
        from dllm_reason.training.supervised_planner import (
            collect_oracle_order, OracleConfig,
        )
        batch = next(iter(train_loader))
        x_0 = batch["input_ids"]
        prompt_mask = batch["prompt_mask"]
        cfg = OracleConfig(num_steps=8, num_noise_levels=2)

        oracle = collect_oracle_order(model, x_0, prompt_mask, config=cfg)

        assert oracle.shape == x_0.shape
        assert oracle.dtype == torch.long
        # Prompt positions should be -1
        assert (oracle[:, :8] == -1).all(), "Prompt positions should be -1"
        # Generation positions should be in [0, num_steps)
        gen_orders = oracle[:, 8:]
        valid = gen_orders >= 0
        assert valid.any()
        assert (gen_orders[valid] < 8).all()

    def test_oracle_dataset_collection(self, model, train_loader):
        from dllm_reason.training.supervised_planner import (
            collect_oracle_dataset, OracleConfig,
        )
        cfg = OracleConfig(num_steps=8, num_noise_levels=2)
        dataset = collect_oracle_dataset(
            model, train_loader, config=cfg, max_samples=8
        )
        assert len(dataset) == 8
        # Should return 3-tuple tensors
        conf, oracle, gen_mask = dataset[0]
        assert conf.shape == (SEQ_LEN,)
        assert oracle.shape == (SEQ_LEN,)
        assert gen_mask.shape == (SEQ_LEN,)

    def test_supervised_planner_training(self, model, train_loader):
        from dllm_reason.training.supervised_planner import (
            SupervisedPlannerTrainer, SupervisedPlannerConfig,
        )
        cfg = SupervisedPlannerConfig(
            num_epochs=1,
            batch_size=4,
            oracle_num_steps=4,
            oracle_noise_levels=2,
            oracle_max_samples=8,
            loss_type="ranking",
            log_every=1,
        )
        trainer = SupervisedPlannerTrainer(model, train_loader, config=cfg)
        trainer.train()

        planner = trainer.get_planner()
        assert planner is not None
        # Test forward pass
        dummy_conf = torch.rand(2, SEQ_LEN)
        logits = planner(dummy_conf)
        assert logits.shape == (2, SEQ_LEN)

    def test_regression_loss_mode(self, model, train_loader):
        from dllm_reason.training.supervised_planner import (
            SupervisedPlannerTrainer, SupervisedPlannerConfig,
        )
        cfg = SupervisedPlannerConfig(
            num_epochs=1,
            batch_size=4,
            oracle_num_steps=4,
            oracle_noise_levels=2,
            oracle_max_samples=8,
            loss_type="regression",
            log_every=1,
        )
        trainer = SupervisedPlannerTrainer(model, train_loader, config=cfg)
        trainer.train()  # Should complete without error

    def test_planner_scheduler(self, model, train_loader):
        from dllm_reason.training.supervised_planner import (
            SupervisedPlannerTrainer, SupervisedPlannerConfig,
            PlannerScheduler,
        )
        cfg = SupervisedPlannerConfig(
            num_epochs=1, batch_size=4,
            oracle_num_steps=4, oracle_noise_levels=2,
            oracle_max_samples=8, log_every=1,
        )
        trainer = SupervisedPlannerTrainer(model, train_loader, config=cfg)
        trainer.train()

        scheduler = PlannerScheduler(trainer.get_planner())

        # Test select_positions
        B, L = 2, SEQ_LEN
        current_mask = torch.ones(B, L, dtype=torch.bool)
        current_mask[:, :8] = False  # Prompt already unmasked
        is_unmasked = ~current_mask
        logits = torch.randn(B, L, VOCAB_SIZE)
        confidences = torch.rand(B, L)

        positions = scheduler.select_positions(
            step=0, total_steps=8,
            current_mask=current_mask,
            is_unmasked=is_unmasked,
            logits=logits,
            confidences=confidences,
            n_to_select=4,
        )

        assert positions.shape == (B, L)
        assert positions.dtype == torch.bool
        # Should only select from masked positions
        assert not positions[:, :8].any()
        # Should select exactly n_to_select (or fewer if not enough masked)
        for b in range(B):
            assert positions[b].sum() <= 4

    def test_save_load_planner(self, model, train_loader, tmp_path):
        from dllm_reason.training.supervised_planner import (
            SupervisedPlannerTrainer, SupervisedPlannerConfig,
        )
        cfg = SupervisedPlannerConfig(
            num_epochs=1, batch_size=4,
            oracle_num_steps=4, oracle_noise_levels=2,
            oracle_max_samples=8, log_every=1,
        )
        trainer = SupervisedPlannerTrainer(model, train_loader, config=cfg)
        trainer.train()

        path = str(tmp_path / "planner.pt")
        trainer.save_planner(path)
        trainer.load_planner(path)  # Should not raise


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 3: KL-Regularised UnmaskingPolicyRL
# ═══════════════════════════════════════════════════════════════════════════════

class TestKLRegularisation:

    def _reward_fn(self, seq, batch):
        """Dummy reward: fraction of non-mask tokens."""
        return (seq != MASK_ID).float().mean().item()

    def test_config_kl_defaults(self):
        from dllm_reason.training.rl_train import UnmaskingPolicyConfig
        cfg = UnmaskingPolicyConfig()
        assert cfg.kl_coeff == 0.0
        assert cfg.kl_ref_type == "uniform"
        assert cfg.ref_policy_path is None

    def test_vanilla_reinforce_still_works(self, model, train_loader):
        """kl_coeff=0 should produce the same behaviour as before."""
        from dllm_reason.training.rl_train import (
            UnmaskingPolicyRL, UnmaskingPolicyConfig,
        )
        cfg = UnmaskingPolicyConfig(
            num_iterations=2,
            group_size=2,
            num_steps=4,
            kl_coeff=0.0,
            log_every=1,
        )
        trainer = UnmaskingPolicyRL(model, self._reward_fn, train_loader, cfg)
        trainer.train()

    def test_kl_uniform_reference(self, model, train_loader):
        """KL with uniform reference policy."""
        from dllm_reason.training.rl_train import (
            UnmaskingPolicyRL, UnmaskingPolicyConfig,
        )
        cfg = UnmaskingPolicyConfig(
            num_iterations=2,
            group_size=2,
            num_steps=4,
            kl_coeff=0.01,
            kl_ref_type="uniform",
            log_every=1,
        )
        trainer = UnmaskingPolicyRL(model, self._reward_fn, train_loader, cfg)
        trainer.train()

    def test_kl_dag_reference(self, model, train_loader):
        """KL with DAG-based reference policy."""
        from dllm_reason.training.rl_train import (
            UnmaskingPolicyRL, UnmaskingPolicyConfig,
        )
        from dllm_reason.graph.dag import TokenDAG

        dag = TokenDAG.linear_chain(SEQ_LEN)
        cfg = UnmaskingPolicyConfig(
            num_iterations=2,
            group_size=2,
            num_steps=4,
            kl_coeff=0.01,
            kl_ref_type="dag",
            log_every=1,
        )
        trainer = UnmaskingPolicyRL(
            model, self._reward_fn, train_loader, cfg, dag=dag
        )
        trainer.train()

    def test_kl_pretrained_reference(self, model, train_loader, tmp_path):
        """KL with pretrained planner as reference policy."""
        from dllm_reason.training.rl_train import (
            UnmaskingPolicyRL, UnmaskingPolicyConfig, UnmaskingPolicyNet,
        )
        # Save a dummy planner
        planner = UnmaskingPolicyNet(d_model=64, n_heads=4)
        path = str(tmp_path / "ref_planner.pt")
        torch.save(planner.state_dict(), path)

        cfg = UnmaskingPolicyConfig(
            num_iterations=2,
            group_size=2,
            num_steps=4,
            kl_coeff=0.01,
            kl_ref_type="pretrained",
            ref_policy_path=path,
            log_every=1,
        )
        trainer = UnmaskingPolicyRL(model, self._reward_fn, train_loader, cfg)
        trainer.train()

    def test_ref_log_prob_shapes(self, model):
        """Verify _ref_log_prob returns correct shape."""
        from dllm_reason.training.rl_train import (
            UnmaskingPolicyRL, UnmaskingPolicyConfig,
        )
        # Minimal setup — just need _ref_log_prob
        dummy_loader = DataLoader(
            TensorDataset(torch.zeros(1, SEQ_LEN, dtype=torch.long)),
            batch_size=1,
        )
        cfg = UnmaskingPolicyConfig(kl_coeff=0.01, kl_ref_type="uniform")
        trainer = UnmaskingPolicyRL(
            model, lambda s, b: 0.0, dummy_loader, cfg
        )

        B, L = 2, SEQ_LEN
        conf = torch.rand(B, L)
        action = torch.randint(0, 2, (B, L)).float()
        is_masked = torch.ones(B, L, dtype=torch.bool)

        ref_lp = trainer._ref_log_prob(conf, action, is_masked)
        assert ref_lp.shape == (B,)
        assert torch.isfinite(ref_lp).all()

    def test_policy_rollout_returns_three_tensors(self, model, train_loader):
        """_policy_rollout should now return 3 values (seq, lp, ref_lp)."""
        from dllm_reason.training.rl_train import (
            UnmaskingPolicyRL, UnmaskingPolicyConfig,
        )
        cfg = UnmaskingPolicyConfig(
            num_steps=4, kl_coeff=0.01, kl_ref_type="uniform"
        )
        trainer = UnmaskingPolicyRL(
            model, lambda s, b: 0.0, train_loader, cfg
        )

        batch = next(iter(train_loader))
        prompt_ids = batch["input_ids"]
        prompt_mask = batch["prompt_mask"]
        gen_length = SEQ_LEN - 8

        result = trainer._policy_rollout(prompt_ids, prompt_mask, gen_length)
        assert len(result) == 3
        seq, lp, ref_lp = result
        assert seq.shape == prompt_ids.shape
        assert lp.shape == (BATCH_SIZE,)
        assert ref_lp.shape == (BATCH_SIZE,)
