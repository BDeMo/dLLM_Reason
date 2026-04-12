"""Supervised Planner Training via Oracle Unmasking Order Distillation.

Trains a lightweight planner network to predict a good unmasking order by
imitating an *oracle* order derived from ground-truth answers.

Oracle Order Construction
-------------------------
Given a (prompt, ground_truth_answer) pair:

  1. Start with the generation region fully masked.
  2. Run the frozen dLLM forward to get per-token predicted probabilities.
  3. For each masked position, compute the probability assigned to the
     *correct* token (ground truth).
  4. The oracle unmasks positions in **descending order of correctness
     probability** — i.e., the "easiest" tokens (highest p(correct)) are
     unmasked first.  This produces an easy-to-hard curriculum.
  5. Record the *step number* at which each position is unmasked →
     oracle_order tensor of shape (L,), values in [0, num_steps).

This is the "Gt-Margin" strategy from Where-to-Unmask (arXiv:2602.09501):
tokens that the model already predicts well are unmasked first, leaving
the hardest tokens for last when more context is available.

Planner Architecture
--------------------
Reuses ``UnmaskingPolicyNet`` (single-layer transformer on confidence
scores), but trained with cross-entropy against the oracle order instead
of RL.  The planner outputs per-token logits that rank positions by
predicted unmasking priority.

After supervised pre-training, the planner can be used as:
  - A standalone scheduler (replace max-confidence heuristic)
  - A warm-start for ``UnmaskingPolicyRL`` (RL fine-tuning)

Reference
---------
Where-to-Unmask: Improving Discrete Diffusion Models by Optimizing
Unmasking Schedule (arXiv:2602.09501)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dllm_reason.models.base import DiffusionLM
from dllm_reason.training.rl_train import UnmaskingPolicyNet
from dllm_reason.scheduler.base import UnmaskingScheduler
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


# ── Oracle Order Collection ──────────────────────────────────────────────────

@dataclass
class OracleConfig:
    """Configuration for oracle order collection."""
    num_steps: int = 32
    """Number of discrete steps in the oracle order (granularity)."""

    num_noise_levels: int = 8
    """Number of noise levels to average confidence over for robustness."""


@torch.no_grad()
def collect_oracle_order(
    model: DiffusionLM,
    x_0: torch.Tensor,
    prompt_mask: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    config: OracleConfig | None = None,
) -> torch.Tensor:
    """Collect oracle unmasking order for a batch of sequences.

    For each position in the generation region, compute how "easy" it is
    for the model to predict correctly, then assign an unmasking step
    (0 = unmask first, num_steps-1 = unmask last).

    Args:
        model:          Frozen dLLM.
        x_0:            (B, L) ground-truth token ids.
        prompt_mask:    (B, L) bool — True for prompt positions.
        attention_mask: (B, L) optional.
        config:         OracleConfig.

    Returns:
        oracle_order: (B, L) int tensor.  Values in [0, num_steps) for
                      generation positions; -1 for prompt positions.
    """
    cfg = config or OracleConfig()
    B, L = x_0.shape
    device = x_0.device

    gen_mask = ~prompt_mask.bool()
    if attention_mask is not None:
        gen_mask = gen_mask & attention_mask.bool()

    # Accumulate correctness probability across multiple noise levels
    # for a more robust estimate.
    acc_confidence = torch.zeros(B, L, device=device)
    noise_levels = torch.linspace(0.2, 0.9, cfg.num_noise_levels, device=device)

    for t_val in noise_levels:
        t = torch.full((B,), t_val.item(), device=device)
        x_t = model.noise_input(x_0, t)
        output = model.forward(x_t, t, attention_mask)
        probs = F.softmax(output.logits, dim=-1)  # (B, L, V)
        # Probability of the correct token
        p_correct = probs.gather(-1, x_0.unsqueeze(-1)).squeeze(-1)  # (B, L)
        acc_confidence += p_correct

    avg_confidence = acc_confidence / cfg.num_noise_levels

    # Set non-generation positions to -inf so they sort to the end
    avg_confidence[~gen_mask] = -float("inf")

    # Sort positions by confidence descending → highest confidence = step 0
    _, sorted_indices = avg_confidence.sort(dim=-1, descending=True)

    # Assign step numbers: position at rank r gets step floor(r / chunk_size)
    num_gen = gen_mask.float().sum(dim=-1)  # (B,)
    oracle_order = torch.full((B, L), -1, dtype=torch.long, device=device)

    for b in range(B):
        n = int(num_gen[b].item())
        if n == 0:
            continue
        # Positions sorted by confidence (descending)
        positions = sorted_indices[b, :n]
        # Map rank → step (quantise into num_steps bins)
        steps = torch.arange(n, device=device).float()
        steps = (steps / n * cfg.num_steps).long().clamp(max=cfg.num_steps - 1)
        oracle_order[b, positions] = steps

    return oracle_order


def collect_oracle_dataset(
    model: DiffusionLM,
    data_loader: DataLoader,
    config: OracleConfig | None = None,
    max_samples: int = 10000,
) -> TensorDataset:
    """Collect oracle orders for an entire dataset.

    Returns a TensorDataset of (confidence_at_full_mask, oracle_order, gen_mask)
    suitable for training the supervised planner.
    """
    cfg = config or OracleConfig()
    device = model.device

    all_confidence = []
    all_oracle = []
    all_gen_mask = []
    total = 0

    model.eval()
    for batch in data_loader:
        if total >= max_samples:
            break

        x_0 = batch["input_ids"].to(device)
        prompt_mask = batch.get(
            "prompt_mask",
            torch.zeros_like(x_0, dtype=torch.bool),
        ).to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        B, L = x_0.shape

        # Oracle order
        oracle = collect_oracle_order(model, x_0, prompt_mask, attention_mask, cfg)

        # Confidence at full mask (t≈1) — this is what the planner sees at step 0
        t_high = torch.full((B,), 0.95, device=device)
        x_masked = model.noise_input(x_0, t_high)
        with torch.no_grad():
            output = model.forward(x_masked, t_high, attention_mask)
        confidence = F.softmax(output.logits, dim=-1).max(dim=-1).values  # (B, L)

        gen_mask = ~prompt_mask.bool()
        if attention_mask is not None:
            gen_mask = gen_mask & attention_mask.bool()

        all_confidence.append(confidence.cpu())
        all_oracle.append(oracle.cpu())
        all_gen_mask.append(gen_mask.cpu())
        total += B

    confidence_t = torch.cat(all_confidence, dim=0)[:max_samples]
    oracle_t = torch.cat(all_oracle, dim=0)[:max_samples]
    gen_mask_t = torch.cat(all_gen_mask, dim=0)[:max_samples]

    logger.info(f"Collected oracle dataset: {confidence_t.shape[0]} samples")
    return TensorDataset(confidence_t, oracle_t, gen_mask_t)


# ── Supervised Planner Trainer ───────────────────────────────────────────────

@dataclass
class SupervisedPlannerConfig:
    """Configuration for supervised planner training."""
    lr: float = 3e-4
    num_epochs: int = 20
    batch_size: int = 64
    max_grad_norm: float = 1.0
    log_every: int = 50

    # Planner architecture (same as UnmaskingPolicyNet)
    d_model: int = 64
    n_heads: int = 4
    dropout: float = 0.1

    # Oracle collection
    oracle_num_steps: int = 32
    oracle_noise_levels: int = 8
    oracle_max_samples: int = 10000

    # Loss type
    loss_type: str = "ranking"
    """'ranking' (pairwise ranking loss) or 'regression' (MSE on step number)."""


class SupervisedPlannerTrainer:
    """Train an unmasking planner via supervised learning on oracle orders.

    The planner learns to predict which tokens should be unmasked first
    by imitating an oracle ordering derived from ground-truth answers.

    Two loss functions are supported:

    1. **ranking** (default): Pairwise ranking loss — for each pair of
       positions (i, j) where oracle says i should be unmasked before j,
       the planner should assign higher priority (logit) to i.
       Uses margin ranking loss for robustness.

    2. **regression**: Direct MSE regression on normalised step numbers.
       Simpler but less robust to label noise.
    """

    def __init__(
        self,
        model: DiffusionLM,
        train_loader: DataLoader,
        config: SupervisedPlannerConfig | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.config = config or SupervisedPlannerConfig()

        cfg = self.config
        device = model.device

        # Freeze base model
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        # Build planner (reuse UnmaskingPolicyNet architecture)
        self.planner = UnmaskingPolicyNet(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
        ).to(device)

        self.optimizer = torch.optim.Adam(self.planner.parameters(), lr=cfg.lr)
        self._device = device

        n_params = sum(p.numel() for p in self.planner.parameters())
        logger.info(
            f"SupervisedPlannerTrainer ready — planner params={n_params:,}"
        )

    def _ranking_loss(
        self,
        logits: torch.Tensor,
        oracle_order: torch.Tensor,
        gen_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise margin ranking loss.

        For each sample, sample pairs (i, j) where oracle_order[i] < oracle_order[j]
        (i should be unmasked before j). Train planner to output logits[i] > logits[j].
        """
        B, L = logits.shape
        device = logits.device
        total_loss = torch.tensor(0.0, device=device)
        num_pairs = 0

        for b in range(B):
            mask = gen_mask[b]  # (L,)
            order = oracle_order[b]  # (L,)
            lgt = logits[b]  # (L,)

            valid = mask & (order >= 0)
            valid_idx = valid.nonzero(as_tuple=True)[0]
            n = valid_idx.shape[0]
            if n < 2:
                continue

            # Sample pairs efficiently: up to 64 pairs per sample
            max_pairs = min(64, n * (n - 1) // 2)
            idx_a = torch.randint(n, (max_pairs,), device=device)
            idx_b = torch.randint(n, (max_pairs,), device=device)
            # Remove self-pairs
            diff = idx_a != idx_b
            idx_a, idx_b = idx_a[diff], idx_b[diff]

            pos_a = valid_idx[idx_a]
            pos_b = valid_idx[idx_b]
            order_a = order[pos_a]
            order_b = order[pos_b]
            logit_a = lgt[pos_a]
            logit_b = lgt[pos_b]

            # oracle_order: lower = unmask earlier = should have HIGHER logit
            # target: +1 if a should be unmasked before b (order_a < order_b)
            target = torch.sign((order_b - order_a).float())  # +1, -1, or 0
            nonzero = target != 0
            if nonzero.sum() == 0:
                continue

            loss = F.margin_ranking_loss(
                logit_a[nonzero],
                logit_b[nonzero],
                target[nonzero],
                margin=0.1,
            )
            total_loss = total_loss + loss
            num_pairs += 1

        return total_loss / max(num_pairs, 1)

    def _regression_loss(
        self,
        logits: torch.Tensor,
        oracle_order: torch.Tensor,
        gen_mask: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """MSE regression loss on normalised oracle step numbers."""
        # Normalise oracle order to [0, 1]
        target = oracle_order.float() / max(num_steps - 1, 1)
        # Planner should output HIGH value for early unmask (low step number)
        # So target = 1 - normalised_step (1 = unmask first)
        target = 1.0 - target

        # Only compute loss at valid generation positions
        valid = gen_mask & (oracle_order >= 0)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        pred = torch.sigmoid(logits)  # Map to [0, 1]
        loss = F.mse_loss(pred[valid], target[valid])
        return loss

    def train(self) -> None:
        """Run supervised planner training.

        Step 1: Collect oracle orders from the training data.
        Step 2: Train the planner to match the oracle.
        """
        cfg = self.config
        device = self._device

        # ── Step 1: Collect oracle dataset ──────────────────────────────
        logger.info("Collecting oracle unmasking orders ...")
        oracle_cfg = OracleConfig(
            num_steps=cfg.oracle_num_steps,
            num_noise_levels=cfg.oracle_noise_levels,
        )
        oracle_dataset = collect_oracle_dataset(
            self.model,
            self.train_loader,
            config=oracle_cfg,
            max_samples=cfg.oracle_max_samples,
        )
        oracle_loader = DataLoader(
            oracle_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
        )

        # ── Step 2: Train planner ───────────────────────────────────────
        logger.info(f"Training planner ({cfg.loss_type} loss) ...")
        self.planner.train()
        global_step = 0

        for epoch in range(cfg.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for confidence, oracle_order, gen_mask in oracle_loader:
                confidence = confidence.to(device)
                oracle_order = oracle_order.to(device)
                gen_mask = gen_mask.to(device)

                # Forward: planner takes confidence → priority logits
                logits = self.planner(confidence)  # (B, L)

                # Compute loss
                if cfg.loss_type == "ranking":
                    loss = self._ranking_loss(logits, oracle_order, gen_mask)
                elif cfg.loss_type == "regression":
                    loss = self._regression_loss(
                        logits, oracle_order, gen_mask, cfg.oracle_num_steps
                    )
                else:
                    raise ValueError(f"Unknown loss_type: {cfg.loss_type}")

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.planner.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                if global_step % cfg.log_every == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{cfg.num_epochs}  "
                        f"step {global_step}  "
                        f"loss={loss.item():.4f}"
                    )

            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(
                f"Epoch {epoch+1}/{cfg.num_epochs} complete — "
                f"avg_loss={avg_loss:.4f}"
            )

        logger.info("Supervised planner training complete.")

    def save_planner(self, path: str) -> None:
        """Save the trained planner weights."""
        torch.save(self.planner.state_dict(), path)
        logger.info(f"Planner saved to {path}")

    def load_planner(self, path: str) -> None:
        """Load planner weights from disk."""
        state = torch.load(path, map_location=self._device)
        self.planner.load_state_dict(state)
        logger.info(f"Planner loaded from {path}")

    def get_planner(self) -> UnmaskingPolicyNet:
        """Return the trained planner for use as a scheduler or RL warm-start."""
        return self.planner


# ── Planner-based Scheduler ─────────────────────────────────────────────────

class PlannerScheduler(UnmaskingScheduler):
    """Unmasking scheduler driven by a trained planner network.

    At each step, feeds the model's confidence scores to the planner,
    which outputs per-token priority logits.  Positions with the highest
    priority (logit) among the still-masked tokens are unmasked.

    This replaces the simple max-confidence heuristic with a learned
    ranking that considers inter-token relationships via self-attention.
    """

    def __init__(self, planner: UnmaskingPolicyNet):
        self.planner = planner
        self.planner.eval()

    @torch.no_grad()
    def select_positions(
        self,
        step: int,
        total_steps: int,
        current_mask: torch.Tensor,
        is_unmasked: torch.Tensor,
        logits: torch.Tensor,
        confidences: torch.Tensor,
        block_mask: torch.Tensor | None = None,
        n_to_select: int = 1,
    ) -> torch.Tensor:
        """Select positions to unmask based on planner priority."""
        B, L = current_mask.shape
        device = current_mask.device

        # Planner input: confidence scores (zero out non-masked positions)
        conf_input = confidences * current_mask.float()
        priority = self.planner(conf_input)  # (B, L)

        # Only consider currently masked positions
        eligible = current_mask.clone()
        if block_mask is not None:
            eligible = eligible & block_mask

        # Set ineligible positions to -inf
        priority = priority.masked_fill(~eligible, -float("inf"))

        # Select top-n positions by priority
        positions = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            valid = eligible[b]
            if valid.sum() == 0:
                continue
            n = min(n_to_select, int(valid.sum().item()))
            _, top_idx = priority[b].topk(n)
            positions[b, top_idx] = True

        return positions
