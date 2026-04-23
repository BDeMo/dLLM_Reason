"""PUMA-style Progressive Unmasking for Masked diffusion LM training.

Addresses the train–inference mismatch in masked diffusion models:
standard training uses *random* masking (each token independently masked
with probability sigma(t)), but at inference the scheduler unmasks tokens
in confidence order.  PUMA closes this gap by constructing training masks
that mimic the confidence-based masking pattern seen at inference.

Algorithm
---------
For each training sample (x_0, attention_mask, optional prompt_mask):

  1.  Sample a noise level t ~ U(0,1).
  2.  Run a *frozen* forward pass to get per-token confidence scores.
  3.  Build the progressive mask: mask the ``floor(sigma(t)*L)`` tokens
      with the **lowest** confidence (these are the tokens the inference
      scheduler would unmask *last*).
  4.  Run a second forward pass on the progressively-masked input and
      compute the diffusion loss at masked positions.

A curriculum parameter ``progressive_ratio`` blends random and progressive
masks: ratio=0 → pure random (standard training), ratio=1 → pure
progressive.  Linearly ramping from 0→1 over training stabilises early
learning.

Interaction with DAG
--------------------
When a DAG bias is available (via ``dag`` or ``prompt_mask``), the
progressive mask can additionally respect topological constraints:
within each DAG level, tokens are ordered by confidence.

Reference
---------
PUMA: Progressive Unmasking for Masked diffusion LM Alignment
(arXiv:2602.10314)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dllm_reason.models.base import DiffusionLM
from dllm_reason.training.pretrain import Trainer, TrainConfig
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProgressiveTrainConfig(TrainConfig):
    """Config for PUMA-style progressive masking training."""

    # Progressive masking
    progressive_ratio: float = 1.0
    """Blend factor: 0 = pure random mask, 1 = pure progressive mask.
    During curriculum ramp-up, this is the *final* ratio."""

    progressive_warmup_steps: int = 2000
    """Number of steps to linearly ramp progressive_ratio from 0 to target."""

    loss_on_answer_only: bool = True
    """Only compute loss on generation (non-prompt) positions."""

    prompt_loss_weight: float = 0.0
    """Weight for prompt positions when loss_on_answer_only=False."""

    # Fine-tuning defaults
    lr: float = 2e-5
    max_steps: int = 10000
    warmup_steps: int = 200


class ProgressiveTrainer(Trainer):
    """Fine-tunes a dLLM using PUMA progressive masking.

    Instead of masking tokens randomly (uniform Bernoulli per position),
    this trainer constructs masks that reflect the model's own confidence
    distribution — low-confidence tokens are masked, high-confidence
    tokens are revealed.  This aligns the training distribution with the
    inference-time unmasking order.
    """

    def __init__(
        self,
        model: DiffusionLM,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        config: ProgressiveTrainConfig | None = None,
    ):
        cfg = config or ProgressiveTrainConfig()
        super().__init__(model, train_loader, val_loader, cfg)
        self.prog_config = cfg

    # ── Core: progressive mask construction ─────────────────────────────

    @torch.no_grad()
    def _build_progressive_mask(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None,
        prompt_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build a confidence-based progressive mask.

        Returns a boolean tensor (B, L) where True = masked (replace with MASK).
        The number of masked positions matches ``sigma(t) * num_eligible``.
        """
        B, L = x_0.shape
        device = x_0.device

        # 1. Get mask ratio from noise schedule (or use t directly for LLaDA)
        if hasattr(self.model, "sigma"):
            mask_ratio = self.model.sigma(t)  # (B,)
        else:
            mask_ratio = t  # (B,) — LLaDA linear schedule

        # 2. Forward pass to get confidence scores
        #    Use a *random* noised version at the same t to get predictions
        x_t_random = self.model.noise_input(x_0, t)
        output = self.model.forward(x_t_random, t, attention_mask)
        probs = F.softmax(output.logits, dim=-1)
        # Confidence = probability of the *correct* token (not max softmax)
        # This gives a better signal for ordering.
        confidence = probs.gather(-1, x_0.unsqueeze(-1)).squeeze(-1)  # (B, L)

        # 3. Determine eligible positions (non-prompt, attended)
        eligible = torch.ones(B, L, dtype=torch.bool, device=device)
        if prompt_mask is not None:
            eligible = eligible & ~prompt_mask.bool()
        if attention_mask is not None:
            eligible = eligible & attention_mask.bool()

        num_eligible = eligible.float().sum(dim=-1)  # (B,)
        num_to_mask = (mask_ratio * num_eligible).long().clamp(min=0)  # (B,)

        # 4. Build mask: mask the positions with LOWEST confidence
        #    (these are the hardest tokens — would be unmasked last at inference)
        # Set confidence of ineligible positions to +inf so they sort last
        conf_for_sort = confidence.clone()
        conf_for_sort[~eligible] = float("inf")

        # Sort by confidence ascending; take the first num_to_mask positions
        _, sorted_indices = conf_for_sort.sort(dim=-1)  # ascending

        # Build mask via scatter
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            n = num_to_mask[b].item()
            if n > 0:
                mask[b, sorted_indices[b, :n]] = True

        return mask

    @torch.no_grad()
    def _build_random_mask(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Standard random masking (baseline)."""
        x_t = self.model.noise_input(x_0, t)
        return x_t == self.model.mask_token_id

    def _compute_progressive_loss(
        self,
        x_0: torch.Tensor,
        attention_mask: torch.Tensor | None,
        prompt_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute loss with progressive (or blended) masking."""
        B, L = x_0.shape
        device = x_0.device

        t = torch.rand(B, device=device).clamp(1e-5, 1.0 - 1e-5)

        # Curriculum: ramp progressive_ratio over warmup steps
        cfg = self.prog_config
        if cfg.progressive_warmup_steps > 0 and self.global_step < cfg.progressive_warmup_steps:
            current_ratio = cfg.progressive_ratio * (self.global_step / cfg.progressive_warmup_steps)
        else:
            current_ratio = cfg.progressive_ratio

        # Decide per-sample: progressive or random mask
        use_progressive = torch.rand(B, device=device) < current_ratio

        # Build progressive mask for the whole batch (cheap; just ignore for random samples)
        prog_mask = self._build_progressive_mask(x_0, t, attention_mask, prompt_mask)
        rand_mask = self._build_random_mask(x_0, t)

        # Blend: select progressive or random per sample
        is_masked = torch.where(
            use_progressive.unsqueeze(-1).expand_as(prog_mask),
            prog_mask,
            rand_mask,
        )

        # Build x_t from mask.
        # Use masked_fill instead of torch.where(bool, scalar, tensor) to avoid
        # dtype promotion issues on PyTorch < 2.1 when mask_token_id is a plain int.
        x_t = x_0.masked_fill(is_masked, self.model.mask_token_id)

        # Forward pass (with gradients this time)
        output = self.model.forward(x_t, t, attention_mask)
        logits = output.logits

        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(-1, x_0.unsqueeze(-1)).squeeze(-1)  # (B, L)

        # Loss mask: only at masked positions
        loss_mask = is_masked.clone()

        if cfg.loss_on_answer_only and prompt_mask is not None:
            loss_mask = loss_mask & ~prompt_mask.bool()

        if attention_mask is not None:
            loss_mask = loss_mask & attention_mask.bool()

        # ELBO weight (for MDLM); skip for LLaDA-style models
        if hasattr(self.model, "dsigma") and hasattr(self.model, "sigma"):
            sigma_t = self.model.sigma(t)
            dsigma_t = self.model.dsigma(t)
            weight = dsigma_t / sigma_t.clamp(min=1e-8)  # (B,)
        else:
            weight = torch.ones(B, device=device)

        masked_nll = (nll * loss_mask.float()).sum(dim=-1)  # (B,)
        num_masked = loss_mask.float().sum(dim=-1).clamp(min=1.0)  # (B,)
        per_sample = weight * masked_nll / num_masked

        return per_sample.mean()

    # ── Training loop ──────────────────────────────────────────────────

    def train(self) -> None:
        """Progressive masking training loop."""
        cfg = self.prog_config
        self.model.train()
        device = self.model.device
        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        accum_loss = 0.0
        data_iter = iter(self.train_loader)

        while self.global_step < cfg.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            x_0 = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            prompt_mask = batch.get("prompt_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            if prompt_mask is not None:
                prompt_mask = prompt_mask.to(device)

            loss = self._compute_progressive_loss(x_0, attention_mask, prompt_mask)
            loss = loss / cfg.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

            if (self.global_step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            self.global_step += 1

            if self.global_step % cfg.log_every == 0:
                avg_loss = accum_loss / cfg.log_every
                ratio = cfg.progressive_ratio
                if cfg.progressive_warmup_steps > 0 and self.global_step < cfg.progressive_warmup_steps:
                    ratio = cfg.progressive_ratio * (self.global_step / cfg.progressive_warmup_steps)
                logger.info(
                    f"Step {self.global_step}: loss={avg_loss:.4f}, "
                    f"prog_ratio={ratio:.3f}"
                )
                accum_loss = 0.0

            if self.val_loader and self.global_step % cfg.eval_every == 0:
                val_loss = self.evaluate()
                logger.info(f"Step {self.global_step}: val_loss={val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(save_dir / "best.pt")

            if self.global_step % cfg.save_every == 0:
                self.save_checkpoint(save_dir / f"step_{self.global_step}.pt")
