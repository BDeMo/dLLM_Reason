"""Train the BackPlay CorrectionHead.

Self-generated training data pipeline:
  1. Start from a clean sequence x_0.
  2. Use the frozen diffusion backbone at a given noise level to perform greedy
     unmask, obtaining a "noisy-clean" sequence x_noisy_clean.
     (The backbone's own prediction errors mirror the true error distribution,
     which is more realistic than random token replacement.)
  3. Ground truth: x_0; input: x_noisy_clean; labels:
       - is_error  = (x_noisy_clean != x_0)  ->  BCE target for revise_prob
       - new_token = x_0                      ->  CE  target for new_logits

Loss = BCE(revise_prob, is_error) + alpha * CE(new_logits at error positions, x_0)

Reasoning-aware extension (contribution #4)
--------------------------------------------
Optional ``structure: TokenDAG`` argument: when provided, an extra
"logical consistency loss" is added — for token combinations that violate DAG
topology (e.g. a child is already unmasked but its parent is still wrong),
the revise_prob target for the parent is increased.

Reference
---------
BackPlay (arXiv:2601.06428)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dllm_reason.models.base import DiffusionLM
from dllm_reason.models.correction_head import CorrectionHead, CorrectionHeadConfig
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CorrectionTrainConfig:
    num_epochs: int = 3
    batch_size: int = 8
    lr: float = 1e-3
    noise_levels: tuple = (0.3, 0.5, 0.7)
    """Noise levels sampled at random each batch."""
    ce_weight: float = 1.0
    bce_weight: float = 1.0
    log_every: int = 10
    save_dir: str = "runs/correction_head"
    grad_clip: float = 1.0
    # Reasoning-aware options
    structure_consistency_weight: float = 0.0
    """Enable DAG consistency loss when > 0 (requires ``structure`` arg)."""


class CorrectionTrainer:
    """Train a CorrectionHead with self-generated data.

    Frozen:    base DiffusionLM  (used only to sample erroneous tokens)
    Trainable: CorrectionHead
    """

    def __init__(
        self,
        base_model: DiffusionLM,
        head: CorrectionHead,
        train_loader: DataLoader,
        config: CorrectionTrainConfig | None = None,
        structure=None,  # optional TokenDAG for Ours variant
    ):
        self.base = base_model
        self.head = head
        self.loader = train_loader
        self.cfg = config or CorrectionTrainConfig()
        self.structure = structure
        self.optimizer = torch.optim.AdamW(head.parameters(), lr=self.cfg.lr)
        self.device = next(head.parameters()).device
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()
        self.global_step = 0

    @torch.no_grad()
    def _generate_noisy_clean(
        self,
        x_0: torch.Tensor,
        t_val: float,
        prompt_mask: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Greedy-unmask at noise level *t* to produce a "noisy-clean" sequence.

        Previously ``attention_mask`` was silently dropped here, causing the
        backbone to attend to padding tokens and producing wrong logits for
        padded batches.  It is now forwarded to the base model.
        """
        B = x_0.shape[0]
        t = torch.full((B,), t_val, device=x_0.device)
        x_t = self.base.noise_input(x_0, t)
        # Protect prompt positions from being noised.
        if prompt_mask is not None:
            x_t = torch.where(prompt_mask.bool(), x_0, x_t)
        # Pass attention_mask so padding tokens are ignored by the backbone.
        logits = self.base.forward(x_t, t, attention_mask).logits
        # Prevent the mask token itself from being predicted as output.
        if self.base.mask_token_id < logits.shape[-1]:
            logits[..., self.base.mask_token_id] = -float("inf")
        pred = logits.argmax(dim=-1)
        # Replace only positions that were masked; keep prompt/clean tokens intact.
        is_masked = x_t == self.base.mask_token_id
        return torch.where(is_masked, pred, x_0)

    def _step(self, batch: dict) -> torch.Tensor:
        x_0 = batch["input_ids"].to(self.device)
        prompt_mask = batch.get("prompt_mask")
        if prompt_mask is not None:
            prompt_mask = prompt_mask.to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Sample a noise level uniformly from the configured set.
        idx = torch.randint(0, len(self.cfg.noise_levels), (1,)).item()
        t_val = self.cfg.noise_levels[idx]
        x_noisy = self._generate_noisy_clean(x_0, t_val, prompt_mask, attention_mask)

        # Forward pass through the correction head.
        revise_prob, new_logits = self.head(x_noisy, attention_mask=attention_mask)

        # Binary error labels.
        is_error = (x_noisy != x_0).float()

        # Restrict loss to the answer region (non-prompt, non-padding).
        loss_region = torch.ones_like(is_error, dtype=torch.bool)
        if prompt_mask is not None:
            loss_region = loss_region & ~prompt_mask.bool()
        if attention_mask is not None:
            loss_region = loss_region & attention_mask.bool()

        # BCE on revise_prob
        bce = F.binary_cross_entropy(revise_prob, is_error, reduction="none")
        bce = (bce * loss_region.float()).sum() / loss_region.float().sum().clamp(min=1.0)

        # CE on new_logits — only at positions where the backbone made an error.
        err_mask = (is_error.bool() & loss_region)
        if err_mask.any():
            ce = F.cross_entropy(
                new_logits[err_mask], x_0[err_mask], reduction="mean"
            )
        else:
            ce = torch.tensor(0.0, device=self.device)

        loss = self.cfg.bce_weight * bce + self.cfg.ce_weight * ce

        # Reasoning-aware extension: structural consistency penalty.
        if self.structure is not None and self.cfg.structure_consistency_weight > 0:
            # For each DAG edge (parent -> child): if the child token is already
            # correct but its parent is wrong, the model reached the right child
            # conclusion without a valid premise.  Increase the parent's
            # revise_prob target by adding a loss penalty.
            # Vectorised: parent_has_ok_child[b, p] = any_c( adj[p,c] & ~errs[b,c] )
            adj = self.structure.adjacency  # (L, L) bool
            L = x_0.shape[1]
            if adj.shape[0] == L:
                errs = is_error.bool()  # (B, L)
                adj_f = adj.float().to(self.device)
                child_ok = (~errs).float()  # (B, L)
                parent_has_ok_child = (child_ok @ adj_f.T > 0).float()  # (B, L)
                penalty = (
                    (1 - revise_prob) * parent_has_ok_child * errs.float() * loss_region.float()
                ).sum()
                penalty = penalty / loss_region.float().sum().clamp(min=1.0)
                loss = loss + self.cfg.structure_consistency_weight * penalty

        return loss, {"bce": bce.item(), "ce": ce.item() if isinstance(ce, torch.Tensor) else 0.0}

    def train(self):
        self.head.train()
        for epoch in range(self.cfg.num_epochs):
            for batch in self.loader:
                loss, stats = self._step(batch)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
                self.global_step += 1
                if self.global_step % self.cfg.log_every == 0:
                    logger.info(
                        f"[correction] step {self.global_step} "
                        f"loss={loss.item():.4f} bce={stats['bce']:.4f} ce={stats['ce']:.4f}"
                    )

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"head_state": self.head.state_dict(), "config": self.head.config},
            path,
        )

    def load(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.head.load_state_dict(ckpt["head_state"])
