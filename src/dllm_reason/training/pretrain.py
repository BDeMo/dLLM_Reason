"""Standard dLLM pretraining loop.

Trains a DiffusionLM on text data using the model's native loss function
(ELBO for MDLM, score entropy for SEDD, VLB+CE for D3PM).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dllm_reason.models.base import DiffusionLM
from dllm_reason.utils.logging import get_logger
from dllm_reason.utils.distributed import is_main_process

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    log_every: int = 100
    save_every: int = 5000
    eval_every: int = 2000
    save_dir: str = "checkpoints"
    keep_last_n: int = 2   # rolling: delete older step_*.pt beyond this count
                           #          (best.pt always kept). 0 = keep all.
    use_wandb: bool = False
    wandb_project: str = "dllm-reason"


class Trainer:
    """Standard training loop for discrete diffusion language models."""

    def __init__(
        self,
        model: DiffusionLM,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        config: TrainConfig | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainConfig()

        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_steps,
        )

        self.global_step = 0
        self.best_val_loss = float("inf")

    def train(self):
        """Run the full training loop."""
        cfg = self.config
        self.model.train()
        device = self.model.device

        if cfg.use_wandb and is_main_process():
            import wandb
            wandb.init(project=cfg.wandb_project)

        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        accum_loss = 0.0
        data_iter = iter(self.train_loader)

        while self.global_step < cfg.max_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            x_0 = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Forward + backward
            loss = self.model.compute_loss(x_0, attention_mask)
            loss = loss / cfg.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

            if (self.global_step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            self.global_step += 1

            # Logging
            if self.global_step % cfg.log_every == 0 and is_main_process():
                avg_loss = accum_loss / cfg.log_every
                lr = self.scheduler.get_last_lr()[0]
                logger.info(f"Step {self.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")
                if cfg.use_wandb:
                    import wandb
                    wandb.log({"train/loss": avg_loss, "train/lr": lr}, step=self.global_step)
                accum_loss = 0.0

            # Evaluation. val_loss is rank-local; all-reduce to a global mean
            # so every rank agrees on the best-val decision (else each rank
            # diverges on when to save best.pt → non-deterministic).
            if self.val_loader and self.global_step % cfg.eval_every == 0:
                val_loss = self.evaluate()
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    t = torch.tensor(val_loss, device=self.model.device)
                    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
                    val_loss = t.item()
                if is_main_process():
                    logger.info(f"Step {self.global_step}: val_loss={val_loss:.4f}")
                    if cfg.use_wandb:
                        import wandb
                        wandb.log({"val/loss": val_loss}, step=self.global_step)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    # ALL ranks enter — save_checkpoint is collective-safe
                    # (state_dict() all-gather inside; rank 0 writes to disk)
                    self.save_checkpoint(save_dir / "best.pt")

            # Save step ckpt — call on ALL ranks (collective-safe inside)
            if self.global_step % cfg.save_every == 0:
                self.save_checkpoint(save_dir / f"step_{self.global_step}.pt")

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            x_0 = batch["input_ids"].to(self.model.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.model.device)

            loss = self.model.compute_loss(x_0, attention_mask)
            total_loss += loss.item()
            num_batches += 1

            if num_batches >= 50:
                break

        self.model.train()
        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, path: Path):
        # FSDP-correct: state_dict() is a COLLECTIVE under FULL_SHARD with
        # FULL_STATE_DICT(rank0_only=True) — ALL ranks must call it so the
        # all-gather completes. Only rank 0 holds the actual tensors + writes
        # to disk. Wrapping this whole method in `if is_main_process()`
        # would deadlock under FSDP. DDP path is unaffected (state_dict is
        # rank-local, no collective).
        model_sd = self.model.state_dict()
        if not is_main_process():
            return  # non-main ranks: gathered, nothing to write

        torch.save({
            "model_state_dict": model_sd,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }, path)
        logger.info(f"Saved checkpoint to {path}")
        # Rolling cleanup: keep only the last N step_*.pt (best.pt is
        # untouched). Without this, 8B + fp32 AdamW ckpts (~96 GB each)
        # pile up and eat disk quickly — see minors.zh.md 2026-04-21.
        keep = getattr(self.config, "keep_last_n", 0)
        if keep and path.name.startswith("step_"):
            save_dir = path.parent
            step_ckpts = sorted(
                save_dir.glob("step_*.pt"),
                key=lambda p: int(p.stem.split("_")[1])
                              if p.stem.split("_")[1].isdigit() else 0,
            )
            for old in step_ckpts[:-keep]:
                try:
                    old.unlink()
                    logger.info(f"Rolled off old checkpoint {old.name}")
                except OSError as e:
                    logger.warning(f"Failed to remove {old}: {e}")

    def load_checkpoint(self, path: Path):
        ckpt = torch.load(path, map_location=self.model.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")
