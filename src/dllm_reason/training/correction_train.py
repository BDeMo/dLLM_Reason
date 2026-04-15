"""训练 BackPlay 纠错头 (CorrectionHead).

训练数据自生成：
  1. 从 clean 序列 x_0 开始
  2. 用 diffusion 主干在某个 noise level 下做 greedy unmask，得到"带错"序列 x_noisy_clean
     （主干本身的预测错误就是真实的错误分布，比随机替换更真实）
  3. GT: x_0；输入：x_noisy_clean；label:
       - is_error  = (x_noisy_clean != x_0)  →  BCE target for revise_prob
       - new_token = x_0                      →  CE target for new_logits

Loss = BCE(revise_prob, is_error) + α·CE(new_logits on error positions, x_0)

Reasoning-aware 扩展（贡献 #4）
-------------------------------
可选 `structure: TokenDAG` 参数：训练时若提供，额外加一项
"logical consistency loss"——对于违反 DAG 拓扑的 token 组合（e.g., child
已 unmask 但 parent 还错），加大 revise_prob 目标。

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
    """训练时用到的 noise levels，每个 batch 随机抽一个"""
    ce_weight: float = 1.0
    bce_weight: float = 1.0
    log_every: int = 10
    save_dir: str = "runs/correction_head"
    grad_clip: float = 1.0
    # reasoning-aware 选项
    structure_consistency_weight: float = 0.0
    """>0 时启用 DAG 一致性损失（需传入 structure）"""


class CorrectionTrainer:
    """自生成数据训练 CorrectionHead。

    Frozen: base DiffusionLM（只用它采错 token）
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
        self, x_0: torch.Tensor, t_val: float, prompt_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """在 noise level t 下 greedy unmask 得到"带错"的 clean 序列。"""
        B = x_0.shape[0]
        t = torch.full((B,), t_val, device=x_0.device)
        x_t = self.base.noise_input(x_0, t)
        # 保护 prompt 位置
        if prompt_mask is not None:
            x_t = torch.where(prompt_mask.bool(), x_0, x_t)
        logits = self.base.forward(x_t, t).logits
        # mask 不允许作为输出
        if self.base.mask_token_id < logits.shape[-1]:
            logits[..., self.base.mask_token_id] = -float("inf")
        pred = logits.argmax(dim=-1)
        # 只在 mask 位置替换
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

        # 随机选一个 noise level
        idx = torch.randint(0, len(self.cfg.noise_levels), (1,)).item()
        t_val = self.cfg.noise_levels[idx]
        x_noisy = self._generate_noisy_clean(x_0, t_val, prompt_mask)

        # 前向
        revise_prob, new_logits = self.head(x_noisy, attention_mask=attention_mask)

        # label
        is_error = (x_noisy != x_0).float()

        # 只在 answer 区域算 loss（非 prompt）
        loss_region = torch.ones_like(is_error, dtype=torch.bool)
        if prompt_mask is not None:
            loss_region = loss_region & ~prompt_mask.bool()
        if attention_mask is not None:
            loss_region = loss_region & attention_mask.bool()

        # BCE on revise_prob
        bce = F.binary_cross_entropy(revise_prob, is_error, reduction="none")
        bce = (bce * loss_region.float()).sum() / loss_region.float().sum().clamp(min=1.0)

        # CE on new_logits at error positions only
        err_mask = (is_error.bool() & loss_region)
        if err_mask.any():
            ce = F.cross_entropy(
                new_logits[err_mask], x_0[err_mask], reduction="mean"
            )
        else:
            ce = torch.tensor(0.0, device=self.device)

        loss = self.cfg.bce_weight * bce + self.cfg.ce_weight * ce

        # reasoning-aware 扩展：结构一致性惩罚
        if self.structure is not None and self.cfg.structure_consistency_weight > 0:
            # 简化：对 DAG 上每条 (parent → child) 边，若 child 正确但 parent 错，
            # 说明模型在没有正确前提下猜对了，revise_prob(parent) 目标应更高
            adj = self.structure.adjacency  # (L, L) bool
            L = x_0.shape[1]
            if adj.shape[0] == L:
                errs = is_error.bool()  # (B, L)
                # child 正确：~errs；parent 错：errs
                # 对每个 (p, c) 边：若 ~errs[c] 且 errs[p] → 加惩罚
                # 向量化：parent_err_child_ok[b, p] = any over c of adj[p, c] & ~errs[b, c]
                adj_f = adj.float().to(self.device)
                child_ok = (~errs).float()  # (B, L)
                parent_has_ok_child = (child_ok @ adj_f.T > 0).float()  # (B, L)
                penalty = ((1 - revise_prob) * parent_has_ok_child * errs.float() * loss_region.float()).sum()
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
