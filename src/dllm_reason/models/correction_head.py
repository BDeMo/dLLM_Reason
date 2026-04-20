"""BackPlay 纠错头 (Correction Head).

Reference
---------
BackPlay: Self-correcting discrete diffusion via post-hoc revision
(arXiv:2601.06428)

动机
----
扩散主干 unmask 后的 token 经常有错。挂一个轻量 Transformer head，周期性
回看已 unmask 的 token 序列，输出两个预测：

  1. revise_prob : (B, L) — 每个位置需要修正的概率
  2. new_logits  : (B, L, V) — 修正后的建议 token 分布

在 sampler 中每 k 步调用一次：把 revise_prob > τ 的位置重置为新 token。

训练信号
--------
训练时人工注错（用 teacher-forcing + 随机替换制造"带错"序列），GT 是原始
clean 序列。loss = BCE(revise_prob, is_error) + CE(new_logits, x_0)。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CorrectionHeadConfig:
    vocab_size: int
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    max_seq_len: int = 2048
    dropout: float = 0.0


class CorrectionHead(nn.Module):
    """轻量 Transformer 挂在扩散主干输出上，预测 (revise_prob, new_logits)。

    独立模块：可以单独训练、单独加载。不修改主干。
    """

    def __init__(self, config: CorrectionHeadConfig):
        super().__init__()
        self.config = config
        D = config.hidden_dim

        self.token_embed = nn.Embedding(config.vocab_size, D)
        self.pos_embed = nn.Embedding(config.max_seq_len, D)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=config.num_heads,
            dim_feedforward=4 * D,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=config.num_layers)

        self.norm = nn.LayerNorm(D)
        self.revise_head = nn.Linear(D, 1)
        self.token_head = nn.Linear(D, config.vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L) 当前 token ids（已 unmask 的序列）
            attention_mask: (B, L) bool，True 表示有效位置

        Returns:
            revise_prob: (B, L) ∈ [0,1]，需要修正的概率
            new_logits:  (B, L, V)，修正后的 token 分布
        """
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.token_embed(x) + self.pos_embed(pos)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)

        revise_logit = self.revise_head(h).squeeze(-1)  # (B, L)
        revise_prob = torch.sigmoid(revise_logit)
        new_logits = self.token_head(h)                 # (B, L, V)
        return revise_prob, new_logits

    @torch.no_grad()
    def revise(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
        protect_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """推理期调用：对 revise_prob > threshold 的位置回填 argmax(new_logits)。

        Args:
            x: (B, L) 当前序列
            threshold: revise 阈值
            protect_mask: (B, L) bool，True 表示受保护（prompt/mask/unchanged）
            attention_mask: (B, L) bool
        """
        revise_prob, new_logits = self.forward(x, attention_mask=attention_mask)
        new_tokens = new_logits.argmax(dim=-1)
        to_revise = revise_prob > threshold
        if protect_mask is not None:
            to_revise = to_revise & ~protect_mask.bool()
        return torch.where(to_revise, new_tokens, x)
