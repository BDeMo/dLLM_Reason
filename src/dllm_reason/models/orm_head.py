"""ORM (Outcome Reward Model) head for LLaDA.

Architecture (Cobbe 2021 / V-STaR style, simplified):
  - frozen base model gives last_hidden_state for input (prompt + output)
  - linear head reads the last token's hidden state → 1 scalar logit
  - sigmoid(logit) = P(output is correct given prompt)

Training:
  loss = BCEWithLogitsLoss(logit, label)  where label ∈ {0, 1}

Inference (BoN):
  for each (prompt, sample_i): logit_i = head(forward(prompt, sample_i))
  pick output where logit_i is max.

References:
  Cobbe et al. 2021 "Training Verifiers to Solve Math Word Problems"
    arXiv:2110.14168 — the original ORM head architecture.
  Hosseini et al. 2024 "V-STaR: Training Verifiers for Self-Taught
    Reasoners" arXiv:2402.06457 — joint pos/neg from self-distill.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ORMHead(nn.Module):
    """Single-layer linear head over last-token hidden state.

    Args:
        hidden_size: model's hidden dim (4096 for LLaDA-8B)
        pooling: "last" (last non-pad token) or "mean" (over output tokens)
    """

    def __init__(self, hidden_size: int, pooling: str = "last",
                 dropout: float = 0.0):
        super().__init__()
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                output_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, L, H) from base model
            attention_mask: (B, L) 1=real token, 0=pad
            output_mask:    (B, L) 1=output region (where we want to score),
                            0=prompt region. If None, treat all non-pad as
                            output.
        Returns:
            logits: (B,) — pre-sigmoid score per sample
        """
        if self.pooling == "last":
            # Last non-pad token across each row
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1).long() - 1   # (B,)
            else:
                lengths = torch.full((hidden_states.shape[0],),
                                     hidden_states.shape[1] - 1,
                                     device=hidden_states.device,
                                     dtype=torch.long)
            # gather last hidden per row
            B, L, H = hidden_states.shape
            idx = lengths.view(B, 1, 1).expand(B, 1, H)
            pooled = hidden_states.gather(1, idx).squeeze(1)     # (B, H)
        elif self.pooling == "mean":
            mask = output_mask if output_mask is not None else attention_mask
            if mask is None:
                pooled = hidden_states.mean(dim=1)
            else:
                m = mask.float().unsqueeze(-1)                   # (B, L, 1)
                pooled = (hidden_states * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            raise ValueError(f"unknown pooling: {self.pooling}")

        pooled = self.dropout(pooled)
        return self.classifier(pooled).squeeze(-1)              # (B,)


class ORMWrapper(nn.Module):
    """Wraps a frozen base model + ORM head.

    Forward returns the head logits given input_ids + attention_mask.
    The base is run in no_grad in inference; in training only the head's
    weights have requires_grad=True.
    """

    def __init__(self, base_model: nn.Module, hidden_size: int,
                 pooling: str = "last", dropout: float = 0.0,
                 freeze_base: bool = True):
        super().__init__()
        self.base = base_model
        self.head = ORMHead(hidden_size, pooling=pooling, dropout=dropout)
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                output_mask: torch.Tensor | None = None,
                output_hidden_states: bool = True) -> torch.Tensor:
        # Many HF models return hidden via output_hidden_states=True.
        # LLaDA's modeling_llada has `last_hidden_state` accessible too.
        with torch.set_grad_enabled(any(p.requires_grad for p in self.base.parameters())):
            out = self.base(input_ids, attention_mask=attention_mask,
                            output_hidden_states=output_hidden_states)
        # output_hidden_states list[..., (B, L, H)] last layer = -1
        hidden = out.hidden_states[-1] if hasattr(out, "hidden_states") and out.hidden_states else out.last_hidden_state
        return self.head(hidden, attention_mask=attention_mask,
                         output_mask=output_mask)
