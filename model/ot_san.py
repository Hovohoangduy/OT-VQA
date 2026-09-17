"""Masked stacked attention over Optimal-Transport fused question tokens."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class OTSANConfig:
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.2
    gate_init: float = -2.0

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("OT-SAN hidden_dim must be positive")
        if self.num_layers not in {1, 2}:
            raise ValueError("OT-SAN num_layers must be 1 or 2")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("OT-SAN dropout must be in [0, 1)")
        if not math.isfinite(self.gate_init):
            raise ValueError("OT-SAN gate_init must be finite")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Optional[dict]) -> "OTSANConfig":
        return cls(**(values or {}))


@dataclass
class OTSANOutput:
    memory: torch.Tensor
    memory_padding_mask: torch.Tensor
    summary: torch.Tensor
    attention_weights: Optional[torch.Tensor]
    attention_entropy: Optional[torch.Tensor]
    summary_norm: torch.Tensor
    gate: torch.Tensor


class _MaskedAttentionLayer(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.ff_image = nn.Linear(model_dim, hidden_dim)
        self.ff_context = nn.Linear(model_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ff_attention = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(
            self.ff_image(tokens) + self.ff_context(context).unsqueeze(1)
        )
        scores = self.ff_attention(self.dropout(hidden)).squeeze(-1)
        scores = scores.float().masked_fill(
            padding_mask, torch.finfo(torch.float32).min
        )
        weights = torch.softmax(scores, dim=1).to(tokens.dtype)
        weights = weights.masked_fill(padding_mask, 0.0)
        attended = (weights.unsqueeze(-1) * tokens).sum(dim=1)
        return context + self.dropout(attended), weights


class OTSAN(nn.Module):
    """Add a gated global SAN summary while preserving local OT-fused tokens."""

    def __init__(self, model_dim: int, config: OTSANConfig):
        super().__init__()
        if model_dim < 1:
            raise ValueError("OT-SAN model_dim must be positive")
        self.model_dim = model_dim
        self.config = config
        self.layers = nn.ModuleList([
            _MaskedAttentionLayer(
                model_dim=model_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])
        self.summary_dropout = nn.Dropout(config.dropout)
        self.gate_logit = nn.Parameter(torch.tensor(float(config.gate_init)))

    def forward(
        self,
        fused_tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> OTSANOutput:
        if fused_tokens.ndim != 3:
            raise ValueError("OT-SAN fused_tokens must have shape [batch, length, dim]")
        if padding_mask.ndim != 2 or padding_mask.shape != fused_tokens.shape[:2]:
            raise ValueError("OT-SAN padding_mask must match fused token batch and length")
        if fused_tokens.size(-1) != self.model_dim:
            raise ValueError("OT-SAN fused token dimension must equal model_dim")
        if padding_mask.dtype != torch.bool:
            raise ValueError("OT-SAN padding_mask must have boolean dtype")
        if padding_mask.device != fused_tokens.device:
            raise ValueError("OT-SAN tokens and padding mask must be on the same device")
        valid = ~padding_mask
        if not valid.any(dim=1).all():
            raise ValueError("Every OT-SAN example needs at least one valid token")
        if not torch.isfinite(fused_tokens).all():
            raise ValueError("OT-SAN fused tokens contain NaN or infinity")

        valid_values = valid.to(fused_tokens.dtype).unsqueeze(-1)
        initial_context = (
            (fused_tokens * valid_values).sum(dim=1)
            / valid_values.sum(dim=1).clamp_min(1.0)
        )
        context = initial_context
        weights_by_layer = []
        for layer in self.layers:
            context, weights = layer(fused_tokens, context, padding_mask)
            if return_diagnostics:
                weights_by_layer.append(weights)

        gate = torch.sigmoid(self.gate_logit)
        summary = self.summary_dropout(
            initial_context + gate * (context - initial_context)
        )
        summary_mask = torch.zeros(
            padding_mask.size(0), 1, dtype=torch.bool, device=padding_mask.device
        )
        memory = torch.cat([summary.unsqueeze(1), fused_tokens], dim=1)
        memory_padding_mask = torch.cat([summary_mask, padding_mask], dim=1)

        attention_weights = None
        attention_entropy = None
        if return_diagnostics:
            attention_weights = torch.stack(weights_by_layer, dim=1)
            stable = attention_weights.float().clamp_min(torch.finfo(torch.float32).tiny)
            attention_entropy = -(attention_weights.float() * stable.log()).sum(dim=-1)

        return OTSANOutput(
            memory=memory,
            memory_padding_mask=memory_padding_mask,
            summary=summary,
            attention_weights=attention_weights,
            attention_entropy=attention_entropy,
            summary_norm=summary.float().norm(dim=-1),
            gate=gate,
        )
