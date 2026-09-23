"""Native Cross-Attention fusion used by the VQA student."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class CrossAttentionFusionConfig:
    layers: int = 1
    heads: int = 4
    ffn_hidden: int = 1024
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if min(self.layers, self.heads, self.ffn_hidden) < 1:
            raise ValueError("Cross-Attention dimensions and layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Cross-Attention dropout must be in [0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Optional[dict]) -> "CrossAttentionFusionConfig":
        if isinstance(values, cls):
            return values
        if values is not None and not isinstance(values, dict):
            raise TypeError("fusion_config must be a dictionary")
        return cls(**(values or {}))


@dataclass
class FusionInput:
    visual_tokens: torch.Tensor
    question_tokens: torch.Tensor
    visual_padding_mask: torch.Tensor
    question_padding_mask: torch.Tensor


@dataclass
class FusionOutput:
    memory: torch.Tensor
    memory_padding_mask: torch.Tensor
    diagnostics: Optional[dict[str, torch.Tensor]] = None
    attention_weights: Optional[torch.Tensor] = None


def _validate_inputs(inputs: FusionInput) -> None:
    visual, question = inputs.visual_tokens, inputs.question_tokens
    if visual.ndim != 3 or question.ndim != 3:
        raise ValueError("Fusion tokens must have shape [batch, length, dim]")
    if inputs.visual_padding_mask.shape != visual.shape[:2]:
        raise ValueError("Visual padding mask must match visual tokens")
    if inputs.question_padding_mask.shape != question.shape[:2]:
        raise ValueError("Question padding mask must match question tokens")
    if (inputs.visual_padding_mask.dtype != torch.bool or
            inputs.question_padding_mask.dtype != torch.bool):
        raise ValueError("Fusion padding masks must be Boolean")
    if not (~inputs.visual_padding_mask).any(1).all():
        raise ValueError("Every example needs a valid visual token")
    if not (~inputs.question_padding_mask).any(1).all():
        raise ValueError("Every example needs a valid question token")


def _attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    stable = weights.float().clamp_min(torch.finfo(torch.float32).tiny)
    return -(weights.float() * stable.log()).sum(dim=-1)


class _CrossAttentionLayer(nn.Module):
    def __init__(self, model_dim: int, heads: int, ffn_hidden: int, dropout: float):
        super().__init__()
        if model_dim % heads:
            raise ValueError("Cross-Attention model_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = model_dim // heads
        self.q = nn.Linear(model_dim, model_dim)
        self.k = nn.Linear(model_dim, model_dim)
        self.v = nn.Linear(model_dim, model_dim)
        self.output = nn.Linear(model_dim, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_hidden, model_dim),
        )
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, question, visual, visual_mask, question_mask):
        batch, question_length, model_dim = question.shape
        visual_length = visual.size(1)
        queries = self.q(question).reshape(
            batch, question_length, self.heads, self.head_dim
        ).transpose(1, 2)
        keys = self.k(visual).reshape(
            batch, visual_length, self.heads, self.head_dim
        ).transpose(1, 2)
        values = self.v(visual).reshape(
            batch, visual_length, self.heads, self.head_dim
        ).transpose(1, 2)
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.float().masked_fill(
            visual_mask[:, None, None, :], torch.finfo(torch.float32).min
        )
        weights = torch.softmax(scores, dim=-1).to(question.dtype)
        weights = weights.masked_fill(question_mask[:, None, :, None], 0.0)
        attended = torch.matmul(weights, values).transpose(1, 2).reshape(
            batch, question_length, model_dim
        )
        question = self.norm1(question + self.dropout(self.output(attended)))
        question = self.norm2(question + self.dropout(self.ffn(question)))
        return question.masked_fill(question_mask.unsqueeze(-1), 0.0), weights


class CrossAttentionFusion(nn.Module):
    """Question tokens query visual patch keys and values."""

    def __init__(self, visual_dim: int, question_dim: int, model_dim: int,
                 config: CrossAttentionFusionConfig):
        super().__init__()
        if model_dim % config.heads:
            raise ValueError("Cross-Attention model_dim must be divisible by heads")
        self.config = config
        self.visual_projection = nn.Linear(visual_dim, model_dim)
        self.question_projection = nn.Linear(question_dim, model_dim)
        self.layers = nn.ModuleList([
            _CrossAttentionLayer(model_dim, config.heads, config.ffn_hidden, config.dropout)
            for _ in range(config.layers)
        ])

    def forward(self, inputs: FusionInput, return_diagnostics: bool = False) -> FusionOutput:
        _validate_inputs(inputs)
        visual = self.visual_projection(inputs.visual_tokens)
        question = self.question_projection(inputs.question_tokens).masked_fill(
            inputs.question_padding_mask.unsqueeze(-1), 0.0
        )
        weights_by_layer = []
        for layer in self.layers:
            question, weights = layer(
                question, visual, inputs.visual_padding_mask,
                inputs.question_padding_mask,
            )
            if return_diagnostics:
                weights_by_layer.append(weights)
        if not return_diagnostics:
            return FusionOutput(question, inputs.question_padding_mask)
        stacked = torch.stack(weights_by_layer, dim=1)
        return FusionOutput(
            question,
            inputs.question_padding_mask,
            diagnostics={
                "attention_entropy": _attention_entropy(stacked).mean((1, 2, 3)),
            },
            attention_weights=stacked[:, -1],
        )
