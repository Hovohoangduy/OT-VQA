"""Token-level multimodal fusion methods with optional OT augmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional

import torch
from torch import nn

from model.optimal_transport import TransportOutput


@dataclass(frozen=True)
class FusionSpec:
    method: str
    transport: str

    @property
    def uses_ot(self) -> bool:
        return self.transport != "none"


def parse_fusion_spec(name: str) -> FusionSpec:
    if name in {"balanced_ot", "uot"}:
        return FusionSpec(
            method="barycentric",
            transport="balanced" if name == "balanced_ot" else "unbalanced",
        )
    if name.startswith("balanced_ot_"):
        return FusionSpec(name.removeprefix("balanced_ot_"), "balanced")
    if name.startswith("uot_"):
        return FusionSpec(name.removeprefix("uot_"), "unbalanced")
    return FusionSpec(name, "none")


@dataclass(frozen=True)
class BANConfig:
    glimpses: int = 2
    hidden_dim: int = 256
    dropout: float = 0.2

    def __post_init__(self):
        _validate_positive(self.glimpses, self.hidden_dim)
        _validate_dropout(self.dropout)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class MUTANConfig:
    rank: int = 5
    factor_dim: int = 256
    dropout: float = 0.2

    def __post_init__(self):
        _validate_positive(self.rank, self.factor_dim)
        _validate_dropout(self.dropout)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CrossAttentionFusionConfig:
    layers: int = 1
    heads: int = 4
    ffn_hidden: int = 1024
    dropout: float = 0.2

    def __post_init__(self):
        _validate_positive(self.layers, self.heads, self.ffn_hidden)
        _validate_dropout(self.dropout)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class AlignedCrossAttentionConfig:
    layers: int = 1
    heads: int = 4
    ffn_hidden: int = 1024
    dropout: float = 0.2
    gate_init: float = -2.0

    def __post_init__(self):
        _validate_positive(self.layers, self.heads, self.ffn_hidden)
        _validate_dropout(self.dropout)
        if not math.isfinite(self.gate_init):
            raise ValueError("Aligned cross-attention gate_init must be finite")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class QFormerConfig:
    query_tokens: int = 8
    layers: int = 2
    heads: int = 4
    ffn_hidden: int = 512
    dropout: float = 0.2

    def __post_init__(self):
        _validate_positive(self.query_tokens, self.layers, self.heads, self.ffn_hidden)
        _validate_dropout(self.dropout)

    def to_dict(self):
        return asdict(self)


def _validate_positive(*values):
    if any(value < 1 for value in values):
        raise ValueError("Fusion dimensions, ranks, layers, and counts must be positive")


def _validate_dropout(value):
    if not 0.0 <= value < 1.0:
        raise ValueError("Fusion dropout must be in [0, 1)")


def config_for_method(method: str, values: Optional[dict] = None):
    config_types = {
        "ban": BANConfig,
        "mutan": MUTANConfig,
        "cross_attention": CrossAttentionFusionConfig,
        "aligned_cross_attention": AlignedCrossAttentionConfig,
        "qformer": QFormerConfig,
    }
    if method not in config_types:
        return None
    config_type = config_types[method]
    if isinstance(values, config_type):
        return values
    if values is not None and not isinstance(values, dict):
        raise TypeError(f"{method} fusion_config must be a dictionary or {config_type.__name__}")
    return config_type(**(values or {}))


@dataclass
class FusionInput:
    visual_tokens: torch.Tensor
    question_tokens: torch.Tensor
    visual_padding_mask: torch.Tensor
    question_padding_mask: torch.Tensor
    transport: Optional[TransportOutput] = None


@dataclass
class FusionOutput:
    memory: torch.Tensor
    memory_padding_mask: torch.Tensor
    diagnostics: Optional[dict[str, torch.Tensor]] = None


def _validate_inputs(inputs: FusionInput) -> None:
    visual, question = inputs.visual_tokens, inputs.question_tokens
    if visual.ndim != 3 or question.ndim != 3:
        raise ValueError("Fusion tokens must have shape [batch, length, dim]")
    if inputs.visual_padding_mask.shape != visual.shape[:2]:
        raise ValueError("Visual padding mask must match visual tokens")
    if inputs.question_padding_mask.shape != question.shape[:2]:
        raise ValueError("Question padding mask must match question tokens")
    if inputs.visual_padding_mask.dtype != torch.bool or inputs.question_padding_mask.dtype != torch.bool:
        raise ValueError("Fusion padding masks must be Boolean")
    if not (~inputs.visual_padding_mask).any(1).all():
        raise ValueError("Every example needs a valid visual token")
    if not (~inputs.question_padding_mask).any(1).all():
        raise ValueError("Every example needs a valid question token")


def _attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    stable = weights.float().clamp_min(torch.finfo(torch.float32).tiny)
    return -(weights.float() * stable.log()).sum(dim=-1)


class BANFusion(nn.Module):
    def __init__(self, visual_dim: int, question_dim: int, model_dim: int,
                 config: BANConfig):
        super().__init__()
        self.config = config
        self.visual_projection = nn.Linear(visual_dim, model_dim)
        self.question_projection = nn.Linear(question_dim, model_dim)
        self.visual_attention = nn.Linear(model_dim, config.hidden_dim)
        self.question_attention = nn.Linear(model_dim, config.hidden_dim)
        self.glimpse_vectors = nn.Parameter(torch.empty(config.glimpses, config.hidden_dim))
        nn.init.xavier_uniform_(self.glimpse_vectors)
        self.joint = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.GELU(), nn.Dropout(config.dropout),
        )
        self.norm = nn.LayerNorm(model_dim)
        self.ot_prior_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: FusionInput, return_diagnostics=False) -> FusionOutput:
        _validate_inputs(inputs)
        visual = self.visual_projection(inputs.visual_tokens)
        question = self.question_projection(inputs.question_tokens)
        visual_hidden = torch.tanh(self.visual_attention(visual))
        question_hidden = torch.tanh(self.question_attention(question))
        # [B, G, M, N]
        scores = torch.einsum(
            "bnh,bmh,gh->bgmn", visual_hidden, question_hidden, self.glimpse_vectors
        ) / math.sqrt(self.config.hidden_dim)
        if inputs.transport is not None:
            prior = inputs.transport.plan.transpose(1, 2).float()
            prior = prior / prior.sum(-1, keepdim=True).clamp_min(1e-8)
            scores = scores + self.ot_prior_scale * prior.clamp_min(1e-8).log().unsqueeze(1)
        visual_mask = inputs.visual_padding_mask[:, None, None, :]
        weights = torch.softmax(
            scores.float().masked_fill(visual_mask, torch.finfo(torch.float32).min),
            dim=-1,
        ).to(scores.dtype)
        weights = weights.masked_fill(
            inputs.question_padding_mask[:, None, :, None], 0.0
        )
        contexts = torch.einsum("bgmn,bnd->bgmd", weights, visual)
        memory = question
        for glimpse in range(self.config.glimpses):
            update = self.joint(contexts[:, glimpse] * memory)
            memory = self.norm(memory + update).masked_fill(
                inputs.question_padding_mask.unsqueeze(-1), 0.0
            )
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "attention_entropy": _attention_entropy(weights).mean((1, 2)),
                "active_glimpses": torch.tensor(
                    float(self.config.glimpses), device=memory.device
                ),
            }
            if inputs.transport is not None:
                diagnostics["ot_prior_scale"] = self.ot_prior_scale.detach()
        return FusionOutput(memory, inputs.question_padding_mask, diagnostics)


class MUTANFusion(nn.Module):
    def __init__(self, visual_dim: int, question_dim: int, model_dim: int,
                 config: MUTANConfig):
        super().__init__()
        self.config = config
        self.visual_projection = nn.Linear(visual_dim, model_dim)
        self.question_projection = nn.Linear(question_dim, model_dim)
        self.visual_factor = nn.Linear(model_dim, config.rank * config.factor_dim)
        self.question_factor = nn.Linear(model_dim, config.rank * config.factor_dim)
        self.output = nn.Linear(config.factor_dim, model_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, inputs: FusionInput, return_diagnostics=False) -> FusionOutput:
        _validate_inputs(inputs)
        visual = self.visual_projection(inputs.visual_tokens)
        question = self.question_projection(inputs.question_tokens)
        if inputs.transport is None:
            scores = torch.matmul(question, visual.transpose(1, 2)) / math.sqrt(visual.size(-1))
            weights = torch.softmax(
                scores.float().masked_fill(
                    inputs.visual_padding_mask.unsqueeze(1),
                    torch.finfo(torch.float32).min,
                ),
                dim=-1,
            ).to(visual.dtype)
            weights = weights.masked_fill(inputs.question_padding_mask.unsqueeze(-1), 0.0)
        else:
            weights = inputs.transport.plan.transpose(1, 2).to(visual.dtype)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            weights = weights.masked_fill(inputs.question_padding_mask.unsqueeze(-1), 0.0)
        evidence = torch.bmm(weights, visual)
        shape = (*evidence.shape[:2], self.config.rank, self.config.factor_dim)
        visual_factors = self.visual_factor(evidence).reshape(shape)
        question_factors = self.question_factor(question).reshape(shape)
        fused = (self.dropout(visual_factors) * self.dropout(question_factors)).sum(dim=2)
        memory = self.norm(question + self.output(fused)).masked_fill(
            inputs.question_padding_mask.unsqueeze(-1), 0.0
        )
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "attention_entropy": _attention_entropy(weights).mean(1),
                "factor_norm": fused.float().norm(dim=-1).mean(1),
            }
        return FusionOutput(memory, inputs.question_padding_mask, diagnostics)


class _CrossAttentionLayer(nn.Module):
    def __init__(self, model_dim: int, heads: int, ffn_hidden: int, dropout: float):
        super().__init__()
        if model_dim % heads:
            raise ValueError("Cross-attention model_dim must be divisible by heads")
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

    def forward(self, question, visual, visual_mask, question_mask, prior, prior_scale):
        batch, qlen, dim = question.shape
        vlen = visual.size(1)
        q = self.q(question).reshape(batch, qlen, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(visual).reshape(batch, vlen, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(visual).reshape(batch, vlen, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if prior is not None:
            scores = scores + prior_scale * prior.unsqueeze(1)
        scores = scores.float().masked_fill(
            visual_mask[:, None, None, :], torch.finfo(torch.float32).min
        )
        weights = torch.softmax(scores, dim=-1).to(question.dtype)
        weights = weights.masked_fill(question_mask[:, None, :, None], 0.0)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(batch, qlen, dim)
        question = self.norm1(question + self.dropout(self.output(attended)))
        question = self.norm2(question + self.dropout(self.ffn(question)))
        return question.masked_fill(question_mask.unsqueeze(-1), 0.0), weights


class CrossAttentionFusion(nn.Module):
    def __init__(self, visual_dim: int, question_dim: int, model_dim: int,
                 config: CrossAttentionFusionConfig):
        super().__init__()
        if model_dim % config.heads:
            raise ValueError("Cross-attention model_dim must be divisible by heads")
        self.visual_projection = nn.Linear(visual_dim, model_dim)
        self.question_projection = nn.Linear(question_dim, model_dim)
        self.layers = nn.ModuleList([
            _CrossAttentionLayer(
                model_dim, config.heads, config.ffn_hidden, config.dropout
            ) for _ in range(config.layers)
        ])
        self.ot_prior_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: FusionInput, return_diagnostics=False) -> FusionOutput:
        _validate_inputs(inputs)
        visual = self.visual_projection(inputs.visual_tokens)
        question = self.question_projection(inputs.question_tokens)
        prior = None
        if inputs.transport is not None:
            prior = inputs.transport.plan.transpose(1, 2).float()
            prior = prior / prior.sum(-1, keepdim=True).clamp_min(1e-8)
            prior = prior.clamp_min(1e-8).log()
        weights_by_layer = []
        for layer in self.layers:
            question, weights = layer(
                question, visual, inputs.visual_padding_mask,
                inputs.question_padding_mask, prior, self.ot_prior_scale,
            )
            if return_diagnostics:
                weights_by_layer.append(weights)
        diagnostics = None
        if return_diagnostics:
            stacked = torch.stack(weights_by_layer, dim=1)
            diagnostics = {
                "attention_entropy": _attention_entropy(stacked).mean((1, 2, 3)),
            }
            if inputs.transport is not None:
                diagnostics["ot_prior_scale"] = self.ot_prior_scale.detach()
        return FusionOutput(question, inputs.question_padding_mask, diagnostics)


class AlignedCrossAttentionFusion(nn.Module):
    """Cross-attention over raw or softly UOT-grounded question tokens."""

    def __init__(self, visual_dim: int, question_dim: int, model_dim: int,
                 config: AlignedCrossAttentionConfig):
        super().__init__()
        if model_dim % config.heads:
            raise ValueError("Aligned cross-attention model_dim must be divisible by heads")
        self.visual_projection = nn.Linear(visual_dim, model_dim)
        self.question_projection = nn.Linear(question_dim, model_dim)
        self.ot_grounded_projection = nn.Linear(model_dim, model_dim)
        self.ot_gate = nn.Linear(2 * model_dim, 1)
        nn.init.zeros_(self.ot_gate.weight)
        nn.init.constant_(self.ot_gate.bias, config.gate_init)
        self.layers = nn.ModuleList([
            _CrossAttentionLayer(
                model_dim, config.heads, config.ffn_hidden, config.dropout
            ) for _ in range(config.layers)
        ])

    def forward(self, inputs: FusionInput, return_diagnostics=False) -> FusionOutput:
        _validate_inputs(inputs)
        visual = self.visual_projection(inputs.visual_tokens)
        raw_question = self.question_projection(inputs.question_tokens)
        raw_question = raw_question.masked_fill(
            inputs.question_padding_mask.unsqueeze(-1), 0.0
        )
        question = raw_question
        gate = None
        alignment_distance = None
        if inputs.transport is not None:
            grounded = self.ot_grounded_projection(inputs.transport.fused_tokens)
            grounded = grounded.masked_fill(
                inputs.question_padding_mask.unsqueeze(-1), 0.0
            )
            gate = torch.sigmoid(self.ot_gate(torch.cat([raw_question, grounded], dim=-1)))
            gate = gate.masked_fill(inputs.question_padding_mask.unsqueeze(-1), 0.0)
            question = raw_question + gate * (grounded - raw_question)
            question = question.masked_fill(
                inputs.question_padding_mask.unsqueeze(-1), 0.0
            )
            alignment_distance = (question - raw_question).float().norm(dim=-1)

        weights_by_layer = []
        for layer in self.layers:
            question, weights = layer(
                question, visual, inputs.visual_padding_mask,
                inputs.question_padding_mask, None, 0.0,
            )
            if return_diagnostics:
                weights_by_layer.append(weights)

        diagnostics = None
        if return_diagnostics:
            stacked = torch.stack(weights_by_layer, dim=1)
            diagnostics = {
                "attention_entropy": _attention_entropy(stacked).mean((1, 2, 3)),
            }
            if gate is not None:
                valid = (~inputs.question_padding_mask).to(gate.dtype)
                count = valid.sum(1).clamp_min(1.0)
                gate_values = gate.squeeze(-1)
                gate_mean = (gate_values * valid).sum(1) / count
                gate_variance = (
                    (gate_values - gate_mean.unsqueeze(1)).square() * valid
                ).sum(1) / count
                diagnostics.update({
                    "ot_gate_mean": gate_mean,
                    "ot_gate_std": gate_variance.sqrt(),
                    "ot_alignment_distance": (
                        alignment_distance * valid
                    ).sum(1) / count,
                })
        return FusionOutput(question, inputs.question_padding_mask, diagnostics)


class _QFormerLayer(nn.Module):
    def __init__(self, model_dim: int, heads: int, ffn_hidden: int, dropout: float):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_hidden, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, memory, memory_mask):
        attended, _ = self.self_attention(queries, queries, queries, need_weights=False)
        queries = self.norm1(queries + self.dropout(attended))
        attended, weights = self.cross_attention(
            queries, memory, memory, key_padding_mask=memory_mask,
            need_weights=True, average_attn_weights=False,
        )
        queries = self.norm2(queries + self.dropout(attended))
        queries = self.norm3(queries + self.dropout(self.ffn(queries)))
        return queries, weights


class QFormerFusion(nn.Module):
    def __init__(self, visual_dim: int, question_dim: int, model_dim: int,
                 config: QFormerConfig):
        super().__init__()
        if model_dim % config.heads:
            raise ValueError("Q-Former model_dim must be divisible by heads")
        self.query_tokens = nn.Parameter(torch.empty(config.query_tokens, model_dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        self.visual_projection = nn.Linear(visual_dim, model_dim)
        self.question_projection = nn.Linear(question_dim, model_dim)
        self.ot_grounded_projection = nn.Linear(model_dim, model_dim)
        self.layers = nn.ModuleList([
            _QFormerLayer(model_dim, config.heads, config.ffn_hidden, config.dropout)
            for _ in range(config.layers)
        ])

    def forward(self, inputs: FusionInput, return_diagnostics=False) -> FusionOutput:
        _validate_inputs(inputs)
        visual = self.visual_projection(inputs.visual_tokens)
        if inputs.transport is None:
            question_memory = self.question_projection(inputs.question_tokens)
        else:
            question_memory = self.ot_grounded_projection(inputs.transport.fused_tokens)
        memory = torch.cat([visual, question_memory], dim=1)
        memory_mask = torch.cat(
            [inputs.visual_padding_mask, inputs.question_padding_mask], dim=1
        )
        queries = self.query_tokens.unsqueeze(0).expand(memory.size(0), -1, -1)
        weights_by_layer = []
        for layer in self.layers:
            queries, weights = layer(queries, memory, memory_mask)
            if return_diagnostics:
                weights_by_layer.append(weights)
        output_mask = torch.zeros(
            queries.shape[:2], dtype=torch.bool, device=queries.device
        )
        diagnostics = None
        if return_diagnostics:
            stacked = torch.stack(weights_by_layer, dim=1)
            diagnostics = {
                "attention_entropy": _attention_entropy(stacked).mean((1, 2, 3)),
                "query_norm": queries.float().norm(dim=-1).mean(1),
            }
        return FusionOutput(queries, output_mask, diagnostics)


FUSION_REGISTRY = {
    "ban": BANFusion,
    "mutan": MUTANFusion,
    "cross_attention": CrossAttentionFusion,
    "aligned_cross_attention": AlignedCrossAttentionFusion,
    "qformer": QFormerFusion,
}


def build_fusion_module(method: str, visual_dim: int, question_dim: int,
                        model_dim: int, config, uses_ot: Optional[bool] = None):
    if method not in FUSION_REGISTRY:
        return None
    module = FUSION_REGISTRY[method](visual_dim, question_dim, model_dim, config)
    if uses_ot is not None:
        if hasattr(module, "ot_prior_scale"):
            module.ot_prior_scale.requires_grad_(uses_ot)
        if isinstance(module, QFormerFusion):
            module.question_projection.requires_grad_(not uses_ot)
            module.ot_grounded_projection.requires_grad_(uses_ot)
        if isinstance(module, AlignedCrossAttentionFusion):
            module.ot_grounded_projection.requires_grad_(uses_ot)
            module.ot_gate.requires_grad_(uses_ot)
    return module
