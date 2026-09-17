"""Differentiable Optimal Transport alignment and fusion for VQA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class OTConfig:
    transport_type: str = "unbalanced"
    marginal_mode: str = "question_conditioned"
    cost_type: str = "hybrid"
    ot_dim: int = 256
    cost_mix: float = 0.5
    epsilon: float = 0.05
    tau_visual: float = 1.0
    tau_question: float = 1.0
    max_iterations: int = 50
    tolerance: float = 1e-4
    minimum_mass: float = 1e-8
    return_diagnostics: bool = False

    def __post_init__(self) -> None:
        if self.transport_type not in {"balanced", "unbalanced"}:
            raise ValueError("transport_type must be 'balanced' or 'unbalanced'")
        if self.marginal_mode not in {"uniform", "question_conditioned"}:
            raise ValueError("marginal_mode must be 'uniform' or 'question_conditioned'")
        if self.cost_type not in {"cosine", "learned", "hybrid"}:
            raise ValueError("cost_type must be 'cosine', 'learned', or 'hybrid'")
        if self.ot_dim < 1 or self.max_iterations < 1:
            raise ValueError("ot_dim and max_iterations must be positive")
        if not 0.0 <= self.cost_mix <= 1.0:
            raise ValueError("cost_mix must be in [0, 1]")
        if min(self.epsilon, self.tau_visual, self.tau_question,
               self.tolerance, self.minimum_mass) <= 0:
            raise ValueError("OT regularization, tolerance, and minimum mass must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Optional[dict]) -> "OTConfig":
        return cls(**(values or {}))

    @classmethod
    def from_json(cls, path: str | Path) -> "OTConfig":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass
class TransportOutput:
    plan: torch.Tensor
    cost: Optional[torch.Tensor]
    visual_marginal: Optional[torch.Tensor]
    question_marginal: Optional[torch.Tensor]
    fused_tokens: torch.Tensor
    memory_padding_mask: torch.Tensor
    transport_cost: torch.Tensor
    entropy: torch.Tensor
    matched_mass: torch.Tensor
    unmatched_mass: torch.Tensor
    excess_mass: torch.Tensor
    residual: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor
    ot_san: Optional[object] = None


@dataclass
class SinkhornOutput:
    plan: torch.Tensor
    residual: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor


def masked_softmax(logits: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != padding_mask.shape:
        raise ValueError("logits and padding_mask must have equal shapes")
    valid = ~padding_mask
    if not valid.any(dim=1).all():
        raise ValueError("Every example needs at least one valid token")
    masked = logits.float().masked_fill(padding_mask, float("-inf"))
    return torch.softmax(masked, dim=-1).to(logits.dtype).masked_fill(padding_mask, 0.0)


def uniform_marginal(padding_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    valid = ~padding_mask
    counts = valid.sum(dim=1, keepdim=True)
    if (counts == 0).any():
        raise ValueError("Every example needs at least one valid token")
    return valid.to(dtype) / counts.to(dtype)


def sinkhorn_transport(
    cost: torch.Tensor,
    visual_marginal: torch.Tensor,
    question_marginal: torch.Tensor,
    visual_padding_mask: torch.Tensor,
    question_padding_mask: torch.Tensor,
    config: OTConfig,
) -> SinkhornOutput:
    """Solve balanced or KL-relaxed OT using batched log-domain updates."""
    if cost.ndim != 3:
        raise ValueError("cost must have shape [batch, visual_tokens, question_tokens]")
    batch, visual_length, question_length = cost.shape
    expected = ((batch, visual_length), (batch, question_length))
    actual = (visual_marginal.shape, question_marginal.shape)
    if actual != expected or visual_padding_mask.shape != expected[0] or question_padding_mask.shape != expected[1]:
        raise ValueError("Marginal and mask shapes do not match the cost matrix")
    if not torch.isfinite(cost).all():
        raise ValueError("OT cost contains NaN or infinity")
    valid_visual = ~visual_padding_mask
    valid_question = ~question_padding_mask
    if not valid_visual.any(1).all() or not valid_question.any(1).all():
        raise ValueError("Every example needs valid visual and question tokens")

    # The solver stays in float32 even under autocast/mixed precision.
    cost32 = cost.float()
    a = visual_marginal.float().clamp_min(config.minimum_mass)
    b = question_marginal.float().clamp_min(config.minimum_mass)
    pair_valid = valid_visual.unsqueeze(2) & valid_question.unsqueeze(1)
    log_kernel = (-cost32 / config.epsilon).masked_fill(~pair_valid, float("-inf"))
    # Avoid differentiating through logsumexp([-inf, ...]) for a wholly padded
    # row or column. Those dual entries are discarded immediately afterward.
    visual_update_kernel = log_kernel.masked_fill(
        ~valid_visual.unsqueeze(2), 0.0
    )
    question_update_kernel = log_kernel.masked_fill(
        ~valid_question.unsqueeze(1), 0.0
    )
    log_a = a.log().masked_fill(~valid_visual, 0.0)
    log_b = b.log().masked_fill(~valid_question, 0.0)
    log_u = torch.zeros_like(a)
    log_v = torch.zeros_like(b)
    if config.transport_type == "balanced":
        visual_power = question_power = 1.0
    else:
        visual_power = config.tau_visual / (config.tau_visual + config.epsilon)
        question_power = config.tau_question / (config.tau_question + config.epsilon)

    residual = torch.full((batch,), float("inf"), device=cost.device)
    converged = torch.zeros(batch, dtype=torch.bool, device=cost.device)
    iterations = torch.zeros(batch, dtype=torch.long, device=cost.device)
    for step in range(1, config.max_iterations + 1):
        old_u, old_v = log_u, log_v
        visual_lse = torch.logsumexp(visual_update_kernel + log_v.unsqueeze(1), dim=2)
        log_u = visual_power * (log_a - visual_lse)
        log_u = log_u.masked_fill(~valid_visual, 0.0)
        question_lse = torch.logsumexp(question_update_kernel + log_u.unsqueeze(2), dim=1)
        log_v = question_power * (log_b - question_lse)
        log_v = log_v.masked_fill(~valid_question, 0.0)

        delta_u = (log_u - old_u).abs().masked_fill(~valid_visual, 0.0).amax(1)
        delta_v = (log_v - old_v).abs().masked_fill(~valid_question, 0.0).amax(1)
        residual = torch.maximum(delta_u, delta_v)
        newly_converged = (~converged) & residual.le(config.tolerance)
        iterations = torch.where(newly_converged, torch.full_like(iterations, step), iterations)
        converged = converged | newly_converged
        if bool(converged.all()):
            break
    iterations = torch.where(converged, iterations,
                             torch.full_like(iterations, config.max_iterations))

    log_plan = log_u.unsqueeze(2) + log_kernel + log_v.unsqueeze(1)
    plan = torch.exp(log_plan).masked_fill(~pair_valid, 0.0)
    if not torch.isfinite(plan).all() or (plan < 0).any():
        raise FloatingPointError("Sinkhorn produced an invalid transport plan")
    if config.transport_type == "balanced":
        row_error = (plan.sum(2) - visual_marginal.float()).abs().masked_fill(
            ~valid_visual, 0.0
        ).amax(1)
        column_error = (plan.sum(1) - question_marginal.float()).abs().masked_fill(
            ~valid_question, 0.0
        ).amax(1)
        residual = torch.maximum(row_error, column_error)
        converged = residual.le(config.tolerance)
    return SinkhornOutput(plan=plan, residual=residual,
                          iterations=iterations, converged=converged)


class PairwiseCost(nn.Module):
    def __init__(self, ot_dim: int):
        super().__init__()
        self.learned = nn.Sequential(
            nn.Linear(4 * ot_dim, ot_dim),
            nn.GELU(),
            nn.Linear(ot_dim, 1),
        )

    def forward(self, visual: torch.Tensor, question: torch.Tensor,
                config: OTConfig) -> torch.Tensor:
        visual_unit = F.normalize(visual.float(), dim=-1)
        question_unit = F.normalize(question.float(), dim=-1)
        semantic = 1.0 - torch.matmul(visual_unit, question_unit.transpose(1, 2))
        if config.cost_type == "cosine":
            return semantic.to(visual.dtype)
        visual_pairs = visual.unsqueeze(2).expand(-1, -1, question.size(1), -1)
        question_pairs = question.unsqueeze(1).expand(-1, visual.size(1), -1, -1)
        pair_features = torch.cat(
            [visual_pairs, question_pairs, visual_pairs * question_pairs,
             (visual_pairs - question_pairs).abs()], dim=-1
        )
        learned = F.softplus(-self.learned(pair_features).squeeze(-1).float())
        if config.cost_type == "learned":
            return learned.to(visual.dtype)
        return (config.cost_mix * semantic + (1.0 - config.cost_mix) * learned).to(visual.dtype)


class ConditionalMarginal(nn.Module):
    def __init__(self, ot_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(4 * ot_dim, ot_dim),
            nn.GELU(),
            nn.Linear(ot_dim, 1),
        )

    def forward(self, tokens: torch.Tensor, context: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        expanded = context.unsqueeze(1).expand_as(tokens)
        features = torch.cat(
            [tokens, expanded, tokens * expanded, (tokens - expanded).abs()], dim=-1
        )
        return masked_softmax(self.scorer(features).squeeze(-1), padding_mask)


def masked_mean(tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    valid = (~padding_mask).to(tokens.dtype).unsqueeze(-1)
    return (tokens * valid).sum(1) / valid.sum(1).clamp_min(1.0)


class OptimalTransportFusion(nn.Module):
    def __init__(self, visual_dim: int, question_dim: int, model_dim: int,
                 config: OTConfig):
        super().__init__()
        self.config = config
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, config.ot_dim), nn.LayerNorm(config.ot_dim)
        )
        self.question_projection = nn.Sequential(
            nn.Linear(question_dim, config.ot_dim), nn.LayerNorm(config.ot_dim)
        )
        self.visual_marginal = ConditionalMarginal(config.ot_dim)
        self.question_marginal = ConditionalMarginal(config.ot_dim)
        self.pairwise_cost = PairwiseCost(config.ot_dim)
        self.fusion = nn.Sequential(
            nn.Linear(4 * config.ot_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )

    def forward(self, visual_tokens: torch.Tensor, question_tokens: torch.Tensor,
                visual_padding_mask: torch.Tensor,
                question_padding_mask: torch.Tensor,
                return_diagnostics: Optional[bool] = None) -> TransportOutput:
        if visual_tokens.ndim != 3 or question_tokens.ndim != 3:
            raise ValueError("Visual and question tokens must have shape [batch, length, dim]")
        visual = self.visual_projection(visual_tokens)
        question = self.question_projection(question_tokens)
        if self.config.marginal_mode == "uniform":
            a = uniform_marginal(visual_padding_mask, visual.dtype)
            b = uniform_marginal(question_padding_mask, question.dtype)
        else:
            global_question = masked_mean(question, question_padding_mask)
            global_visual = masked_mean(visual, visual_padding_mask)
            a = self.visual_marginal(visual, global_question, visual_padding_mask)
            b = self.question_marginal(question, global_visual, question_padding_mask)
        cost = self.pairwise_cost(visual, question, self.config)
        sinkhorn = sinkhorn_transport(cost, a, b, visual_padding_mask,
                                      question_padding_mask, self.config)
        plan = sinkhorn.plan
        received_mass = plan.sum(1)
        visual_evidence = torch.bmm(plan.transpose(1, 2), visual.float())
        visual_evidence = visual_evidence / received_mass.unsqueeze(-1).clamp_min(
            self.config.minimum_mass
        )
        visual_evidence = visual_evidence.to(question.dtype)
        fusion_features = torch.cat(
            [question, visual_evidence, question * visual_evidence,
             (question - visual_evidence).abs()], dim=-1
        )
        fused = self.fusion(fusion_features).masked_fill(
            question_padding_mask.unsqueeze(-1), 0.0
        )
        plan32, cost32 = plan.float(), cost.float()
        matched = plan32.sum((1, 2))
        transport_cost = (plan32 * cost32).sum((1, 2))
        entropy = -(plan32 * plan32.clamp_min(self.config.minimum_mass).log()).sum((1, 2))
        diagnostics = self.config.return_diagnostics if return_diagnostics is None else return_diagnostics
        return TransportOutput(
            plan=plan,
            cost=cost if diagnostics else None,
            visual_marginal=a if diagnostics else None,
            question_marginal=b if diagnostics else None,
            fused_tokens=fused,
            memory_padding_mask=question_padding_mask,
            transport_cost=transport_cost,
            entropy=entropy,
            matched_mass=matched,
            unmatched_mass=(1.0 - matched).clamp_min(0.0),
            excess_mass=(matched - 1.0).clamp_min(0.0),
            residual=sinkhorn.residual,
            iterations=sinkhorn.iterations,
            converged=sinkhorn.converged,
        )
