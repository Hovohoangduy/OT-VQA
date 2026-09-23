"""Minimal float32 Sinkhorn primitives for the training-only OT teacher."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OTConfig:
    epsilon: float = 0.1
    tau_visual: float = 0.5
    tau_question: float = 0.5
    max_iterations: int = 20
    tolerance: float = 1e-3
    minimum_mass: float = 1e-8

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if min(
            self.epsilon, self.tau_visual, self.tau_question,
            self.tolerance, self.minimum_mass,
        ) <= 0:
            raise ValueError("OT numerical values must be positive")


@dataclass
class SinkhornOutput:
    plan: torch.Tensor
    residual: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor


def uniform_marginal(padding_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    valid = ~padding_mask
    counts = valid.sum(dim=1, keepdim=True)
    if (counts == 0).any():
        raise ValueError("Every example needs at least one valid token")
    return valid.to(dtype) / counts.to(dtype)


def masked_mean(tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    valid = (~padding_mask).to(tokens.dtype).unsqueeze(-1)
    return (tokens * valid).sum(1) / valid.sum(1).clamp_min(1.0)


def sinkhorn_transport(
    cost: torch.Tensor,
    visual_marginal: torch.Tensor,
    question_marginal: torch.Tensor,
    visual_padding_mask: torch.Tensor,
    question_padding_mask: torch.Tensor,
    config: OTConfig,
) -> SinkhornOutput:
    """Solve KL-relaxed Unbalanced OT with float32 log-domain updates."""
    if cost.ndim != 3:
        raise ValueError("cost must have shape [batch, visual, question]")
    batch, visual_length, question_length = cost.shape
    expected = ((batch, visual_length), (batch, question_length))
    if ((visual_marginal.shape, question_marginal.shape) != expected or
            visual_padding_mask.shape != expected[0] or
            question_padding_mask.shape != expected[1]):
        raise ValueError("Marginal and mask shapes do not match the cost matrix")
    if not torch.isfinite(cost).all():
        raise ValueError("OT cost contains NaN or infinity")
    valid_visual = ~visual_padding_mask
    valid_question = ~question_padding_mask
    if not valid_visual.any(1).all() or not valid_question.any(1).all():
        raise ValueError("Every example needs valid visual and question tokens")

    cost32 = cost.float()
    visual_marginal32 = visual_marginal.float()
    question_marginal32 = question_marginal.float()
    a = visual_marginal32.clamp_min(config.minimum_mass)
    b = question_marginal32.clamp_min(config.minimum_mass)
    pair_valid = valid_visual.unsqueeze(2) & valid_question.unsqueeze(1)
    log_kernel = (-cost32 / config.epsilon).masked_fill(~pair_valid, float("-inf"))
    visual_kernel = log_kernel.masked_fill(~valid_visual.unsqueeze(2), 0.0)
    question_kernel = log_kernel.masked_fill(~valid_question.unsqueeze(1), 0.0)
    log_a = a.log().masked_fill(~valid_visual, 0.0)
    log_b = b.log().masked_fill(~valid_question, 0.0)
    log_u = torch.zeros_like(a)
    log_v = torch.zeros_like(b)
    visual_power = config.tau_visual / (config.tau_visual + config.epsilon)
    question_power = config.tau_question / (config.tau_question + config.epsilon)

    residual = torch.full((batch,), float("inf"), device=cost.device)
    converged = torch.zeros(batch, dtype=torch.bool, device=cost.device)
    iterations = torch.zeros(batch, dtype=torch.long, device=cost.device)
    for step in range(1, config.max_iterations + 1):
        old_u, old_v = log_u, log_v
        visual_lse = torch.logsumexp(visual_kernel + log_v.unsqueeze(1), dim=2)
        log_u = (visual_power * (log_a - visual_lse)).masked_fill(~valid_visual, 0.0)
        question_lse = torch.logsumexp(question_kernel + log_u.unsqueeze(2), dim=1)
        log_v = (question_power * (log_b - question_lse)).masked_fill(
            ~valid_question, 0.0
        )
        delta_u = (log_u - old_u).abs().masked_fill(~valid_visual, 0.0).amax(1)
        delta_v = (log_v - old_v).abs().masked_fill(~valid_question, 0.0).amax(1)
        residual = torch.maximum(delta_u, delta_v)
        newly_converged = (~converged) & residual.le(config.tolerance)
        iterations = torch.where(
            newly_converged, torch.full_like(iterations, step), iterations
        )
        converged |= newly_converged
        if bool(converged.all()):
            break
    iterations = torch.where(
        converged, iterations, torch.full_like(iterations, config.max_iterations)
    )
    plan = torch.exp(log_u.unsqueeze(2) + log_kernel + log_v.unsqueeze(1))
    plan = plan.masked_fill(~pair_valid, 0.0)
    if not torch.isfinite(plan).all() or (plan < 0).any():
        raise FloatingPointError("Sinkhorn produced an invalid transport plan")
    return SinkhornOutput(plan, residual, iterations, converged)
