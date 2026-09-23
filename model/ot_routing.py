"""Question-conditioned evidence routing with semi-relaxed Optimal Transport."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class OTEvidenceRoutingConfig:
    slots: int = 4
    reasoning_steps: int = 2
    routing_dim: int = 256
    heads: int = 4
    dropout: float = 0.2
    epsilon: float = 0.1
    tau: float = 0.5
    sinkhorn_iterations: int = 40
    diagnostic_tolerance: float = 1e-3
    preference_smoothing: float = 0.05
    null_min: float = 0.02
    null_max: float = 0.25
    shared_step_weights: bool = True
    visual_preference: str = "question_conditioned"
    memory_mode: str = "slots"
    preference_transform: str = "softmax"
    preference_topk: int = 32
    cost_scale_mode: str = "fixed"
    cost_scale: float = 1.0
    cost_scale_min: float = 1.0
    cost_scale_max: float = 20.0
    question_conditioned_keys: bool = False
    routed_gate_max: float = 4.0

    def __post_init__(self) -> None:
        if min(
            self.slots, self.reasoning_steps, self.routing_dim,
            self.heads, self.sinkhorn_iterations,
        ) < 1:
            raise ValueError("Routing dimensions, steps, and iterations must be positive")
        if self.routing_dim % self.heads:
            raise ValueError("routing_dim must be divisible by routing heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("routing dropout must be in [0, 1)")
        if self.epsilon <= 0 or self.tau < 0 or self.diagnostic_tolerance <= 0:
            raise ValueError("epsilon/tolerance must be positive and tau nonnegative")
        if not 0.0 <= self.preference_smoothing < 1.0:
            raise ValueError("preference_smoothing must be in [0, 1)")
        if not 0.0 < self.null_min < self.null_max < 1.0:
            raise ValueError("null preference bounds must satisfy 0 < min < max < 1")
        if self.visual_preference not in {"question_conditioned", "uniform"}:
            raise ValueError("visual_preference must be question_conditioned or uniform")
        if self.memory_mode not in {"slots", "routed_patches"}:
            raise ValueError("memory_mode must be slots or routed_patches")
        if self.preference_transform not in {"softmax", "sparsemax", "topk"}:
            raise ValueError("preference_transform must be softmax, sparsemax, or topk")
        if self.preference_topk < 1:
            raise ValueError("preference_topk must be positive")
        if self.cost_scale_mode not in {"fixed", "learned"}:
            raise ValueError("cost_scale_mode must be fixed or learned")
        if self.cost_scale <= 0 or self.cost_scale_min <= 0:
            raise ValueError("Cost scales must be positive")
        if self.cost_scale_max <= self.cost_scale_min:
            raise ValueError("cost_scale_max must exceed cost_scale_min")
        if self.cost_scale_mode == "learned" and not (
            self.cost_scale_min < self.cost_scale < self.cost_scale_max
        ):
            raise ValueError("Learned cost_scale must lie strictly inside its bounds")
        if self.routed_gate_max <= 0:
            raise ValueError("routed_gate_max must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Optional[dict]) -> "OTEvidenceRoutingConfig":
        if isinstance(values, cls):
            return values
        if values is not None and not isinstance(values, dict):
            raise TypeError("routing_config must be a dictionary")
        return cls(**(values or {}))


@dataclass
class SemiRelaxedTransportOutput:
    plan: torch.Tensor
    row_residual: torch.Tensor
    fixed_point_residual: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor


@dataclass
class EvidenceRoutingOutput:
    memory: torch.Tensor
    memory_padding_mask: torch.Tensor
    diagnostics: Optional[dict[str, torch.Tensor]] = None
    attention_weights: Optional[torch.Tensor] = None
    auxiliary_loss: Optional[torch.Tensor] = None


def _validate_transport_inputs(cost, slot_marginal, evidence_preference, evidence_mask):
    if cost.ndim != 3:
        raise ValueError("cost must have shape [batch, slots, evidence]")
    batch, slots, evidence = cost.shape
    if slot_marginal.shape != (batch, slots):
        raise ValueError("slot_marginal must match cost batch and slot dimensions")
    if evidence_preference.shape != (batch, evidence):
        raise ValueError("evidence_preference must match cost batch and evidence dimensions")
    if evidence_mask.shape != (batch, evidence) or evidence_mask.dtype != torch.bool:
        raise ValueError("evidence_mask must be a Boolean mask matching the evidence")
    if not (~evidence_mask).any(1).all():
        raise ValueError("Every example needs at least one valid evidence token")
    if not torch.isfinite(cost).all():
        raise ValueError("Routing cost contains NaN or infinity")
    if not torch.isfinite(slot_marginal).all() or not torch.isfinite(evidence_preference).all():
        raise ValueError("Routing marginals contain NaN or infinity")
    if (slot_marginal <= 0).any():
        raise ValueError("Every slot must have positive transport mass")
    if (evidence_preference.masked_fill(evidence_mask, 0) < 0).any():
        raise ValueError("Evidence preference cannot be negative")
    ones = torch.ones(batch, device=cost.device, dtype=torch.float32)
    if not torch.allclose(slot_marginal.float().sum(-1), ones, atol=2e-3, rtol=2e-3):
        raise ValueError("Slot marginal must sum to one")
    valid_preference = evidence_preference.masked_fill(evidence_mask, 0).float()
    if not torch.allclose(valid_preference.sum(-1), ones, atol=2e-3, rtol=2e-3):
        raise ValueError("Evidence preference must sum to one over valid entries")


def independent_softmax_transport(
    cost: torch.Tensor,
    slot_marginal: torch.Tensor,
    evidence_mask: torch.Tensor,
    epsilon: float,
) -> SemiRelaxedTransportOutput:
    """The exact tau=0 limit: independent entropy-regularized routing rows."""
    dummy_preference = (~evidence_mask).to(cost.dtype)
    dummy_preference = dummy_preference / dummy_preference.sum(-1, keepdim=True)
    _validate_transport_inputs(cost, slot_marginal, dummy_preference, evidence_mask)
    with torch.autocast(device_type=cost.device.type, enabled=False):
        logits = (-cost.float() / epsilon).masked_fill(
            evidence_mask[:, None, :], float("-inf")
        )
        plan = torch.softmax(logits, dim=-1) * slot_marginal.float().unsqueeze(-1)
        plan = plan.masked_fill(evidence_mask[:, None, :], 0.0)
        row_error = (plan.sum(-1) - slot_marginal.float()).abs().amax(-1)
    batch = cost.size(0)
    zeros = torch.zeros(batch, device=cost.device, dtype=torch.float32)
    return SemiRelaxedTransportOutput(
        plan=plan,
        row_residual=row_error,
        fixed_point_residual=zeros,
        iterations=torch.zeros(batch, device=cost.device, dtype=torch.long),
        converged=row_error <= 1e-6,
    )


def semi_relaxed_sinkhorn(
    cost: torch.Tensor,
    slot_marginal: torch.Tensor,
    evidence_preference: torch.Tensor,
    evidence_mask: torch.Tensor,
    epsilon: float = 0.1,
    tau: float = 0.5,
    iterations: int = 20,
    tolerance: float = 1e-3,
    minimum_mass: float = 1e-8,
) -> SemiRelaxedTransportOutput:
    """Solve hard-row, KL-relaxed-column entropic OT in the log domain."""
    _validate_transport_inputs(cost, slot_marginal, evidence_preference, evidence_mask)
    if epsilon <= 0 or tau < 0 or iterations < 1 or tolerance <= 0 or minimum_mass <= 0:
        raise ValueError("Invalid semi-relaxed transport hyperparameters")
    if tau == 0:
        return independent_softmax_transport(
            cost, slot_marginal, evidence_mask, epsilon
        )

    with torch.autocast(device_type=cost.device.type, enabled=False):
        cost32 = cost.float()
        a = slot_marginal.float()
        b = evidence_preference.float().masked_fill(evidence_mask, 0.0)
        b = b / b.sum(-1, keepdim=True).clamp_min(minimum_mass)
        log_a = a.clamp_min(minimum_mass).log()
        log_b = b.clamp_min(minimum_mass).log().masked_fill(evidence_mask, 0.0)
        # A finite floor keeps derivatives through entirely masked columns
        # well-defined. Returned padded plan entries are still forced to zero.
        log_floor = torch.finfo(torch.float32).min / 4
        log_kernel = (-cost32 / epsilon).masked_fill(
            evidence_mask[:, None, :], log_floor
        )
        log_v = torch.zeros_like(b)
        power = tau / (tau + epsilon)
        residual = torch.full(
            (cost.size(0),), float("inf"), device=cost.device, dtype=torch.float32
        )
        for _ in range(iterations):
            old_v = log_v
            log_u = log_a - torch.logsumexp(log_kernel + log_v[:, None, :], dim=-1)
            column_lse = torch.logsumexp(log_kernel + log_u[:, :, None], dim=1)
            log_v = (power * (log_b - column_lse)).masked_fill(evidence_mask, 0.0)
            residual = (log_v - old_v).abs().masked_fill(evidence_mask, 0.0).amax(-1)

        # The final row update makes the hard slot marginal accurate without a
        # post-hoc normalization that would hide a broken fixed point.
        log_u = log_a - torch.logsumexp(log_kernel + log_v[:, None, :], dim=-1)
        log_plan = log_kernel + log_u[:, :, None] + log_v[:, None, :]
        plan = torch.exp(log_plan).masked_fill(evidence_mask[:, None, :], 0.0)
        row_error = (plan.sum(-1) - a).abs().amax(-1)
        finite = torch.isfinite(plan).all((1, 2)) & torch.isfinite(residual)
        converged = finite & (row_error <= tolerance) & (residual <= tolerance)
    return SemiRelaxedTransportOutput(
        plan=plan,
        row_residual=row_error,
        fixed_point_residual=residual,
        iterations=torch.full(
            (cost.size(0),), iterations, device=cost.device, dtype=torch.long
        ),
        converged=converged,
    )


def _masked_mean(tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    valid = (~padding_mask).to(tokens.dtype).unsqueeze(-1)
    return (tokens * valid).sum(1) / valid.sum(1).clamp_min(1.0)


def _masked_sparsemax(logits: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Sparsemax over the last dimension with exact zeros on padding."""
    if logits.shape != padding_mask.shape or padding_mask.dtype != torch.bool:
        raise ValueError("Sparsemax mask must be Boolean and match logits")
    valid = ~padding_mask
    if not valid.any(-1).all():
        raise ValueError("Sparsemax requires at least one valid value per row")
    floor = torch.finfo(logits.dtype).min
    shifted = logits - logits.masked_fill(padding_mask, floor).amax(
        dim=-1, keepdim=True
    )
    shifted = shifted.masked_fill(padding_mask, floor)
    sorted_values = shifted.sort(dim=-1, descending=True).values
    cumulative = sorted_values.cumsum(dim=-1)
    ranks = torch.arange(
        1, logits.size(-1) + 1, device=logits.device, dtype=logits.dtype
    ).view(*([1] * (logits.ndim - 1)), -1)
    support = (
        (1 + ranks * sorted_values > cumulative)
        & (ranks <= valid.sum(dim=-1, keepdim=True))
    )
    support_size = support.sum(dim=-1, keepdim=True).clamp_min(1)
    threshold = (
        cumulative.gather(-1, support_size - 1) - 1
    ) / support_size.to(logits.dtype)
    output = (shifted - threshold).clamp_min(0).masked_fill(padding_mask, 0.0)
    return output / output.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class _ReasoningUpdate(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.input_projection = nn.Linear(2 * dim + 5, dim)
        self.recurrent = nn.GRUCell(dim, dim)
        self.slot_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.question_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * dim, dim), nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, slots, evidence, spatial_moments, type_mass, question, question_mask):
        batch, count, dim = slots.shape
        question_summary = _masked_mean(question, question_mask)
        update_input = self.input_projection(torch.cat([
            evidence,
            spatial_moments,
            type_mass,
            question_summary.unsqueeze(1).expand(-1, count, -1),
        ], dim=-1))
        slots = self.recurrent(
            update_input.reshape(batch * count, dim),
            slots.reshape(batch * count, dim),
        ).reshape(batch, count, dim)
        attended, _ = self.slot_attention(slots, slots, slots, need_weights=False)
        slots = self.norm1(slots + attended)
        attended, _ = self.question_attention(
            slots, question, question, key_padding_mask=question_mask,
            need_weights=False,
        )
        slots = self.norm2(slots + attended)
        return self.norm3(slots + self.ffn(slots))


class OTEvidenceRouter(nn.Module):
    """Route all visual evidence through question-conditioned reasoning slots."""

    def __init__(
        self,
        visual_dim: int,
        question_dim: int,
        model_dim: int,
        config: OTEvidenceRoutingConfig,
        routing_mode: str = "ot",
    ):
        super().__init__()
        if routing_mode not in {"ot", "softmax"}:
            raise ValueError("routing_mode must be 'ot' or 'softmax'")
        self.config = config
        self.routing_mode = routing_mode
        dim = config.routing_dim
        self.visual_projection = nn.Linear(visual_dim, dim)
        self.question_projection = nn.Linear(question_dim, dim)
        self.question_pool_projection = nn.Linear(dim, dim)
        self.slot_embeddings = nn.Parameter(torch.empty(config.slots, dim))
        nn.init.normal_(self.slot_embeddings, std=0.02)
        self.slot_norm = nn.LayerNorm(dim)
        self.type_embedding = nn.Embedding(3, dim)
        self.position_projection = nn.Sequential(
            nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.null_token = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.null_token, std=0.02)
        self.cost_query = nn.Linear(dim, dim, bias=False)
        self.cost_key = nn.Linear(dim, dim, bias=False)
        if config.question_conditioned_keys:
            self.key_scale = nn.Linear(dim, dim)
            self.key_shift = nn.Linear(dim, dim)
        else:
            self.key_scale = self.key_shift = None
        if config.cost_scale_mode == "learned":
            position = (
                (config.cost_scale - config.cost_scale_min)
                / (config.cost_scale_max - config.cost_scale_min)
            )
            raw_scale = math.log(position / (1.0 - position))
            self.raw_cost_scale = nn.Parameter(torch.tensor(raw_scale))
        else:
            self.register_parameter("raw_cost_scale", None)
        self.preference_evidence = nn.Linear(dim, dim)
        self.preference_question = nn.Linear(dim, dim)
        self.preference_score = nn.Linear(dim, 1)
        self.null_preference = nn.Linear(dim, 1)
        update_count = 1 if config.shared_step_weights else config.reasoning_steps
        self.updates = nn.ModuleList([
            _ReasoningUpdate(dim, config.heads, config.dropout)
            for _ in range(update_count)
        ])
        self.output_projection = (
            nn.Identity() if dim == model_dim else nn.Linear(dim, model_dim)
        )
        if config.memory_mode == "routed_patches":
            self.question_output_projection = (
                nn.Identity() if dim == model_dim else nn.Linear(dim, model_dim)
            )
            self.routed_value_norm = nn.LayerNorm(dim)
            self.routed_role_norm = nn.LayerNorm(dim)
            self.routed_scale = nn.Linear(dim, dim)
            self.routed_shift = nn.Linear(dim, dim)
            self.routed_output_projection = (
                nn.Identity() if dim == model_dim else nn.Linear(dim, model_dim)
            )
        else:
            self.question_output_projection = None
            self.routed_value_norm = self.routed_role_norm = None
            self.routed_scale = self.routed_shift = None
            self.routed_output_projection = None
        self.runtime_tau = float(config.tau)
        self.plan_intervention = None

    def set_tau(self, value: float) -> None:
        if value < 0:
            raise ValueError("Routing tau must be nonnegative")
        self.runtime_tau = float(value)

    def set_plan_intervention(self, value: Optional[str]) -> None:
        if value not in {None, "shuffle_evidence"}:
            raise ValueError("Unknown routing-plan intervention")
        self.plan_intervention = value

    def _intervene_on_plan(
        self, plan: torch.Tensor, evidence_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.plan_intervention is None:
            return plan
        changed = plan.clone()
        # Keep global/null mass fixed and cyclically permute only spatial evidence.
        for batch_index in range(plan.size(0)):
            valid = (~evidence_mask[batch_index, :-2]).nonzero(
                as_tuple=False
            ).squeeze(-1)
            if valid.numel() > 1:
                changed[batch_index, :, valid] = plan[
                    batch_index, :, valid.roll(1)
                ]
        return changed

    def _cost_scale(self) -> torch.Tensor:
        if self.raw_cost_scale is None:
            return self.slot_embeddings.new_tensor(self.config.cost_scale)
        span = self.config.cost_scale_max - self.config.cost_scale_min
        return self.config.cost_scale_min + span * torch.sigmoid(self.raw_cost_scale)

    def _initialize_slots(self, question_summary: torch.Tensor) -> torch.Tensor:
        """Combine question context with role-preserving slot templates.

        Normalizing only after adding the old small slot embeddings to the
        question made every initial slot almost identical.  Normalize the two
        sources independently so learned slot roles remain visible to the first
        transport solve.
        """
        dim = self.config.routing_dim
        templates = F.layer_norm(self.slot_embeddings, (dim,))
        context = F.layer_norm(question_summary, (dim,))
        return self.slot_norm(
            (templates.unsqueeze(0) + context.unsqueeze(1)) / math.sqrt(2.0)
        )

    @staticmethod
    def _positions(length: int, grid_size, device, dtype) -> torch.Tensor:
        if grid_size is None:
            side = math.isqrt(length)
            if side * side != length:
                raise ValueError(
                    "Spatial-token grid metadata is required when token count is not square"
                )
            height = width = side
        else:
            height, width = int(grid_size[0]), int(grid_size[1])
            if height * width != length:
                raise ValueError(
                    f"Visual grid {height}x{width} does not match {length} spatial tokens"
                )
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    def _build_evidence(self, visual_tokens, visual_mask, grid_size):
        batch, length, _ = visual_tokens.shape
        visual = self.visual_projection(visual_tokens)
        positions = self._positions(
            length, grid_size, visual.device, visual.dtype
        ).unsqueeze(0).expand(batch, -1, -1)
        type_embeddings = self.type_embedding.weight.to(visual.dtype)
        spatial = visual + type_embeddings[0] + self.position_projection(positions)
        spatial = spatial.masked_fill(visual_mask.unsqueeze(-1), 0.0)
        global_token = _masked_mean(visual, visual_mask).unsqueeze(1)
        global_token = global_token + type_embeddings[1]
        global_mask = visual_mask.all(1, keepdim=True)
        global_token = global_token.masked_fill(global_mask.unsqueeze(-1), 0.0)
        null = (
            self.null_token.to(visual.dtype).expand(batch, -1, -1)
            + type_embeddings[2]
        )
        evidence = torch.cat([spatial, global_token, null], dim=1)
        evidence_mask = torch.cat([
            visual_mask,
            global_mask,
            torch.zeros(batch, 1, dtype=torch.bool, device=visual.device),
        ], dim=1)
        evidence_positions = torch.cat([
            positions,
            torch.zeros(batch, 2, 2, device=visual.device, dtype=visual.dtype),
        ], dim=1)
        return evidence, evidence_mask, evidence_positions

    def _preference(self, evidence, evidence_mask, question_summary):
        valid_non_null = (~evidence_mask[:, :-1]).to(evidence.dtype)
        has_non_null = valid_non_null.sum(-1, keepdim=True).gt(0)
        if self.config.visual_preference == "uniform":
            non_null = valid_non_null / valid_non_null.sum(-1, keepdim=True).clamp_min(1)
            null = torch.full(
                (evidence.size(0), 1),
                (self.config.null_min + self.config.null_max) / 2,
                device=evidence.device, dtype=evidence.dtype,
            )
        else:
            hidden = torch.tanh(
                self.preference_evidence(evidence[:, :-1])
                + self.preference_question(question_summary).unsqueeze(1)
            )
            logits = self.preference_score(hidden).squeeze(-1).masked_fill(
                evidence_mask[:, :-1], float("-inf")
            )
            safe_logits = logits.masked_fill(~has_non_null, 0.0)
            if self.config.preference_transform == "sparsemax":
                learned = torch.zeros_like(safe_logits)
                if has_non_null.any():
                    active = has_non_null.squeeze(-1)
                    learned[active] = _masked_sparsemax(
                        safe_logits[active].float(), evidence_mask[active, :-1]
                    ).to(device=learned.device, dtype=learned.dtype)
            elif self.config.preference_transform == "topk":
                count = min(self.config.preference_topk, safe_logits.size(-1))
                indices = safe_logits.masked_fill(
                    evidence_mask[:, :-1], torch.finfo(safe_logits.dtype).min
                ).topk(count, dim=-1).indices
                selected = torch.zeros_like(evidence_mask[:, :-1])
                selected.scatter_(1, indices, True)
                selected &= ~evidence_mask[:, :-1]
                selected_logits = safe_logits.masked_fill(~selected, float("-inf"))
                selected_logits = selected_logits.masked_fill(~has_non_null, 0.0)
                learned = torch.softmax(selected_logits.float(), dim=-1).to(
                    evidence.dtype
                )
            else:
                learned = torch.softmax(safe_logits.float(), dim=-1).to(evidence.dtype)
            learned = learned * valid_non_null
            learned = learned / learned.sum(-1, keepdim=True).clamp_min(1e-8)
            uniform = valid_non_null / valid_non_null.sum(-1, keepdim=True).clamp_min(1)
            smooth = self.config.preference_smoothing
            non_null = (1.0 - smooth) * learned + smooth * uniform
            null = self.config.null_min + (
                self.config.null_max - self.config.null_min
            ) * torch.sigmoid(self.null_preference(question_summary))
        null = torch.where(has_non_null, null, torch.ones_like(null))
        non_null = non_null * (1.0 - null)
        return torch.cat([non_null, null], dim=-1).masked_fill(evidence_mask, 0.0)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        visual_padding_mask: torch.Tensor,
        question_padding_mask: torch.Tensor,
        grid_size=None,
        return_diagnostics: bool = False,
    ) -> EvidenceRoutingOutput:
        if visual_tokens.ndim != 3 or question_tokens.ndim != 3:
            raise ValueError("Routing tokens must have shape [batch, length, dim]")
        if visual_padding_mask.shape != visual_tokens.shape[:2]:
            raise ValueError("Visual padding mask must match visual tokens")
        if question_padding_mask.shape != question_tokens.shape[:2]:
            raise ValueError("Question padding mask must match question tokens")
        if visual_padding_mask.dtype != torch.bool or question_padding_mask.dtype != torch.bool:
            raise ValueError("Routing padding masks must be Boolean")
        if not (~question_padding_mask).any(1).all():
            raise ValueError("Every example needs a valid question token")

        reference = self.slot_embeddings
        visual_tokens = visual_tokens.to(reference.device, reference.dtype)
        question_tokens = question_tokens.to(reference.device, reference.dtype)
        visual_padding_mask = visual_padding_mask.to(reference.device)
        question_padding_mask = question_padding_mask.to(reference.device)
        question = self.question_projection(question_tokens).masked_fill(
            question_padding_mask.unsqueeze(-1), 0.0
        )
        question_summary = self.question_pool_projection(
            _masked_mean(question, question_padding_mask)
        )
        slots = self._initialize_slots(question_summary)
        evidence, evidence_mask, positions = self._build_evidence(
            visual_tokens, visual_padding_mask, grid_size
        )
        preference = self._preference(evidence, evidence_mask, question_summary)
        budget = torch.full(
            (visual_tokens.size(0), self.config.slots),
            1.0 / self.config.slots,
            device=slots.device, dtype=slots.dtype,
        )
        step_stats = []
        diversity_losses = []
        final_plan = None
        cost_scale = self._cost_scale()
        for step in range(self.config.reasoning_steps):
            queries = F.normalize(self.cost_query(slots), dim=-1)
            key_evidence = evidence
            if self.key_scale is not None:
                scale = 0.5 * torch.tanh(self.key_scale(question_summary)).unsqueeze(1)
                shift = self.key_shift(question_summary).unsqueeze(1)
                key_evidence = evidence * (1.0 + scale) + shift
            keys = F.normalize(self.cost_key(key_evidence), dim=-1)
            query_similarity = torch.matmul(queries, queries.transpose(1, 2))
            if self.config.slots > 1:
                off_diagonal = ~torch.eye(
                    self.config.slots, dtype=torch.bool, device=queries.device
                ).unsqueeze(0)
                diversity_losses.append(
                    query_similarity.masked_select(off_diagonal).square().mean()
                )
            else:
                diversity_losses.append(queries.sum() * 0.0)
            cost = -cost_scale * torch.matmul(queries, keys.transpose(-1, -2))
            if self.routing_mode == "softmax" or self.runtime_tau == 0:
                # Keep preference parameters in the DDP graph when column
                # coupling is disabled. They receive exact zero gradients.
                cost = cost + preference.sum(-1)[:, None, None] * 0.0
            if self.routing_mode == "softmax":
                transport = independent_softmax_transport(
                    cost, budget, evidence_mask, self.config.epsilon
                )
            else:
                transport = semi_relaxed_sinkhorn(
                    cost, budget, preference, evidence_mask,
                    epsilon=self.config.epsilon,
                    tau=self.runtime_tau,
                    iterations=self.config.sinkhorn_iterations,
                    tolerance=self.config.diagnostic_tolerance,
                )
            plan = transport.plan.to(evidence.dtype)
            plan = self._intervene_on_plan(plan, evidence_mask)
            real_plan = plan[:, :, :-1]
            evidence_readout = torch.matmul(real_plan, evidence[:, :-1]) / budget.unsqueeze(-1)
            spatial_moments = torch.matmul(
                real_plan, positions[:, :-1]
            ) / budget.unsqueeze(-1)
            spatial_mass = real_plan[:, :, :-1].sum(-1) / budget
            global_mass = real_plan[:, :, -1] / budget
            null_mass = plan[:, :, -1] / budget
            type_mass = torch.stack([spatial_mass, global_mass, null_mass], dim=-1)
            update = self.updates[0 if self.config.shared_step_weights else step]
            slots = update(
                slots, evidence_readout, spatial_moments, type_mass,
                question, question_padding_mask,
            )
            final_plan = plan
            if return_diagnostics:
                normalized_rows = plan.float() / budget.float().unsqueeze(-1)
                entropy = -(
                    normalized_rows
                    * normalized_rows.clamp_min(torch.finfo(torch.float32).tiny).log()
                ).sum(-1).mean(-1)
                normalized_assignments = F.normalize(normalized_rows, dim=-1)
                similarity = torch.matmul(
                    normalized_assignments, normalized_assignments.transpose(1, 2)
                )
                off_diagonal = (
                    similarity.sum((1, 2)) - similarity.diagonal(dim1=1, dim2=2).sum(1)
                ) / max(self.config.slots * (self.config.slots - 1), 1)
                column = plan.float().sum(1)
                pref32 = preference.float().clamp_min(1e-8)
                column_kl = (
                    column * (column.clamp_min(1e-8).log() - pref32.log())
                    - column + pref32
                ).masked_fill(evidence_mask, 0.0).sum(-1)
                coverage = (
                    (column[:, :-1] > (0.1 / max(evidence.size(1) - 1, 1))).float()
                    * (~evidence_mask[:, :-1]).float()
                ).sum(-1) / (~evidence_mask[:, :-1]).sum(-1).clamp_min(1)
                step_stats.append({
                    "query_similarity": query_similarity.masked_select(
                        ~torch.eye(
                            self.config.slots,
                            dtype=torch.bool,
                            device=queries.device,
                        ).unsqueeze(0)
                    ).reshape(visual_tokens.size(0), -1).mean(-1)
                    if self.config.slots > 1 else torch.zeros(
                        visual_tokens.size(0),
                        device=queries.device,
                        dtype=queries.dtype,
                    ),
                    "entropy": entropy,
                    "similarity": off_diagonal,
                    "null": null_mass.float().mean(-1),
                    "column_kl": column_kl,
                    "row_residual": transport.row_residual,
                    "fixed_point_residual": transport.fixed_point_residual,
                    "finite": torch.isfinite(plan).all((1, 2)).float(),
                    "converged": transport.converged.float(),
                    "iterations": transport.iterations.float(),
                    "cost_mean": cost.float().mean((1, 2)),
                    "cost_std": cost.float().std((1, 2), unbiased=False),
                    "coverage": coverage,
                    "cost_scale": cost_scale.expand(visual_tokens.size(0)),
                    "cost_to_epsilon": (
                        cost.float().std((1, 2), unbiased=False)
                        / self.config.epsilon
                    ),
                })

        slot_memory = self.output_projection(slots)
        slot_mask = torch.zeros(
            slot_memory.shape[:2], dtype=torch.bool, device=slot_memory.device
        )
        routed_stats = {}
        if self.config.memory_mode == "routed_patches":
            spatial_count = visual_tokens.size(1)
            spatial_plan = final_plan[:, :, :spatial_count]
            column_mass = spatial_plan.sum(1)
            valid_count = (~visual_padding_mask).sum(-1, keepdim=True).to(
                column_mass.dtype
            ).clamp_min(1)
            raw_gate = valid_count * column_mass
            gate = raw_gate.clamp(max=self.config.routed_gate_max).masked_fill(
                visual_padding_mask, 0.0
            )
            role = torch.matmul(spatial_plan.transpose(1, 2), slots)
            role = role / column_mass.unsqueeze(-1).clamp_min(1e-8)
            role = self.routed_role_norm(role)
            base = self.routed_value_norm(evidence[:, :spatial_count])
            film_scale = 0.5 * torch.tanh(self.routed_scale(role))
            film_shift = self.routed_shift(role)
            routed = self.routed_output_projection(
                base * (1.0 + film_scale) + film_shift
            )
            routed = (routed * gate.unsqueeze(-1)).masked_fill(
                visual_padding_mask.unsqueeze(-1), 0.0
            )
            question_memory = self.question_output_projection(question).masked_fill(
                question_padding_mask.unsqueeze(-1), 0.0
            )
            memory = torch.cat([question_memory, slot_memory, routed], dim=1)
            memory_mask = torch.cat([
                question_padding_mask,
                slot_mask,
                visual_padding_mask,
            ], dim=1)
            routed_stats = {
                "routing_gate_mean": (
                    gate.sum(-1) / (~visual_padding_mask).sum(-1).clamp_min(1)
                ).detach(),
                "routing_gate_max": gate.amax(-1).detach(),
                "routing_gate_clipped": (
                    (raw_gate > self.config.routed_gate_max)
                    & ~visual_padding_mask
                ).float().sum(-1).div(
                    (~visual_padding_mask).sum(-1).clamp_min(1)
                ).detach(),
                "routing_spatial_mass": column_mass.sum(-1).detach(),
            }
        else:
            memory = slot_memory
            memory_mask = slot_mask
        diagnostics = None
        if return_diagnostics:
            diagnostics = {}
            for key in step_stats[0]:
                values = torch.stack([row[key] for row in step_stats], dim=1)
                diagnostics[f"routing_{key}"] = values.mean(1).detach()
                diagnostics[f"routing_final_{key}"] = values[:, -1].detach()
            preference32 = preference.float()
            preference_entropy = -(
                preference32
                * preference32.clamp_min(torch.finfo(torch.float32).tiny).log()
            ).sum(-1)
            final_column = final_plan.float().sum(1)
            column_entropy = -(
                final_column
                * final_column.clamp_min(torch.finfo(torch.float32).tiny).log()
            ).sum(-1)
            top_count = min(
                self.config.preference_topk, max(final_column.size(-1) - 1, 1)
            )
            top_mass = final_column[:, :-1].topk(top_count, dim=-1).values.sum(-1)
            diagnostics.update({
                "routing_preference_entropy": preference_entropy.detach(),
                "routing_preference_effective_support": preference_entropy.exp().detach(),
                "routing_column_entropy": column_entropy.detach(),
                "routing_column_effective_support": column_entropy.exp().detach(),
                "routing_topk_mass": top_mass.detach(),
                "routing_global_mass": final_column[:, -2].detach(),
                "routing_runtime_tau": torch.full_like(
                    preference_entropy, self.runtime_tau
                ),
                **routed_stats,
            })
        return EvidenceRoutingOutput(
            memory=memory,
            memory_padding_mask=memory_mask,
            diagnostics=diagnostics,
            attention_weights=final_plan.detach() if return_diagnostics else None,
            auxiliary_loss=torch.stack(diversity_losses).mean(),
        )
