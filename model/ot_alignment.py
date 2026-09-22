"""Training-only contrastive OT teacher and attention-distillation losses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from model.optimal_transport import OTConfig, masked_mean, sinkhorn_transport, uniform_marginal


@dataclass(frozen=True)
class OTAlignmentConfig:
    """Configuration for a cosine-cost UOT alignment teacher."""

    ot_dim: int = 128
    epsilon: float = 0.1
    tau_visual: float = 0.5
    tau_question: float = 0.5
    max_iterations: int = 20
    tolerance: float = 1e-3
    minimum_mass: float = 1e-8
    negative_count: int = 3
    contrastive_temperature: float = 0.07
    collapse_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if min(self.ot_dim, self.max_iterations, self.negative_count) < 1:
            raise ValueError("OT dimension, iterations, and negative count must be positive")
        if min(
            self.epsilon,
            self.tau_visual,
            self.tau_question,
            self.tolerance,
            self.minimum_mass,
            self.contrastive_temperature,
            self.collapse_tolerance,
        ) <= 0:
            raise ValueError("OT alignment regularization values must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Optional[dict]) -> "OTAlignmentConfig":
        return cls(**(values or {}))

    def transport_config(self) -> OTConfig:
        return OTConfig(
            transport_type="unbalanced",
            marginal_mode="uniform",
            cost_type="cosine",
            ot_dim=self.ot_dim,
            epsilon=self.epsilon,
            tau_visual=self.tau_visual,
            tau_question=self.tau_question,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            minimum_mass=self.minimum_mass,
        )


@dataclass
class PairTransportOutput:
    plan: torch.Tensor
    score: torch.Tensor
    cost: torch.Tensor
    matched_mass: torch.Tensor
    entropy: torch.Tensor
    residual: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor


@dataclass
class OTAlignmentOutput:
    loss: torch.Tensor
    positive_transport: PairTransportOutput
    image_to_question_scores: torch.Tensor
    question_to_image_scores: torch.Tensor
    image_to_question_negative_indices: torch.Tensor
    question_to_image_negative_indices: torch.Tensor
    score_margin: torch.Tensor
    image_to_question_accuracy: torch.Tensor
    question_to_image_accuracy: torch.Tensor
    visual_feature_variance: torch.Tensor
    question_feature_variance: torch.Tensor
    collapsed: torch.Tensor

    def diagnostics(self) -> dict[str, torch.Tensor]:
        transport = self.positive_transport
        return {
            "ot_nce_loss": self.loss.detach(),
            "ot_score_margin": self.score_margin.detach(),
            "ot_i2q_accuracy": self.image_to_question_accuracy.detach(),
            "ot_q2i_accuracy": self.question_to_image_accuracy.detach(),
            "ot_matched_mass": transport.matched_mass.detach().mean(),
            "ot_entropy": transport.entropy.detach().mean(),
            "ot_residual": transport.residual.detach().mean(),
            "ot_iterations": transport.iterations.detach().float().mean(),
            "ot_convergence_rate": transport.converged.detach().float().mean(),
            "ot_visual_variance": self.visual_feature_variance.detach(),
            "ot_question_variance": self.question_feature_variance.detach(),
            "ot_collapsed": self.collapsed.detach().float(),
            "ot_candidate_count": torch.tensor(
                float(self.image_to_question_scores.size(1)),
                device=self.loss.device,
            ),
        }


class AlignmentNegativeQueue:
    """Detached FIFO feature queue used when a mini-batch has too few negatives."""

    def __init__(self, capacity: int = 32):
        if capacity < 1:
            raise ValueError("Negative queue capacity must be positive")
        self.capacity = int(capacity)
        self._visual: list[torch.Tensor] = []
        self._visual_mask: list[torch.Tensor] = []
        self._question: list[torch.Tensor] = []
        self._question_mask: list[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self._visual)

    def enqueue(
        self,
        visual_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        visual_padding_mask: torch.Tensor,
        question_padding_mask: torch.Tensor,
    ) -> None:
        for row in range(visual_tokens.size(0)):
            self._visual.append(visual_tokens[row].detach().cpu())
            self._visual_mask.append(visual_padding_mask[row].detach().cpu())
            self._question.append(question_tokens[row].detach().cpu())
            self._question_mask.append(question_padding_mask[row].detach().cpu())
        overflow = max(len(self) - self.capacity, 0)
        if overflow:
            del self._visual[:overflow]
            del self._visual_mask[:overflow]
            del self._question[:overflow]
            del self._question_mask[:overflow]

    def tensors(self, device: torch.device) -> Optional[tuple[torch.Tensor, ...]]:
        if not self._visual:
            return None
        visual = pad_sequence(self._visual, batch_first=True).to(device)
        visual_mask = pad_sequence(
            self._visual_mask, batch_first=True, padding_value=True
        ).to(device)
        question = pad_sequence(self._question, batch_first=True).to(device)
        question_mask = pad_sequence(
            self._question_mask, batch_first=True, padding_value=True
        ).to(device)
        return visual, question, visual_mask, question_mask

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "visual": self._visual,
            "visual_mask": self._visual_mask,
            "question": self._question,
            "question_mask": self._question_mask,
        }

    def load_state_dict(self, state: Optional[dict]) -> None:
        if not state:
            return
        self.capacity = int(state["capacity"])
        fields = ("visual", "visual_mask", "question", "question_mask")
        if len({len(state[field]) for field in fields}) != 1:
            raise ValueError("Negative queue checkpoint fields have inconsistent lengths")
        self._visual = [value.detach().cpu() for value in state["visual"]]
        self._visual_mask = [value.detach().cpu().bool() for value in state["visual_mask"]]
        self._question = [value.detach().cpu() for value in state["question"]]
        self._question_mask = [value.detach().cpu().bool() for value in state["question_mask"]]


class _AlignmentAdapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        reference = self.layers[0].weight
        tokens = tokens.to(device=reference.device, dtype=reference.dtype)
        return F.normalize(self.layers(tokens), dim=-1)


def _concat_padded(
    current_tokens: torch.Tensor,
    current_mask: torch.Tensor,
    queued_tokens: Optional[torch.Tensor],
    queued_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if queued_tokens is None:
        return current_tokens, current_mask
    length = max(current_tokens.size(1), queued_tokens.size(1))

    def pad(tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        amount = length - tokens.size(1)
        if not amount:
            return tokens, mask
        return F.pad(tokens, (0, 0, 0, amount)), F.pad(mask, (0, amount), value=True)

    current_tokens, current_mask = pad(current_tokens, current_mask)
    queued_tokens, queued_mask = pad(queued_tokens, queued_mask)
    return (
        torch.cat([current_tokens, queued_tokens.to(current_tokens.dtype)]),
        torch.cat([current_mask, queued_mask]),
    )


class OTContrastiveAligner(nn.Module):
    """Learn cross-modal token alignment with hard-negative UOT InfoNCE."""

    def __init__(
        self,
        visual_dim: int,
        question_dim: int,
        config: Optional[OTAlignmentConfig] = None,
    ):
        super().__init__()
        self.config = config or OTAlignmentConfig()
        self.visual_adapter = _AlignmentAdapter(visual_dim, self.config.ot_dim)
        self.question_adapter = _AlignmentAdapter(question_dim, self.config.ot_dim)
        self._ot_config = self.config.transport_config()

    @staticmethod
    def _validate(
        visual_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        visual_padding_mask: torch.Tensor,
        question_padding_mask: torch.Tensor,
    ) -> None:
        if visual_tokens.ndim != 3 or question_tokens.ndim != 3:
            raise ValueError("Alignment tokens must have shape [batch, length, dim]")
        if visual_padding_mask.shape != visual_tokens.shape[:2]:
            raise ValueError("Visual alignment mask has the wrong shape")
        if question_padding_mask.shape != question_tokens.shape[:2]:
            raise ValueError("Question alignment mask has the wrong shape")
        if visual_padding_mask.dtype != torch.bool or question_padding_mask.dtype != torch.bool:
            raise ValueError("Alignment masks must be Boolean")
        if not (~visual_padding_mask).any(1).all() or not (~question_padding_mask).any(1).all():
            raise ValueError("Every alignment example needs valid visual and question tokens")

    def _score_projected_pairs(
        self,
        visual: torch.Tensor,
        question: torch.Tensor,
        visual_mask: torch.Tensor,
        question_mask: torch.Tensor,
        visual_indices: torch.Tensor,
        question_indices: torch.Tensor,
    ) -> PairTransportOutput:
        paired_visual = visual.index_select(0, visual_indices)
        paired_question = question.index_select(0, question_indices)
        paired_visual_mask = visual_mask.index_select(0, visual_indices)
        paired_question_mask = question_mask.index_select(0, question_indices)
        cost = 1.0 - torch.bmm(
            paired_visual.float(), paired_question.float().transpose(1, 2)
        )
        cost = cost.clamp(0.0, 2.0)
        visual_marginal = uniform_marginal(paired_visual_mask, torch.float32)
        question_marginal = uniform_marginal(paired_question_mask, torch.float32)
        sinkhorn = sinkhorn_transport(
            cost,
            visual_marginal,
            question_marginal,
            paired_visual_mask,
            paired_question_mask,
            self._ot_config,
        )
        plan = sinkhorn.plan
        matched_mass = plan.sum((1, 2))
        weighted_cost = (plan * cost).sum((1, 2))
        score = -weighted_cost / matched_mass.clamp_min(self.config.minimum_mass)
        entropy = -(
            plan * plan.clamp_min(self.config.minimum_mass).log()
        ).sum((1, 2))
        return PairTransportOutput(
            plan=plan,
            score=score,
            cost=cost,
            matched_mass=matched_mass,
            entropy=entropy,
            residual=sinkhorn.residual,
            iterations=sinkhorn.iterations,
            converged=sinkhorn.converged,
        )

    def positive_transport(
        self,
        visual_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        visual_padding_mask: torch.Tensor,
        question_padding_mask: torch.Tensor,
    ) -> PairTransportOutput:
        self._validate(
            visual_tokens, question_tokens, visual_padding_mask, question_padding_mask
        )
        visual = self.visual_adapter(visual_tokens)
        question = self.question_adapter(question_tokens)
        indices = torch.arange(visual.size(0), device=visual.device)
        return self._score_projected_pairs(
            visual, question, visual_padding_mask, question_padding_mask,
            indices, indices,
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        visual_padding_mask: torch.Tensor,
        question_padding_mask: torch.Tensor,
        queue: Optional[AlignmentNegativeQueue] = None,
    ) -> OTAlignmentOutput:
        self._validate(
            visual_tokens, question_tokens, visual_padding_mask, question_padding_mask
        )
        batch = visual_tokens.size(0)
        queued = queue.tensors(visual_tokens.device) if queue is not None else None
        if queued is None:
            queued_visual = queued_question = queued_visual_mask = queued_question_mask = None
        else:
            queued_visual, queued_question, queued_visual_mask, queued_question_mask = queued
        all_visual_tokens, all_visual_mask = _concat_padded(
            visual_tokens, visual_padding_mask, queued_visual, queued_visual_mask
        )
        all_question_tokens, all_question_mask = _concat_padded(
            question_tokens, question_padding_mask, queued_question, queued_question_mask
        )
        if all_visual_tokens.size(0) < 2 or all_question_tokens.size(0) < 2:
            raise ValueError("OT contrastive alignment needs at least two examples or a queue")

        visual = self.visual_adapter(all_visual_tokens)
        question = self.question_adapter(all_question_tokens)
        visual_pool = F.normalize(masked_mean(visual, all_visual_mask), dim=-1)
        question_pool = F.normalize(masked_mean(question, all_question_mask), dim=-1)
        current_visual_pool = visual_pool[:batch]
        current_question_pool = question_pool[:batch]
        image_to_question_prefilter = current_visual_pool @ question_pool.transpose(0, 1)
        question_to_image_prefilter = current_question_pool @ visual_pool.transpose(0, 1)
        diagonal = torch.arange(batch, device=visual.device)
        image_to_question_prefilter[diagonal, diagonal] = float("-inf")
        question_to_image_prefilter[diagonal, diagonal] = float("-inf")
        negative_count = min(
            self.config.negative_count,
            all_visual_tokens.size(0) - 1,
            all_question_tokens.size(0) - 1,
        )
        i2q_indices = image_to_question_prefilter.topk(negative_count, dim=1).indices
        q2i_indices = question_to_image_prefilter.topk(negative_count, dim=1).indices

        positive = self._score_projected_pairs(
            visual, question, all_visual_mask, all_question_mask, diagonal, diagonal
        )
        repeated_current = diagonal[:, None].expand(-1, negative_count).reshape(-1)
        image_negatives = self._score_projected_pairs(
            visual,
            question,
            all_visual_mask,
            all_question_mask,
            repeated_current,
            i2q_indices.reshape(-1),
        ).score.reshape(batch, negative_count)
        question_negatives = self._score_projected_pairs(
            visual,
            question,
            all_visual_mask,
            all_question_mask,
            q2i_indices.reshape(-1),
            repeated_current,
        ).score.reshape(batch, negative_count)
        image_scores = torch.cat([positive.score[:, None], image_negatives], dim=1)
        question_scores = torch.cat([positive.score[:, None], question_negatives], dim=1)
        labels = torch.zeros(batch, dtype=torch.long, device=visual.device)
        temperature = self.config.contrastive_temperature
        loss = 0.5 * (
            F.cross_entropy(image_scores / temperature, labels)
            + F.cross_entropy(question_scores / temperature, labels)
        )
        hardest_negative = torch.maximum(
            image_negatives.max(1).values, question_negatives.max(1).values
        )
        visual_values = visual[~all_visual_mask]
        question_values = question[~all_question_mask]
        visual_variance = visual_values.var(0, unbiased=False).mean()
        question_variance = question_values.var(0, unbiased=False).mean()
        all_scores = torch.cat([image_scores, question_scores], dim=1)
        collapsed = (
            (visual_variance <= self.config.collapse_tolerance)
            | (question_variance <= self.config.collapse_tolerance)
            | (all_scores.var(unbiased=False) <= self.config.collapse_tolerance)
        )
        return OTAlignmentOutput(
            loss=loss,
            positive_transport=positive,
            image_to_question_scores=image_scores,
            question_to_image_scores=question_scores,
            image_to_question_negative_indices=i2q_indices,
            question_to_image_negative_indices=q2i_indices,
            score_margin=(positive.score - hardest_negative).mean(),
            image_to_question_accuracy=image_scores.argmax(1).eq(0).float().mean(),
            question_to_image_accuracy=question_scores.argmax(1).eq(0).float().mean(),
            visual_feature_variance=visual_variance,
            question_feature_variance=question_variance,
            collapsed=collapsed,
        )


def ot_attention_distillation_loss(
    transport_plan: torch.Tensor,
    student_attention: torch.Tensor,
    question_padding_mask: torch.Tensor,
    visual_padding_mask: Optional[torch.Tensor] = None,
    minimum_mass: float = 1e-8,
) -> torch.Tensor:
    """KL(stopgrad(UOT patch target) || mean-head Cross-Attention)."""
    if transport_plan.ndim != 3:
        raise ValueError("Transport plan must have shape [batch, visual, question]")
    if student_attention.ndim != 4:
        raise ValueError("Student attention must have shape [batch, heads, question, visual]")
    batch, visual_length, question_length = transport_plan.shape
    if student_attention.shape[0] != batch or student_attention.shape[2:] != (
        question_length, visual_length
    ):
        raise ValueError("Student attention and transport plan shapes do not agree")
    if question_padding_mask.shape != (batch, question_length):
        raise ValueError("Question mask does not match the transport plan")
    if visual_padding_mask is None:
        visual_padding_mask = torch.zeros(
            batch, visual_length, dtype=torch.bool, device=transport_plan.device
        )
    if visual_padding_mask.shape != (batch, visual_length):
        raise ValueError("Visual mask does not match the transport plan")

    target = transport_plan.detach().float().transpose(1, 2)
    target = target.masked_fill(visual_padding_mask[:, None, :], 0.0)
    target = target / target.sum(-1, keepdim=True).clamp_min(minimum_mass)
    student = student_attention.float().mean(1)
    student = student.masked_fill(visual_padding_mask[:, None, :], 0.0)
    student = student / student.sum(-1, keepdim=True).clamp_min(minimum_mass)
    per_token = (
        target
        * (
            target.clamp_min(minimum_mass).log()
            - student.clamp_min(minimum_mass).log()
        )
    ).sum(-1)
    valid_question = ~question_padding_mask
    if not valid_question.any():
        raise ValueError("Attention distillation needs at least one valid question token")
    return per_token.masked_select(valid_question).mean()
