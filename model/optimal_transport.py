"""Question-conditioned partial optimal transport for VQA fusion."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def log_sinkhorn(cost, row_marginal, column_marginal, epsilon=0.05, iterations=20):
    """Solve batched entropic OT in FP32; zero marginals denote padded entries."""
    if cost.ndim != 3 or row_marginal.shape != cost.shape[:2] or column_marginal.shape != (cost.size(0), cost.size(2)):
        raise ValueError("Cost and marginal shapes are incompatible")
    if epsilon <= 0 or iterations < 1:
        raise ValueError("epsilon and iterations must be positive")
    cost = cost.float()
    row_marginal = row_marginal.float()
    column_marginal = column_marginal.float()
    if (row_marginal < 0).any() or (column_marginal < 0).any():
        raise ValueError("Marginals must be nonnegative")
    if not torch.allclose(row_marginal.sum(-1), column_marginal.sum(-1), atol=1e-5):
        raise ValueError("Row and column marginals must have equal total mass")
    valid_rows = row_marginal > 0
    valid_columns = column_marginal > 0
    if not valid_rows.any(-1).all() or not valid_columns.any(-1).all():
        raise ValueError("Each sample needs a positive row and column marginal")

    # A finite sentinel gives padded rows/columns finite logsumexp values before
    # torch.where removes them. Their dual variables are -inf thereafter.
    valid_pairs = valid_rows.unsqueeze(-1) & valid_columns.unsqueeze(-2)
    log_kernel = (-cost / epsilon).masked_fill(~valid_pairs, -1e4)
    log_rows = row_marginal.clamp_min(1e-30).log()
    log_columns = column_marginal.clamp_min(1e-30).log()
    log_u = torch.zeros_like(row_marginal).masked_fill(~valid_rows, float("-inf"))
    log_v = torch.zeros_like(column_marginal).masked_fill(~valid_columns, float("-inf"))
    for _ in range(iterations):
        log_u = (log_rows - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)).masked_fill(
            ~valid_rows, float("-inf")
        )
        log_v = (log_columns - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)).masked_fill(
            ~valid_columns, float("-inf")
        )
    log_plan = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
    return log_plan.masked_fill(~valid_pairs, float("-inf")).exp()


class PartialTransportFusion(nn.Module):
    """Align ViT patches with valid BERT tokens and build decoder memory."""

    def __init__(self, text_dim, d_model, epsilon=0.05, iterations=20,
                 dustbin_mass=0.2, dustbin_cost=1.0):
        super().__init__()
        if not 0 <= dustbin_mass < 1:
            raise ValueError("dustbin_mass must be in [0, 1)")
        if epsilon <= 0 or iterations < 1:
            raise ValueError("epsilon and iterations must be positive")
        if not 0 <= dustbin_cost <= 2:
            raise ValueError("dustbin_cost must be in [0, 2]")
        self.epsilon = float(epsilon)
        self.iterations = int(iterations)
        self.dustbin_mass = float(dustbin_mass)
        self.dustbin_cost = float(dustbin_cost)
        self.question_projection = nn.Linear(text_dim, d_model)
        self.grounding_projection = nn.Linear(d_model, d_model)
        self.global_projection = nn.Linear(2 * d_model, d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self.global_norm = nn.LayerNorm(d_model)

    def forward(self, image_tokens, question_tokens, question_mask, question_global=None,
                return_transport=False):
        """Return memory, blocked-memory mask, and optionally OT diagnostics.

        image_tokens: projected patch values [B,K,D], excluding ViT CLS.
        question_mask: True for valid content tokens, False for PAD/specials.
        """
        if image_tokens.ndim != 3 or question_tokens.ndim != 3:
            raise ValueError("Image and question tokens must be rank-three tensors")
        batch, patches, d_model = image_tokens.shape
        if patches < 1 or question_tokens.shape[:2] != question_mask.shape or question_tokens.size(0) != batch:
            raise ValueError("Image or question token shapes are invalid")
        if question_tokens.device != image_tokens.device:
            raise ValueError("Image and question tokens must be on the same device")
        question_mask = question_mask.bool()
        valid_counts = question_mask.sum(-1)
        if (valid_counts == 0).any():
            raise ValueError("Every question needs at least one non-special token")
        question_values = self.question_projection(question_tokens)
        if question_global is None:
            question_global_value = (question_values * question_mask.unsqueeze(-1)).sum(1) / valid_counts.unsqueeze(-1)
        else:
            question_global_value = self.question_projection(question_global)
        image_global_value = image_tokens.mean(1)

        image_cost_values = F.normalize(image_tokens.float(), dim=-1)
        question_cost_values = F.normalize(question_values.float(), dim=-1)
        image_cost_global = F.normalize(image_global_value.float(), dim=-1)
        question_cost_global = F.normalize(question_global_value.float(), dim=-1)
        local_cost = 1 - torch.bmm(image_cost_values, question_cost_values.transpose(1, 2))
        image_to_dustbin = 1 - torch.bmm(image_cost_values, question_cost_global.unsqueeze(-1))
        dustbin_to_question = 1 - torch.bmm(image_cost_global.unsqueeze(1), question_cost_values.transpose(1, 2))
        # This cost controls whether unused dustbin capacity pairs with the
        # other dustbin. A fixed high cost would force the maximum discard rate.
        dustbin_to_dustbin = local_cost.new_full((batch, 1, 1), self.dustbin_cost)
        cost = torch.cat((
            torch.cat((local_cost, image_to_dustbin), dim=-1),
            torch.cat((dustbin_to_question, dustbin_to_dustbin), dim=-1),
        ), dim=-2)

        local_mass = 1 - self.dustbin_mass
        row_marginal = cost.new_full((batch, patches), local_mass / patches)
        row_marginal = torch.cat((row_marginal, cost.new_full((batch, 1), self.dustbin_mass)), dim=-1)
        column_marginal = question_mask.to(cost.dtype) * (local_mass / valid_counts.unsqueeze(-1))
        column_marginal = torch.cat((column_marginal, cost.new_full((batch, 1), self.dustbin_mass)), dim=-1)
        plan = log_sinkhorn(cost, row_marginal, column_marginal, self.epsilon, self.iterations)
        local_plan = plan[:, :patches, :question_tokens.size(1)]
        matched_per_token = local_plan.sum(1)
        aligned = torch.bmm(local_plan.transpose(1, 2).to(image_tokens.dtype), image_tokens)
        aligned = aligned / matched_per_token.clamp_min(1e-8).unsqueeze(-1).to(aligned.dtype)
        gate = (matched_per_token / column_marginal[:, :-1].clamp_min(1e-8)).clamp(0, 1)
        grounded = self.token_norm(question_values + gate.unsqueeze(-1).to(question_values.dtype)
                                   * self.grounding_projection(aligned))
        grounded = grounded.masked_fill(~question_mask.unsqueeze(-1), 0)
        global_memory = self.global_norm(self.global_projection(
            torch.cat((image_global_value, question_global_value), dim=-1)
        )).unsqueeze(1)
        memory = torch.cat((global_memory, grounded), dim=1)
        memory_blocked = torch.cat((torch.zeros((batch, 1), dtype=torch.bool, device=question_mask.device),
                                    ~question_mask), dim=1)
        if not return_transport:
            return memory, memory_blocked
        diagnostics = {
            "plan": plan,
            "matched_mass": local_plan.sum(dim=(1, 2)),
            "image_to_dustbin_mass": plan[:, :patches, -1].sum(1),
            "dustbin_to_question_mass": plan[:, -1, :-1].sum(1),
            "dustbin_to_dustbin_mass": plan[:, -1, -1],
            "row_residual": (plan.sum(-1) - row_marginal).abs().amax(-1),
            "column_residual": (plan.sum(-2) - column_marginal).abs().amax(-1),
        }
        return memory, memory_blocked, diagnostics
