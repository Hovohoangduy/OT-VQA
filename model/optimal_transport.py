"""Question-conditioned partial optimal transport for VQA fusion."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def compute_fgw_cost(local_cost, patches, num_q_tokens, plan, alpha=0.2, device=None, dtype=None, question_mask=None):
    """Compute Fused Gromov-Wasserstein cost combining feature cost and 2D spatial-structural distortion."""
    if alpha <= 0 or plan is None:
        return local_cost
    side = math.isqrt(patches)
    if side * side != patches:
        return local_cost
    device = device or local_cost.device
    dtype = dtype or local_cost.dtype

    # 2D visual distance matrix D_I [patches, patches]
    axis = torch.linspace(0, 1, side, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    d_i = torch.cdist(coords, coords) / math.sqrt(2.0)

    # Question sequence distance matrix D_Q [num_q_tokens, num_q_tokens]
    if question_mask is None:
        question_mask = torch.ones(plan.size(0), num_q_tokens, device=device, dtype=torch.bool)
    # Normalize valid-token ranks per sample, independent of batch padding.
    positions = (question_mask.long().cumsum(-1) - 1).clamp_min(0).to(dtype)
    positions = positions / (question_mask.sum(-1, keepdim=True) - 1).clamp_min(1)
    d_q = (positions.unsqueeze(-1) - positions.unsqueeze(-2)).abs()

    d_i_sq = d_i.pow(2)
    d_q_sq = d_q.pow(2)

    plan_det = plan.detach()
    term1 = torch.bmm(d_i_sq.unsqueeze(0).expand(plan_det.size(0), -1, -1), plan_det).sum(-1, keepdim=True)
    term2 = torch.bmm(plan_det, d_q_sq).sum(-2, keepdim=True)
    term3 = torch.bmm(torch.bmm(d_i.unsqueeze(0).expand(plan_det.size(0), -1, -1), plan_det),
                      d_q)

    c_gw = (term1 + term2 - 2 * term3).clamp_min(0.0)
    return (1.0 - alpha) * local_cost + alpha * c_gw


def log_sinkhorn(cost, row_marginal, column_marginal, epsilon=0.05, iterations=20, tol=1e-4, tau=None):
    with torch.autocast(device_type=cost.device.type, enabled=False):
        return _log_sinkhorn_fp32(cost, row_marginal, column_marginal, epsilon, iterations, tol, tau)


def _log_sinkhorn_fp32(cost, row_marginal, column_marginal, epsilon, iterations, tol, tau):
    """Solve batched entropic OT in FP32; supports both balanced and unbalanced (tau < inf)."""
    if cost.ndim != 3 or row_marginal.shape != cost.shape[:2] or column_marginal.shape != (cost.size(0), cost.size(2)):
        raise ValueError("Cost and marginal shapes are incompatible")
    if tau is not None and (not math.isfinite(tau) or tau <= 0):
        raise ValueError("unbalanced tau must be finite and positive")
    if not math.isfinite(epsilon) or epsilon <= 0 or iterations < 1:
        raise ValueError("epsilon and iterations must be positive")
    cost = cost.float()
    row_marginal = row_marginal.float()
    column_marginal = column_marginal.float()
    if (row_marginal < 0).any() or (column_marginal < 0).any():
        raise ValueError("Marginals must be nonnegative")
    if tau is None and not torch.allclose(row_marginal.sum(-1), column_marginal.sum(-1), atol=1e-5):
        raise ValueError("Row and column marginals must have equal total mass")
    valid_rows = row_marginal > 0
    valid_columns = column_marginal > 0
    if not valid_rows.any(-1).all() or not valid_columns.any(-1).all():
        raise ValueError("Each sample needs a positive row and column marginal")

    reg = (tau / (tau + epsilon)) if (tau is not None and tau > 0) else 1.0

    # A finite sentinel gives padded rows/columns finite logsumexp values before
    # torch.where removes them. Their dual variables are -inf thereafter.
    valid_pairs = valid_rows.unsqueeze(-1) & valid_columns.unsqueeze(-2)
    log_kernel = (-cost / epsilon).masked_fill(~valid_pairs, -1e4)
    log_rows = row_marginal.clamp_min(1e-30).log()
    log_columns = column_marginal.clamp_min(1e-30).log()
    log_u = torch.zeros_like(row_marginal).masked_fill(~valid_rows, float("-inf"))
    log_v = torch.zeros_like(column_marginal).masked_fill(~valid_columns, float("-inf"))
    for step in range(iterations):
        log_u = reg * (log_rows - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)).masked_fill(
            ~valid_rows, float("-inf")
        )
        log_v = reg * (log_columns - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)).masked_fill(
            ~valid_columns, float("-inf")
        )
        if tol is not None and step >= 4 and tau is None:
            log_plan_cur = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
            plan_cur = log_plan_cur.masked_fill(~valid_pairs, float("-inf")).exp()
            row_err = (plan_cur.sum(-1) - row_marginal).abs().amax()
            if bool(row_err < tol):
                break
    log_plan = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
    return log_plan.masked_fill(~valid_pairs, float("-inf")).exp()


class PartialTransportFusion(nn.Module):
    """Align spatial image tokens with valid question tokens and build decoder memory."""

    def __init__(self, text_dim, d_model, epsilon=0.05, iterations=20,
                 dustbin_mass=0.2, dustbin_cost=1.0, visual_memory=False,
                 num_heads=1, tol=1e-4, fgw_alpha=0.0, unbalanced_tau=None):
        super().__init__()
        if not 0 <= dustbin_mass < 1:
            raise ValueError("dustbin_mass must be in [0, 1)")
        if epsilon <= 0 or iterations < 1:
            raise ValueError("epsilon and iterations must be positive")
        if not 0 <= dustbin_cost <= 2:
            raise ValueError("dustbin_cost must be in [0, 2]")
        if num_heads < 1 or d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.epsilon = float(epsilon)
        self.iterations = int(iterations)
        self.dustbin_mass = float(dustbin_mass)
        self.dustbin_cost = float(dustbin_cost)
        self.num_heads = int(num_heads)
        self.head_dim = d_model // num_heads
        self.tol = tol
        self.fgw_alpha = float(fgw_alpha)
        self.unbalanced_tau = float(unbalanced_tau) if unbalanced_tau is not None else None
        self.question_projection = nn.Linear(text_dim, d_model)
        self.grounding_projection = nn.Linear(d_model, d_model)
        self.global_projection = nn.Linear(2 * d_model, d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self.global_norm = nn.LayerNorm(d_model)
        self.visual_memory = visual_memory
        if visual_memory:
            self.image_grounding_projection = nn.Linear(d_model, d_model)
            self.image_position_projection = nn.Linear(2, d_model, bias=False)
            self.image_norm = nn.LayerNorm(d_model)

    def forward(self, image_tokens, question_tokens, question_mask, question_global=None,
                return_transport=False):
        with torch.autocast(device_type=image_tokens.device.type, enabled=False):
            return self._forward_fp32(image_tokens.float(), question_tokens.float(), question_mask,
                                     None if question_global is None else question_global.float(), return_transport)

    def _forward_fp32(self, image_tokens, question_tokens, question_mask, question_global,
                      return_transport):
        """Return memory, blocked-memory mask, and optionally OT diagnostics.

        image_tokens: projected spatial values [B,K,D], excluding a CLS token when present.
        question_mask: True for valid content tokens, False for PAD/specials.
        """
        if image_tokens.ndim != 3 or question_tokens.ndim != 3:
            raise ValueError("Image and question tokens must be rank-three tensors")
        batch, patches, d_model = image_tokens.shape
        num_q_tokens = question_tokens.size(1)
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

        H = self.num_heads
        dh = self.head_dim
        if H > 1:
            v = image_tokens.view(batch, patches, H, dh).transpose(1, 2).reshape(batch * H, patches, dh)
            q = question_values.view(batch, num_q_tokens, H, dh).transpose(1, 2).reshape(batch * H, num_q_tokens, dh)
            v_glob = image_global_value.view(batch, H, dh).reshape(batch * H, dh)
            q_glob = question_global_value.view(batch, H, dh).reshape(batch * H, dh)
            eff_batch = batch * H
            mask_eff = question_mask.unsqueeze(1).expand(-1, H, -1).reshape(eff_batch, -1)
            valid_counts_eff = valid_counts.unsqueeze(1).expand(-1, H).reshape(eff_batch)
        else:
            v = image_tokens
            q = question_values
            v_glob = image_global_value
            q_glob = question_global_value
            eff_batch = batch
            mask_eff = question_mask
            valid_counts_eff = valid_counts

        image_cost_values = F.normalize(v.float(), dim=-1)
        question_cost_values = F.normalize(q.float(), dim=-1)
        image_cost_global = F.normalize(v_glob.float(), dim=-1)
        question_cost_global = F.normalize(q_glob.float(), dim=-1)
        local_cost = 1 - torch.bmm(image_cost_values, question_cost_values.transpose(1, 2))
        image_to_dustbin = 1 - torch.bmm(image_cost_values, question_cost_global.unsqueeze(-1))
        dustbin_to_question = 1 - torch.bmm(image_cost_global.unsqueeze(1), question_cost_values.transpose(1, 2))
        dustbin_to_dustbin = local_cost.new_full((eff_batch, 1, 1), self.dustbin_cost)
        cost = torch.cat((
            torch.cat((local_cost, image_to_dustbin), dim=-1),
            torch.cat((dustbin_to_question, dustbin_to_dustbin), dim=-1),
        ), dim=-2)

        local_mass = 1 - self.dustbin_mass
        row_marginal = cost.new_full((eff_batch, patches), local_mass / patches)
        row_marginal = torch.cat((row_marginal, cost.new_full((eff_batch, 1), self.dustbin_mass)), dim=-1)
        column_marginal = mask_eff.to(cost.dtype) * (local_mass / valid_counts_eff.unsqueeze(-1))
        column_marginal = torch.cat((column_marginal, cost.new_full((eff_batch, 1), self.dustbin_mass)), dim=-1)
        plan = log_sinkhorn(cost, row_marginal, column_marginal, self.epsilon, self.iterations,
                            tol=self.tol, tau=self.unbalanced_tau)
        if self.fgw_alpha > 0:
            local_cost = compute_fgw_cost(
                local_cost, patches, num_q_tokens, plan[:, :patches, :num_q_tokens],
                alpha=self.fgw_alpha, device=cost.device, dtype=cost.dtype, question_mask=mask_eff
            )
            cost = torch.cat((
                torch.cat((local_cost, image_to_dustbin), dim=-1),
                torch.cat((dustbin_to_question, dustbin_to_dustbin), dim=-1),
            ), dim=-2)
            plan = log_sinkhorn(cost, row_marginal, column_marginal, self.epsilon,
                                max(self.iterations // 2, 5), tol=self.tol, tau=self.unbalanced_tau)
        local_plan = plan[:, :patches, :num_q_tokens]
        matched_per_token = local_plan.sum(1)
        aligned_h = torch.bmm(local_plan.transpose(1, 2).to(v.dtype), v)
        aligned_h = aligned_h / matched_per_token.clamp_min(1e-8).unsqueeze(-1).to(aligned_h.dtype)
        gate_h = (matched_per_token / column_marginal[:, :-1].clamp_min(1e-8)).clamp(0, 1)

        if H > 1:
            aligned = aligned_h.view(batch, H, num_q_tokens, dh).transpose(1, 2).reshape(batch, num_q_tokens, d_model)
            gate = gate_h.view(batch, H, num_q_tokens).mean(dim=1)
        else:
            aligned = aligned_h
            gate = gate_h

        grounded = self.token_norm(question_values + gate.unsqueeze(-1).to(question_values.dtype)
                                   * self.grounding_projection(aligned))
        grounded = grounded.masked_fill(~question_mask.unsqueeze(-1), 0)
        global_memory = self.global_norm(self.global_projection(
            torch.cat((image_global_value, question_global_value), dim=-1)
        )).unsqueeze(1)
        memory = torch.cat((global_memory, grounded), dim=1)
        memory_blocked = torch.cat((torch.zeros((batch, 1), dtype=torch.bool, device=question_mask.device),
                                    ~question_mask), dim=1)
        if self.visual_memory:
            matched_per_patch = local_plan.sum(-1)
            aligned_questions_h = torch.bmm(local_plan.to(q.dtype), q)
            aligned_questions_h = aligned_questions_h / matched_per_patch.clamp_min(1e-8).unsqueeze(-1)
            image_gate_h = (matched_per_patch / row_marginal[:, :-1].clamp_min(1e-8)).clamp(0, 1)
            if H > 1:
                aligned_questions = aligned_questions_h.view(batch, H, patches, dh).transpose(1, 2).reshape(batch, patches, d_model)
                image_gate = image_gate_h.view(batch, H, patches).mean(dim=1)
            else:
                aligned_questions = aligned_questions_h
                image_gate = image_gate_h

            side = math.isqrt(patches)
            if side * side != patches:
                raise ValueError('Visual memory requires a square spatial token grid')
            axis = torch.linspace(-1, 1, side, device=image_tokens.device, dtype=image_tokens.dtype)
            y, x = torch.meshgrid(axis, axis, indexing='ij')
            coordinates = torch.stack((x.flatten(), y.flatten()), dim=-1)
            visual = self.image_norm(
                image_tokens + image_gate.unsqueeze(-1).to(image_tokens.dtype)
                * self.image_grounding_projection(aligned_questions.to(image_tokens.dtype))
                + self.image_position_projection(coordinates).unsqueeze(0)
            )
            memory = torch.cat((memory, visual), dim=1)
            memory_blocked = torch.cat((memory_blocked, torch.zeros(
                (batch, patches), dtype=torch.bool, device=image_tokens.device)), dim=1)
        if not return_transport:
            return memory, memory_blocked
        ot_loss = (local_plan * local_cost).sum(dim=(1, 2)).mean()
        diagnostics = {
            "plan": plan.view(batch, H, patches + 1, -1) if H > 1 else plan,
            "matched_mass": local_plan.sum(dim=(1, 2)).view(batch, H).mean(-1) if H > 1 else local_plan.sum(dim=(1, 2)),
            "ot_loss": ot_loss,
            "image_to_dustbin_mass": plan[:, :patches, -1].sum(1).view(batch, H).mean(-1) if H > 1 else plan[:, :patches, -1].sum(1),
            "dustbin_to_question_mass": plan[:, -1, :-1].sum(1).view(batch, H).mean(-1) if H > 1 else plan[:, -1, :-1].sum(1),
            "dustbin_to_dustbin_mass": plan[:, -1, -1].view(batch, H).mean(-1) if H > 1 else plan[:, -1, -1],
            "row_residual": (plan.sum(-1) - row_marginal).abs().amax(-1),
            "column_residual": (plan.sum(-2) - column_marginal).abs().amax(-1),
        }
        return memory, memory_blocked, diagnostics


def independent_coupling(plan):
    """Remove pairwise alignment, preserving local marginals and all dustbins."""
    local = plan[..., :-1, :-1]
    mass = local.sum((-1, -2), keepdim=True)
    replacement = local.sum(-1, keepdim=True) * local.sum(-2, keepdim=True) / mass.clamp_min(1e-30)
    result = plan.clone()
    result[..., :-1, :-1] = torch.where(mass > 0, replacement, torch.zeros_like(replacement))
    return result
