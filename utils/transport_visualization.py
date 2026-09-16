"""Diagnostic plots for a single OT-fusion VQA example."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_transport_diagnostics(transport, output_path, image=None,
                               question_tokens=None, sample_index=0):
    if transport is None or transport.cost is None:
        raise ValueError("Full OT diagnostics are required to create a visualization")
    plan = transport.plan[sample_index].detach().float().cpu().numpy()
    cost = transport.cost[sample_index].detach().float().cpu().numpy()
    question_mask = transport.memory_padding_mask[sample_index].detach().cpu().numpy()
    valid_question = ~question_mask
    plan = plan[:, valid_question]
    cost = cost[:, valid_question]
    labels = list(question_tokens or [])
    if labels:
        labels = [label for label, valid in zip(labels, valid_question) if valid]

    visual_mass = plan.sum(1)
    question_mass = plan.sum(0)
    visual_target = (transport.visual_marginal[sample_index].detach().float().cpu().numpy()
                     if transport.visual_marginal is not None else None)
    question_target = (transport.question_marginal[sample_index].detach().float().cpu().numpy()[valid_question]
                       if transport.question_marginal is not None else None)
    side = math.isqrt(len(visual_mass))
    patch_map = visual_mass.reshape(side, side) if side * side == len(visual_mass) else None
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].set_title("Image and received patch mass")
    if image is not None:
        axes[0, 0].imshow(np.asarray(image))
    if patch_map is not None:
        extent = None
        if image is not None:
            array = np.asarray(image)
            extent = (0, array.shape[1], array.shape[0], 0)
        axes[0, 0].imshow(patch_map, cmap="magma", alpha=0.55,
                          interpolation="nearest", extent=extent)
    axes[0, 0].axis("off")

    plan_plot = axes[0, 1].imshow(plan, aspect="auto", cmap="viridis")
    axes[0, 1].set_title("Transport plan")
    axes[0, 1].set_xlabel("Question token")
    axes[0, 1].set_ylabel("Image patch")
    figure.colorbar(plan_plot, ax=axes[0, 1], fraction=0.046)

    cost_plot = axes[0, 2].imshow(cost, aspect="auto", cmap="magma_r")
    axes[0, 2].set_title("Ground cost")
    axes[0, 2].set_xlabel("Question token")
    axes[0, 2].set_ylabel("Image patch")
    figure.colorbar(cost_plot, ax=axes[0, 2], fraction=0.046)

    visual_positions = np.arange(len(visual_mass))
    axes[1, 0].bar(visual_positions - 0.2, visual_mass, width=0.4, label="received")
    if visual_target is not None:
        axes[1, 0].bar(visual_positions + 0.2, visual_target, width=0.4, label="marginal")
    axes[1, 0].set_title("Visual mass")
    axes[1, 0].set_xlabel("Patch")
    axes[1, 0].legend()

    question_positions = np.arange(len(question_mass))
    axes[1, 1].bar(question_positions - 0.2, question_mass, width=0.4, label="received")
    if question_target is not None:
        axes[1, 1].bar(question_positions + 0.2, question_target, width=0.4, label="marginal")
    axes[1, 1].set_title("Question mass")
    axes[1, 1].set_xticks(question_positions)
    axes[1, 1].legend()
    if labels and len(labels) == len(question_mass):
        axes[1, 1].set_xticklabels(labels, rotation=45, ha="right")

    axes[1, 2].axis("off")
    values = {
        "transport cost": transport.transport_cost[sample_index].item(),
        "entropy": transport.entropy[sample_index].item(),
        "matched mass": transport.matched_mass[sample_index].item(),
        "unmatched mass": transport.unmatched_mass[sample_index].item(),
        "residual": transport.residual[sample_index].item(),
        "iterations": transport.iterations[sample_index].item(),
        "converged": bool(transport.converged[sample_index].item()),
    }
    axes[1, 2].text(
        0.02, 0.98,
        "\n".join(f"{key}: {value:.5g}" if isinstance(value, float)
                  else f"{key}: {value}" for key, value in values.items()),
        va="top", family="monospace",
    )
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
