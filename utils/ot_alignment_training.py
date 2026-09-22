"""Staged training helpers for contrastive OT alignment and distillation."""

from __future__ import annotations

import math
from typing import Iterable

import torch

from model.ot_alignment import (
    AlignmentNegativeQueue,
    OTContrastiveAligner,
    ot_attention_distillation_loss,
)
from utils.metrics import compute_em_and_f1


def extract_alignment_features(model, batch, device):
    """Return frozen spatial/image and content/question tokens for either data path."""
    with torch.no_grad():
        if isinstance(batch, dict):
            image_embeddings = batch["image_features"].to(device)
            question_embeddings = batch["question_features"].to(device)
            question_padding_mask = batch["question_padding_mask"].to(
                device, dtype=torch.bool
            )
            answers = batch["answers"]
        else:
            anno_ids, images, questions, answers = batch
            image_embeddings, _ = model.image_model(
                images.to(device), image_ids=anno_ids
            )
            question_embeddings, question_padding_mask, _ = (
                model.question_encoder.encode_tokens(questions)
            )
        visual_tokens = model.image_model.spatial_tokens(image_embeddings)
        visual_padding_mask = torch.zeros(
            visual_tokens.shape[:2], dtype=torch.bool, device=device
        )
    return (
        visual_tokens,
        question_embeddings,
        visual_padding_mask,
        question_padding_mask,
        answers,
    )


def _mean_metrics(rows: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    examples = sum(count for count, _ in rows)
    if not examples:
        return {}
    keys = rows[0][1]
    return {
        key: sum(count * values[key] for count, values in rows) / examples
        for key in keys
    }


def _diagnostic_values(output) -> dict[str, float]:
    return {
        key: value.detach().float().mean().item()
        for key, value in output.diagnostics().items()
    }


def train_alignment_epoch(
    model,
    teacher: OTContrastiveAligner,
    loader: Iterable,
    optimizer,
    scheduler,
    queue: AlignmentNegativeQueue,
    device,
    gradient_clip: float | None = None,
    contrastive_weight: float = 1.0,
) -> dict[str, float]:
    """Warm up only the OT teacher with symmetric hard-negative InfoNCE."""
    model.eval()
    teacher.train()
    rows = []
    skipped = 0
    for batch in loader:
        features = extract_alignment_features(model, batch, device)
        visual, question, visual_mask, question_mask, _ = features
        if visual.size(0) < 2 and len(queue) == 0:
            queue.enqueue(visual, question, visual_mask, question_mask)
            skipped += visual.size(0)
            continue
        # The design targets an effective candidate batch of at least eight.
        use_queue = queue if len(queue) and visual.size(0) < 8 else None
        output = teacher(
            visual, question, visual_mask, question_mask, queue=use_queue
        )
        if not torch.isfinite(output.loss):
            raise FloatingPointError("OT contrastive loss is NaN or infinity")
        optimizer.zero_grad(set_to_none=True)
        (contrastive_weight * output.loss).backward()
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), gradient_clip)
        optimizer.step()
        scheduler.step()
        rows.append((visual.size(0), _diagnostic_values(output)))
        queue.enqueue(visual, question, visual_mask, question_mask)
    metrics = _mean_metrics(rows)
    metrics["ot_skipped_examples"] = float(skipped)
    if not rows:
        raise ValueError("Alignment warm-up needs at least two examples")
    return metrics


@torch.no_grad()
def evaluate_alignment(model, teacher, loader, device) -> dict[str, float]:
    """Evaluate teacher retrieval and transport validity without updating its queue."""
    model.eval()
    teacher.eval()
    validation_queue = AlignmentNegativeQueue(
        capacity=max(8, teacher.config.negative_count)
    )
    rows = []
    for batch in loader:
        visual, question, visual_mask, question_mask, _ = extract_alignment_features(
            model, batch, device
        )
        if visual.size(0) < 2 and len(validation_queue) == 0:
            validation_queue.enqueue(visual, question, visual_mask, question_mask)
            continue
        use_queue = (
            validation_queue
            if len(validation_queue) and visual.size(0) < 8
            else None
        )
        output = teacher(
            visual, question, visual_mask, question_mask,
            queue=use_queue,
        )
        rows.append((visual.size(0), _diagnostic_values(output)))
        validation_queue.enqueue(visual, question, visual_mask, question_mask)
    metrics = _mean_metrics(rows)
    if not metrics:
        raise ValueError("Alignment validation needs at least two examples")
    return metrics


def teacher_passes_gate(metrics: dict[str, float], negative_count: int) -> tuple[bool, str]:
    """Apply the pre-distillation validity checks from the research plan."""
    required = (
        "ot_score_margin",
        "ot_i2q_accuracy",
        "ot_q2i_accuracy",
        "ot_matched_mass",
        "ot_entropy",
        "ot_residual",
        "ot_convergence_rate",
        "ot_visual_variance",
        "ot_question_variance",
    )
    missing = [name for name in required if name not in metrics]
    if missing:
        return False, f"missing diagnostics: {', '.join(missing)}"
    if not all(math.isfinite(metrics[name]) for name in required):
        return False, "non-finite alignment diagnostics"
    candidate_count = metrics.get("ot_candidate_count", negative_count + 1)
    random_accuracy = 1.0 / candidate_count
    failures = []
    if metrics["ot_score_margin"] <= 0:
        failures.append("positive pairs do not outrank hard negatives")
    if metrics["ot_i2q_accuracy"] <= random_accuracy:
        failures.append("image-to-question retrieval is not above chance")
    if metrics["ot_q2i_accuracy"] <= random_accuracy:
        failures.append("question-to-image retrieval is not above chance")
    if metrics["ot_matched_mass"] <= 1e-8:
        failures.append("matched mass collapsed")
    if min(metrics["ot_visual_variance"], metrics["ot_question_variance"]) <= 1e-8:
        failures.append("adapter features collapsed")
    if metrics.get("ot_collapsed", 0.0) > 0:
        failures.append("collapse diagnostic triggered")
    return not failures, "; ".join(failures)


def train_distillation_epoch(
    model,
    teacher: OTContrastiveAligner,
    loader: Iterable,
    optimizer,
    scheduler,
    criterion,
    device,
    distill_weight: float,
    gradient_clip: float | None = None,
) -> tuple[list[float], float, float, dict[str, float]]:
    """Train native Cross-Attention from VQA loss plus detached OT supervision."""
    model.train()
    teacher.eval()
    losses = []
    total_em = total_f1 = 0.0
    examples = tokens = 0
    total_vqa = total_distill = 0.0
    total_transport_mass = total_transport_convergence = 0.0
    for batch in loader:
        visual, question, visual_mask, question_mask, answers = (
            extract_alignment_features(model, batch, device)
        )
        # encode_from_features expects encoder outputs including ViT prefix tokens.
        # The student fusion consumes spatial tokens, so call the fusion and decoder
        # directly to avoid re-attaching synthetic prefix tokens.
        fusion_output = model.fusion_module(
            model._fusion_input_from_spatial_features(
                visual, question, visual_mask, question_mask
            ),
            return_diagnostics=True,
        )
        if fusion_output.attention_weights is None:
            raise RuntimeError("Cross-Attention did not expose training attention weights")
        ids = model.answer_embedding.tokenize(answers)
        logits = model.decode(
            ids[:, :-1],
            fusion_output.memory,
            memory_padding_mask=fusion_output.memory_padding_mask,
        )
        targets = ids[:, 1:]
        with torch.no_grad():
            transport = teacher.positive_transport(
                visual, question, visual_mask, question_mask
            )
        vqa_loss = criterion(logits.transpose(1, 2), targets)
        distill_loss = ot_attention_distillation_loss(
            transport.plan,
            fusion_output.attention_weights,
            question_mask,
            visual_mask,
            minimum_mass=teacher.config.minimum_mass,
        )
        loss = vqa_loss + distill_weight * distill_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("OT-distilled VQA loss is NaN or infinity")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                gradient_clip,
            )
        optimizer.step()
        scheduler.step()

        count_tokens = targets.ne(model.pad_token_id).sum().item()
        count = len(answers)
        total_vqa += vqa_loss.item() * count_tokens
        total_distill += distill_loss.item() * count
        total_transport_mass += transport.matched_mass.mean().item() * count
        total_transport_convergence += transport.converged.float().mean().item() * count
        tokens += count_tokens
        examples += count
        losses.append(loss.item())
        hypotheses = model.answers_from_ids(logits.detach().argmax(-1))
        em, f1 = compute_em_and_f1(answers, hypotheses)
        total_em += em * count
        total_f1 += f1 * count
    if not examples:
        raise ValueError("Distillation training dataset is empty")
    diagnostics = {
        "vqa_loss": total_vqa / max(tokens, 1),
        "ot_distill_loss": total_distill / examples,
        "ot_distill_weight": distill_weight,
        "ot_matched_mass": total_transport_mass / examples,
        "ot_convergence_rate": total_transport_convergence / examples,
    }
    return losses, total_em / examples, total_f1 / examples, diagnostics
