"""Train Cross-Attention or OT evidence-routing VQA and select by generated F1."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_linear_schedule_with_warmup

from configs.arg_parser import get_args
from configs.config import Config
from model.ot_alignment import (
    AlignmentNegativeQueue, OTAlignmentConfig, OTContrastiveAligner,
)
from model.vqa_model import VQAModel
from utils.checkpoint import (
    load_student_initialization, read_checkpoint, restore_alignment_state,
    restore_training_state, save_checkpoint,
)
from utils.data_processing import load_dataframe
from utils.device import seed_everything
from utils.distributed import (
    broadcast_object, cleanup_distributed, initialize_distributed,
    reduce_totals, unwrap_model,
)
from utils.feature_cache import FeatureCacheDataset, collate_feature_cache
from utils.metrics import compute_em_and_f1
from utils.ot_alignment_training import (
    evaluate_alignment, teacher_passes_gate, train_alignment_epoch,
    train_distillation_epoch,
)
from utils.vqa_dataset import VQADataset, resolve_image_root


def _forward_batch(model, batch, device, diagnostics=False):
    if isinstance(batch, dict):
        answers = batch["answers"]
        result = model(
            image_features=batch["image_features"].to(device),
            question_features=batch["question_features"].to(device),
            question_padding_mask=batch["question_padding_mask"].to(device),
            answers=answers,
            return_diagnostics=diagnostics,
        )
    else:
        anno_ids, images, questions, answers = batch
        result = model(images.to(device), questions, answers, anno_ids,
                       return_diagnostics=diagnostics)
    return result, answers


def _module_gradient_norm(module):
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not squares:
        return 0.0
    return torch.stack(squares).sum().sqrt().item()


def train(model, train_loader, num_epochs, optimizer, scheduler, criterion,
          vocab_swap=None, device=None, diagnostics=False, epoch_offset=0,
          total_epochs=None, gradient_clip=None, distributed=None,
          grad_scaler=None):
    """Run teacher-forced optimization; generation is reserved for validation."""
    device = device or next(model.parameters()).device
    base_model = unwrap_model(model)
    losses, em_scores, f1_scores = [], [], []
    if len(train_loader) == 0:
        raise ValueError("Training dataset is empty")
    for epoch in range(num_epochs):
        displayed_epoch = epoch_offset + epoch + 1
        displayed_total = total_epochs if total_epochs is not None else epoch_offset + num_epochs
        model.train()
        total_loss = total_em = total_f1 = 0.0
        examples = tokens = 0
        diagnostic_rows = []
        for batch_idx, batch in enumerate(train_loader):
            amp_enabled = grad_scaler is not None and grad_scaler.is_enabled()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else None,
                enabled=amp_enabled,
            ):
                result, answers = _forward_batch(model, batch, device, diagnostics)
                if diagnostics:
                    logits, targets, fusion_output = result
                    if fusion_output.diagnostics:
                        diagnostic_rows.append({
                            key: value.detach().float().mean().item()
                            for key, value in fusion_output.diagnostics.items()
                        })
                else:
                    logits, targets = result
                loss = criterion(logits.transpose(1, 2), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is NaN or infinity")
            optimizer.zero_grad(set_to_none=True)
            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
            else:
                loss.backward()
            if diagnostics and diagnostic_rows:
                diagnostic_rows[-1]["fusion_gradient_norm"] = _module_gradient_norm(
                    base_model.fusion_module
                )
                if base_model.fusion_type != "cross_attention":
                    diagnostic_rows[-1]["routing_cost_gradient_norm"] = (
                        _module_gradient_norm(base_model.fusion_module.cost_query)
                    )
                    diagnostic_rows[-1]["routing_preference_gradient_norm"] = (
                        _module_gradient_norm(base_model.fusion_module.preference_score)
                    )
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    gradient_clip,
                )
            if grad_scaler is not None:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            batch_tokens = targets.ne(base_model.pad_token_id).sum().item()
            total_loss += loss.item() * batch_tokens
            tokens += batch_tokens
            losses.append(loss.item())
            hypotheses = base_model.answers_from_ids(logits.detach().argmax(-1))
            em, f1 = compute_em_and_f1(answers, hypotheses)
            count = len(answers)
            total_em += em * count
            total_f1 += f1 * count
            examples += count
            if (batch_idx + 1) % 2000 == 0 and (
                distributed is None or distributed.is_main
            ):
                print(f"Epoch {displayed_epoch}, batch {batch_idx + 1}: loss={loss.item():.4f}")
        if distributed is not None:
            total_loss, total_em, total_f1, examples, tokens = reduce_totals(
                [total_loss, total_em, total_f1, examples, tokens], distributed
            )
            if distributed.enabled:
                losses = [total_loss / max(tokens, 1)]
        em_scores.append(total_em / examples)
        f1_scores.append(total_f1 / examples)
        message = (f"Epoch {displayed_epoch}/{displayed_total}: "
                   f"loss={total_loss / max(tokens, 1):.4f}, "
                   f"teacher-forced EM={em_scores[-1]:.4f}, F1={f1_scores[-1]:.4f}")
        if diagnostic_rows:
            means = {key: sum(row[key] for row in diagnostic_rows) / len(diagnostic_rows)
                     for key in diagnostic_rows[0]}
            message += ", fusion " + ", ".join(
                f"{key}={value:.4g}" for key, value in means.items()
            )
        if distributed is None or distributed.is_main:
            print(message)
    return losses, em_scores, f1_scores


def _make_loader(args, split, shuffle, text_model, image_model, distributed=None):
    csv_path = getattr(args, f"{split}_csv_path")
    if args.feature_cache:
        candidate = Path(args.feature_cache) / split
        cache_path = candidate if (candidate / "manifest.json").is_file() else Path(args.feature_cache)
        dataset = FeatureCacheDataset(cache_path, csv_path, text_model, image_model)
        collate_fn = collate_feature_cache
    else:
        frame = load_dataframe(csv_path)
        image_path = resolve_image_root(
            frame,
            args.img_path,
            split,
            override=getattr(args, f"{split}_img_path"),
        )
        dataset = VQADataset(frame, transform=Config.transforms, img_path=image_path)
        collate_fn = None
    sampler = None
    if distributed is not None and distributed.enabled and split == "train":
        sampler = DistributedSampler(
            dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=shuffle,
            seed=args.seed,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        collate_fn=collate_fn,
    )


def _set_loader_epoch(loader, epoch):
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)


def _fusion_config_from_args(args):
    return {
        "layers": args.cross_fusion_layers,
        "heads": args.num_heads,
        "ffn_hidden": args.ffn_hidden,
        "dropout": args.fusion_dropout,
    }


def _routing_config_from_args(args):
    return {
        "slots": args.routing_slots,
        "reasoning_steps": args.routing_steps,
        "routing_dim": args.routing_dim,
        "heads": args.num_heads,
        "dropout": args.fusion_dropout,
        "epsilon": args.routing_epsilon,
        "tau": args.routing_tau,
        "sinkhorn_iterations": args.routing_iterations,
        "diagnostic_tolerance": args.routing_tolerance,
        "preference_smoothing": args.routing_preference_smoothing,
        "null_min": args.routing_null_min,
        "null_max": args.routing_null_max,
        "shared_step_weights": True,
        "visual_preference": args.routing_visual_preference,
    }


def _alignment_config_from_args(args):
    return OTAlignmentConfig(
        ot_dim=args.ot_alignment_dim,
        epsilon=args.ot_alignment_epsilon,
        tau_visual=args.ot_alignment_tau_visual,
        tau_question=args.ot_alignment_tau_question,
        max_iterations=args.ot_alignment_iterations,
        tolerance=args.ot_alignment_tolerance,
        negative_count=args.ot_negative_count,
        contrastive_temperature=args.ot_contrastive_temperature,
    )


def _set_requires_grad(module, enabled_by_name):
    for name, parameter in module.named_parameters():
        parameter.requires_grad_(bool(enabled_by_name.get(name, False)))


def _run_ot_alignment_training(
    args,
    model,
    train_loader,
    dev_loader,
    train_criterion,
    validation_criterion,
    text_model,
    image_model,
    device,
    resume,
    distributed,
):
    """Run Stage 1 teacher warm-up and Stage 2 student distillation."""
    base_model = unwrap_model(model)
    if base_model.fusion_type != "cross_attention":
        raise ValueError(
            "ot_contrastive_distill requires --fusion cross_attention"
        )
    if resume is not None and resume.get("format_version") != 4:
        raise ValueError("OT alignment training can resume only from version-4 last_training.pt")
    saved_alignment = resume.get("alignment_config") if resume is not None else None
    if saved_alignment:
        teacher_config = OTAlignmentConfig.from_dict(saved_alignment["teacher"])
        warmup_epochs = int(saved_alignment["alignment_warmup_epochs"])
        distill_warmup_epochs = int(saved_alignment["ot_distill_warmup_epochs"])
        distill_target_weight = float(saved_alignment["ot_distill_weight"])
        alignment_lr = float(saved_alignment.get("ot_alignment_lr", 1e-4))
        gate_failure_policy = str(
            saved_alignment.get("ot_gate_failure_policy", args.ot_gate_failure_policy)
        )
        queue_size = int(saved_alignment["ot_negative_queue_size"])
        training_seed = int(saved_alignment.get("seed", args.seed))
    else:
        teacher_config = _alignment_config_from_args(args)
        warmup_epochs = args.alignment_warmup_epochs
        distill_warmup_epochs = args.ot_distill_warmup_epochs
        distill_target_weight = args.ot_distill_weight
        alignment_lr = args.ot_alignment_lr
        gate_failure_policy = args.ot_gate_failure_policy
        queue_size = args.ot_negative_queue_size
        training_seed = args.seed
    alignment_config = {
        "alignment_mode": "ot_contrastive_distill",
        "alignment_warmup_epochs": warmup_epochs,
        "ot_alignment_lr": alignment_lr,
        "ot_gate_failure_policy": gate_failure_policy,
        "ot_distill_weight": distill_target_weight,
        "ot_distill_warmup_epochs": distill_warmup_epochs,
        "ot_negative_queue_size": queue_size,
        "seed": training_seed,
        "teacher": teacher_config.to_dict(),
    }
    if min(warmup_epochs, distill_warmup_epochs, queue_size) < 0:
        raise ValueError("Alignment epoch counts and queue size cannot be negative")
    if warmup_epochs < 1 or queue_size < 1:
        raise ValueError("Alignment warm-up and negative queue must be enabled")
    if distill_target_weight < 0:
        raise ValueError("OT distillation weight cannot be negative")
    if alignment_lr <= 0:
        raise ValueError("OT alignment learning rate must be positive")

    teacher_base = OTContrastiveAligner(
        base_model.image_model.hidden_size,
        base_model.question_encoder.hidden_size,
        teacher_config,
    ).to(device)
    teacher = (
        DistributedDataParallel(
            teacher_base,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
            broadcast_buffers=False,
        )
        if distributed.enabled else teacher_base
    )
    queue = AlignmentNegativeQueue(queue_size)
    student_trainability = {
        name: parameter.requires_grad for name, parameter in base_model.named_parameters()
    }
    student_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    student_lr = args.lr if args.lr is not None else Config.lr
    optimizer = optim.AdamW(
        [
            {"params": student_parameters, "lr": student_lr},
            {"params": list(teacher.parameters()), "lr": alignment_lr},
        ],
        weight_decay=args.weight_decay,
    )
    total_epochs = warmup_epochs + args.epochs
    alignment_steps = len(train_loader) * warmup_epochs
    student_steps = len(train_loader) * args.epochs
    # Keep the student's base learning rate intact while only the teacher is
    # active, then apply the same full linear schedule used by its baseline.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            1.0
            if step <= alignment_steps
            else max(0.0, (alignment_steps + student_steps - step) / student_steps)
        ),
    )
    start_epoch = global_step = 0
    best_metric = None
    epochs_without_improvement = 0
    restored_stage = None
    fallback_active = False
    if resume is not None:
        start_epoch, global_step, best_metric = restore_training_state(
            resume, base_model, optimizer, scheduler
        )
        restored_stage = restore_alignment_state(resume, teacher_base, queue)
        fallback_active = bool((restored_stage or {}).get("fallback_active", False))
        if (
            start_epoch >= warmup_epochs
            and not (restored_stage or {}).get("gate_passed", False)
            and gate_failure_policy == "fallback"
        ):
            # Version-4 checkpoints written before fallback support stopped at
            # the failed gate. They can safely resume as native VQA training.
            fallback_active = True
        epochs_without_improvement = int(resume.get("epochs_without_improvement", 0))
        if (
            start_epoch >= warmup_epochs
            and not (restored_stage or {}).get("gate_passed", False)
            and not fallback_active
        ):
            raise RuntimeError(
                "Cannot resume distillation: the saved OT teacher did not pass "
                "the alignment decision gate"
            )
    if start_epoch >= total_epochs:
        raise ValueError("Resume checkpoint already reached the requested staged epoch count")

    destination = Path(args.model_path)
    if distributed.is_main:
        destination.mkdir(parents=True, exist_ok=True)
    distributed.barrier()
    metrics_path = destination / "metrics.jsonl"
    if resume is None and distributed.is_main:
        metrics_path.write_text("", encoding="utf-8")
    history = []
    from test import evaluation

    for epoch in range(start_epoch, total_epochs):
        _set_loader_epoch(train_loader, epoch)
        in_warmup = epoch < warmup_epochs
        if in_warmup:
            _set_requires_grad(base_model, {})
            teacher_base.requires_grad_(True)
            train_metrics = train_alignment_epoch(
                model,
                teacher,
                train_loader,
                optimizer,
                scheduler,
                queue,
                device,
                gradient_clip=args.gradient_clip or None,
                distributed=distributed,
            )
            global_step += len(train_loader)
            distributed.barrier()
            val_metrics = (
                evaluate_alignment(base_model, teacher_base, dev_loader, device)
                if distributed.is_main else None
            )
            val_metrics = broadcast_object(val_metrics, distributed)
            gate_passed = None
            gate_reason = ""
            if epoch + 1 == warmup_epochs:
                gate_passed, gate_reason = teacher_passes_gate(
                    val_metrics, teacher_config.negative_count
                )
                fallback_active = (
                    not gate_passed and gate_failure_policy == "fallback"
                )
            row = {
                "epoch": epoch + 1,
                "stage": "alignment_warmup",
                "learning_rate": scheduler.get_last_lr()[1],
                "student_learning_rate": scheduler.get_last_lr()[0],
            }
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            if distributed.is_main:
                history.append(row)
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                save_checkpoint(
                    destination / "last_training.pt",
                    model=base_model,
                    text_model=text_model,
                    image_model=image_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    global_step=global_step,
                    best_metric=best_metric,
                    epochs_without_improvement=epochs_without_improvement,
                    format_version=4,
                    alignment_teacher=teacher_base,
                    alignment_config=alignment_config,
                    negative_queue=queue,
                    training_stage={
                        "phase": "alignment_warmup",
                        "warmup_complete": epoch + 1 >= warmup_epochs,
                        "gate_passed": gate_passed,
                        "gate_reason": gate_reason,
                        "fallback_active": fallback_active,
                    },
                )
                print(
                    f"Alignment epoch {epoch + 1}/{warmup_epochs}: "
                    f"loss={train_metrics['ot_nce_loss']:.4f}, "
                    f"val margin={val_metrics['ot_score_margin']:.4f}, "
                    f"I2Q={val_metrics['ot_i2q_accuracy']:.3f}, "
                    f"Q2I={val_metrics['ot_q2i_accuracy']:.3f}"
                )
            distributed.barrier()
            if epoch + 1 == warmup_epochs:
                if not gate_passed:
                    if gate_failure_policy == "error":
                        raise RuntimeError(
                            "OT teacher failed the pre-distillation decision gate: "
                            + gate_reason
                        )
                    if distributed.is_main:
                        print(
                            "OT teacher failed the decision gate; continuing with "
                            "native Cross-Attention fallback. Reason: " + gate_reason
                        )
                elif distributed.is_main:
                    print("OT teacher passed the alignment gate; starting student distillation.")
            continue

        _set_requires_grad(base_model, student_trainability)
        teacher_base.requires_grad_(False)
        student_epoch = epoch - warmup_epochs
        if student_epoch == 0:
            # Make paired student training independent of random numbers consumed
            # while constructing and warming the training-only OT teacher.
            seed_everything(training_seed + distributed.rank)
        if fallback_active:
            distill_weight = 0.0
        elif distill_warmup_epochs:
            distill_weight = distill_target_weight * min(
                1.0, (student_epoch + 1) / distill_warmup_epochs
            )
        else:
            distill_weight = distill_target_weight
        if fallback_active:
            losses, train_em_rows, train_f1_rows = train(
                model,
                train_loader,
                1,
                optimizer,
                scheduler,
                train_criterion,
                device=device,
                diagnostics=args.diagnostics,
                epoch_offset=student_epoch,
                total_epochs=args.epochs,
                gradient_clip=args.gradient_clip or None,
                distributed=distributed,
            )
            train_em = train_em_rows[-1]
            train_f1 = train_f1_rows[-1]
            train_diagnostics = {
                "vqa_loss": losses[-1],
                "ot_distill_loss": 0.0,
                "ot_distill_weight": 0.0,
                "ot_matched_mass": 0.0,
                "ot_convergence_rate": 0.0,
            }
        else:
            losses, train_em, train_f1, train_diagnostics = train_distillation_epoch(
                model,
                teacher,
                train_loader,
                optimizer,
                scheduler,
                train_criterion,
                device,
                distill_weight,
                gradient_clip=args.gradient_clip or None,
                distributed=distributed,
            )
        global_step += len(train_loader)
        distributed.barrier()
        validation = (
            evaluation(
                base_model,
                dev_loader,
                validation_criterion,
                device=device,
                diagnostics=args.diagnostics,
            )
            if distributed.is_main else None
        )
        validation = broadcast_object(validation, distributed)
        val_loss, val_em, val_f1 = validation[:3]
        val_diagnostics = validation[3] if len(validation) > 3 else {}
        current = {"f1": val_f1, "loss": val_loss, "epoch": epoch + 1}
        improved = (
            best_metric is None
            or val_f1 > best_metric["f1"]
            or (val_f1 == best_metric["f1"] and val_loss < best_metric["loss"])
        )
        if improved:
            best_metric = current
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if distributed.is_main:
            save_checkpoint(
                destination / "last_training.pt",
                model=base_model,
                text_model=text_model,
                image_model=image_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                global_step=global_step,
                best_metric=best_metric,
                epochs_without_improvement=epochs_without_improvement,
                format_version=4,
                alignment_teacher=teacher_base,
                alignment_config=alignment_config,
                negative_queue=queue,
                training_stage={
                    "phase": "distillation",
                    "warmup_complete": True,
                    "gate_passed": not fallback_active,
                    "fallback_active": fallback_active,
                },
            )
            if improved:
                # Deployment artifact: student only, standard v3, and no OT teacher.
                save_checkpoint(
                    destination / "best.pt",
                    model=base_model,
                    text_model=text_model,
                    image_model=image_model,
                    epoch=epoch + 1,
                    global_step=global_step,
                    best_metric=best_metric,
                )
        row = {
            "epoch": epoch + 1,
            "stage": "vqa_fallback" if fallback_active else "distillation",
            "train_loss": sum(losses) / len(losses),
            "train_em": train_em,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_em": val_em,
            "val_f1": val_f1,
            "learning_rate": scheduler.get_last_lr()[0],
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
            "ot_gate_fallback": fallback_active,
        }
        row.update({f"train_{key}": value for key, value in train_diagnostics.items()})
        row.update({f"val_{key}": value for key, value in val_diagnostics.items()})
        if distributed.is_main:
            history.append(row)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            phase_name = "VQA fallback" if fallback_active else "Distillation"
            print(
                f"{phase_name} epoch {student_epoch + 1}/{args.epochs}: "
                f"loss={row['train_loss']:.4f}, KL={train_diagnostics['ot_distill_loss']:.4f}, "
                f"generated val F1={val_f1:.4f}, weight={distill_weight:.4f}"
            )
        distributed.barrier()
        if (
            args.early_stopping_patience
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            if distributed.is_main:
                print(
                    f"Early stopping at staged epoch {epoch + 1}; best generated "
                    f"F1={best_metric['f1']:.4f}."
                )
            break

    distillation_rows = [
        row for row in history
        if row["stage"] in {"distillation", "vqa_fallback"}
    ]
    if distillation_rows and distributed.is_main:
        plt.figure(figsize=(10, 6))
        plt.plot(
            [row["epoch"] for row in distillation_rows],
            [row["val_em"] for row in distillation_rows],
            label="Generated EM",
        )
        plt.plot(
            [row["epoch"] for row in distillation_rows],
            [row["val_f1"] for row in distillation_rows],
            label="Generated F1",
        )
        plt.xlabel("Staged epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(destination / "evaluation_metrics_plot.png")
        plt.close()


def main():
    args = get_args()
    if args.batch_size < 1 or args.epochs < 1:
        raise ValueError("batch_size and epochs must be positive")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience cannot be negative")
    if min(args.d_model, args.ffn_hidden, args.num_layers, args.num_heads) < 1:
        raise ValueError("model dimensions, layers, and heads must be positive")
    if args.d_model % args.num_heads:
        raise ValueError("d_model must be divisible by num_heads")
    if not 0.0 <= args.drop_prob < 1.0 or not 0.0 <= args.fusion_dropout < 1.0:
        raise ValueError("drop_prob and fusion_dropout must be in [0, 1)")
    if args.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    if args.gradient_clip < 0:
        raise ValueError("gradient_clip cannot be negative")
    if args.student_init_checkpoint and args.resume:
        raise ValueError("Use either --student_init_checkpoint or --resume, not both")
    if args.save_student_initialization and (args.resume or args.student_init_checkpoint):
        raise ValueError(
            "--save_student_initialization creates a fresh checkpoint and cannot be combined "
            "with --resume or --student_init_checkpoint"
        )
    if args.alignment_mode == "ot_contrastive_distill":
        if args.fusion != "cross_attention" and not args.resume:
            raise ValueError(
                "OT contrastive distillation requires --fusion cross_attention"
            )
        if min(
            args.alignment_warmup_epochs,
            args.ot_alignment_dim,
            args.ot_alignment_iterations,
            args.ot_negative_count,
            args.ot_negative_queue_size,
        ) < 1:
            raise ValueError("OT alignment dimensions, counts, queue, and warm-up must be positive")
        if min(
            args.ot_alignment_epsilon,
            args.ot_alignment_tau_visual,
            args.ot_alignment_tau_question,
            args.ot_alignment_tolerance,
            args.ot_contrastive_temperature,
            args.ot_alignment_lr,
        ) <= 0:
            raise ValueError("OT alignment regularization values must be positive")
        if args.ot_distill_weight < 0:
            raise ValueError("OT distillation weight cannot be negative")
        if args.ot_distill_warmup_epochs < 0:
            raise ValueError("OT distillation warm-up cannot be negative")
        if args.mixed_precision:
            raise ValueError(
                "Mixed precision is currently supported by direct VQA training only"
            )
    if args.fusion != "cross_attention":
        if args.alignment_mode != "none":
            raise ValueError(
                "OT evidence routing is trained directly from the VQA loss; "
                "use --alignment_mode none"
            )
        if min(
            args.routing_slots, args.routing_steps, args.routing_dim,
            args.routing_iterations,
        ) < 1:
            raise ValueError("Routing slots, steps, dimension, and iterations must be positive")
        if args.routing_dim % args.num_heads:
            raise ValueError("routing_dim must be divisible by num_heads")
        if args.routing_epsilon <= 0 or args.routing_tau < 0 or args.routing_tolerance <= 0:
            raise ValueError("Routing epsilon/tolerance must be positive and tau nonnegative")
        if not 0 <= args.routing_preference_smoothing < 1:
            raise ValueError("routing_preference_smoothing must be in [0, 1)")
        if not 0 < args.routing_null_min < args.routing_null_max < 1:
            raise ValueError("Routing null bounds must satisfy 0 < min < max < 1")
    seed_everything(args.seed)
    distributed = initialize_distributed(args.device)
    device = distributed.device
    if distributed.is_main:
        suffix = (
            f" with DistributedDataParallel ({distributed.world_size} GPUs)"
            if distributed.enabled else ""
        )
        print(f"Training on device: {device}{suffix}")

    embeddings_file = None
    if args.feature_cache:
        cache_path = Path(args.feature_cache)
        candidates = [
            cache_path / "embeddings.pt",
            cache_path / "train" / "embeddings.pt",
            cache_path / "dev" / "embeddings.pt",
            cache_path.parent / "embeddings.pt",
        ]
        for candidate in candidates:
            if candidate.is_file():
                embeddings_file = str(candidate)
                break

    resume = read_checkpoint(args.resume, device) if args.resume else None
    if resume is not None:
        expected_version = 4 if args.alignment_mode == "ot_contrastive_distill" else 3
        if resume["format_version"] != expected_version:
            raise ValueError(
                f"This training mode can resume only from a version-{expected_version} checkpoint"
            )
        text_model, image_model = resume["text_model"], resume["image_model"]
        model = VQAModel(text_model=text_model, image_model=image_model,
                         skip_encoders=bool(args.feature_cache),
                         embeddings_path=embeddings_file,
                         **resume["model_config"]).to(device)
    else:
        text_model, image_model = args.text_model, args.image_model
        model = VQAModel(text_model=text_model, image_model=image_model,
                         d_model=args.d_model,
                         ffn_hidden=args.ffn_hidden, num_layers=args.num_layers,
                         num_heads=args.num_heads, drop_prob=args.drop_prob,
                         freeze_answer_embeddings=args.freeze_answer_embeddings,
                         fusion=args.fusion,
                         fusion_config=_fusion_config_from_args(args),
                         routing_config=(
                             _routing_config_from_args(args)
                             if args.fusion != "cross_attention" else None
                         ),
                         skip_encoders=bool(args.feature_cache),
                         embeddings_path=embeddings_file).to(device)
        if args.student_init_checkpoint:
            load_student_initialization(args.student_init_checkpoint, model)
            # Common weights and common RNG state make paired baseline/OT runs
            # comparable even though their auxiliary modules differ.
            seed_everything(args.seed)
    if args.feature_cache and distributed.is_main:
        print("[Model] Feature cache active: ViT and BERT backbones skipped (0 ViT / 0 BERT weights loaded).")
    if args.save_student_initialization:
        if distributed.is_main:
            save_checkpoint(
                args.save_student_initialization,
                model=model,
                text_model=text_model,
                image_model=image_model,
            )
            print(f"Saved common student initialization: {args.save_student_initialization}")
        distributed.barrier()
        cleanup_distributed(distributed)
        return

    train_loader = _make_loader(
        args, "train", True, text_model, image_model, distributed
    )
    dev_loader = _make_loader(
        args, "dev", False, text_model, image_model, distributed
    )
    if not len(train_loader) or not len(dev_loader):
        raise ValueError("Training and development datasets must be non-empty")
    base_model = model
    if distributed.enabled:
        model = DistributedDataParallel(
            base_model,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
            broadcast_buffers=False,
        )
        # Give different workers independent dropout streams after DDP has
        # synchronized the initial parameters.
        seed_everything(args.seed + distributed.rank)
    train_criterion = nn.CrossEntropyLoss(
        ignore_index=base_model.pad_token_id, label_smoothing=args.label_smoothing
    )
    validation_criterion = nn.CrossEntropyLoss(ignore_index=base_model.pad_token_id)
    if args.alignment_mode == "ot_contrastive_distill":
        _run_ot_alignment_training(
            args,
            model,
            train_loader,
            dev_loader,
            train_criterion,
            validation_criterion,
            text_model,
            image_model,
            device,
            resume,
            distributed,
        )
        cleanup_distributed(distributed)
        return
    trainable_parameters = [parameter for parameter in model.parameters()
                            if parameter.requires_grad]
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    optimizer = optim.AdamW(
        trainable_parameters,
        lr=args.lr if args.lr is not None else Config.lr,
        weight_decay=args.weight_decay,
    )
    grad_scaler = (
        torch.amp.GradScaler("cuda")
        if args.mixed_precision and device.type == "cuda" else None
    )
    if args.mixed_precision and device.type != "cuda" and distributed.is_main:
        print("Mixed precision requested but disabled because the selected device is not CUDA.")
    if distributed.is_main:
        print(
            f"Parameters: total={total_parameter_count:,}, "
            f"trainable={trainable_parameter_count:,}"
        )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * args.epochs
    )
    start_epoch = global_step = 0
    best_metric = None
    epochs_without_improvement = 0
    if resume is not None:
        start_epoch, global_step, best_metric = restore_training_state(
            resume, base_model, optimizer, scheduler, grad_scaler
        )
        epochs_without_improvement = int(resume.get("epochs_without_improvement", 0))
    if start_epoch >= args.epochs:
        raise ValueError("Resume checkpoint has already reached the requested epoch count")

    destination = Path(args.model_path)
    if distributed.is_main:
        destination.mkdir(parents=True, exist_ok=True)
    distributed.barrier()
    metrics_path = destination / "metrics.jsonl"
    if best_metric is not None and "epoch" not in best_metric:
        # Version-3 checkpoints written before early stopping tracked the best
        # score but not its epoch. Recover it from the adjacent history when possible.
        best_metric = dict(best_metric)
        best_metric["epoch"] = "unknown"
        if metrics_path.is_file():
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                try:
                    old_row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (old_row.get("val_f1") == best_metric.get("f1") and
                        old_row.get("val_loss") == best_metric.get("loss")):
                    best_metric["epoch"] = old_row.get("epoch", "unknown")
    if resume is None and distributed.is_main:
        metrics_path.write_text("", encoding="utf-8")
    history = []
    from test import evaluation
    for epoch in range(start_epoch, args.epochs):
        _set_loader_epoch(train_loader, epoch)
        losses, train_em, train_f1 = train(
            model, train_loader, 1, optimizer, scheduler, train_criterion,
            device=device, diagnostics=args.diagnostics,
            epoch_offset=epoch, total_epochs=args.epochs,
            gradient_clip=args.gradient_clip or None,
            distributed=distributed,
            grad_scaler=grad_scaler,
        )
        global_step += len(train_loader)
        distributed.barrier()
        validation = (
            evaluation(
                base_model, dev_loader, validation_criterion, device=device,
                diagnostics=args.diagnostics,
            )
            if distributed.is_main else None
        )
        validation = broadcast_object(validation, distributed)
        val_loss, val_em, val_f1 = validation[:3]
        val_diagnostics = validation[3] if len(validation) > 3 else {}
        current = {"f1": val_f1, "loss": val_loss, "epoch": epoch + 1}
        improved = (best_metric is None or val_f1 > best_metric["f1"] or
                    (val_f1 == best_metric["f1"] and val_loss < best_metric["loss"]))
        if improved:
            best_metric = current
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        checkpoint_args = dict(
            model=model, text_model=text_model, image_model=image_model,
            optimizer=optimizer, scheduler=scheduler, grad_scaler=grad_scaler,
            epoch=epoch + 1, global_step=global_step, best_metric=best_metric,
            epochs_without_improvement=epochs_without_improvement,
        )
        if distributed.is_main:
            checkpoint_args["model"] = base_model
            save_checkpoint(destination / "last.pt", **checkpoint_args)
            if improved:
                save_checkpoint(destination / "best.pt", **checkpoint_args)
        row = {"epoch": epoch + 1, "train_loss": sum(losses) / len(losses),
               "train_em": train_em[-1], "train_f1": train_f1[-1],
               "val_loss": val_loss, "val_em": val_em, "val_f1": val_f1,
               "learning_rate": scheduler.get_last_lr()[0],
               "total_parameters": total_parameter_count,
               "trainable_parameters": trainable_parameter_count,
               "improved": improved,
               "epochs_without_improvement": epochs_without_improvement}
        row.update({f"val_{key}": value for key, value in val_diagnostics.items()})
        if distributed.is_main:
            history.append(row)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(f"Validation: loss={val_loss:.4f}, generated EM={val_em:.4f}, "
                  f"F1={val_f1:.4f}, best epoch={best_metric['epoch']}, "
                  f"patience={epochs_without_improvement}/{args.early_stopping_patience or 'off'}")
            if val_diagnostics:
                print(
                    "Validation diagnostics: "
                    f"unique predictions={val_diagnostics.get('unique_predictions', 0):g}, "
                    f"top prediction fraction={val_diagnostics.get('top_prediction_fraction', 0):.3f}, "
                    f"routing/attention entropy="
                    f"{val_diagnostics.get('fusion_routing_entropy', val_diagnostics.get('fusion_attention_entropy', 0)):.3f}, "
                    f"latency={val_diagnostics.get('latency_ms_per_example', 0):.2f} ms/example"
                )
        distributed.barrier()
        if (args.early_stopping_patience and
                epochs_without_improvement >= args.early_stopping_patience):
            if distributed.is_main:
                print(f"Early stopping at epoch {epoch + 1}; best checkpoint is epoch "
                      f"{best_metric['epoch']} with generated F1={best_metric['f1']:.4f}.")
            break

    if distributed.is_main:
        plt.figure(figsize=(10, 6))
        plt.plot([row["epoch"] for row in history], [row["val_em"] for row in history], label="Generated EM")
        plt.plot([row["epoch"] for row in history], [row["val_f1"] for row in history], label="Generated F1")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(destination / "evaluation_metrics_plot.png")
        plt.close()
    cleanup_distributed(distributed)


if __name__ == "__main__":
    try:
        main()
    finally:
        # Also clean up when a validation gate or another training check raises.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
