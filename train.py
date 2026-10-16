"""Train OT VQA, report answer metrics, and select checkpoints by a validation metric.

GQA reports binary generated-answer accuracy for training and validation.
SciVQA reports ROUGE-1/L and BERTScore precision, recall and F1.
"""

from __future__ import annotations

import json
import hashlib
import os
import math
from contextlib import nullcontext
from time import perf_counter
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup

from configs.arg_parser import get_args
from configs.config import Config
from model.vqa_model import VQAModel
from utils.checkpoint import (LEGACY_MAX_ANSWER_TOKENS, LEGACY_MAX_QUESTION_TOKENS,
                              MODEL_CONFIG_KEYS, read_checkpoint, restore_training_state,
                              save_checkpoint)
from utils.data_processing import load_dataframe
from utils.device import resolve_device, seed_everything
from utils.metrics import build_bertscore_scorer, metrics_for_dataset, resolve_dataset
from utils.vqa_dataset import VQADataset, resolve_image_root, native_collate
from utils.answer_text import ANSWER_POSTPROCESSING
from utils.image_processing import build_image_transform
from utils.training_preflight import validate_training_data, validate_trainable_parameters
from utils.losses import AnswerCrossEntropyLoss
from utils.training_runtime import (accumulation_groups, autocast_context, optimizer_update,
                                    gather_rng, restore_rng, source_manifest, cuda_memory_report)


def _forward_batch(model, batch, device, return_ot_loss=False):
    anno_ids, images, questions, answers = batch
    if getattr(model.module if isinstance(model, DistributedDataParallel) else model, "model_family", "legacy") == "qwen3_vl":
        return model(images, questions, answers, anno_ids, return_ot_loss=return_ot_loss)
    if return_ot_loss:
        return model(images.to(device), questions, answers, anno_ids, return_ot_loss=True)
    return model(images.to(device), questions, answers, anno_ids)


def _format_epoch_metrics(epoch, total_epochs, split, loss, scores):
    metrics = ", ".join(f"{name}={value:.4f}" for name, value in scores.items())
    return f"Epoch {epoch}/{total_epochs} {split}: loss={loss:.4f}" + (f", {metrics}" if metrics else "")


def train(model, train_loader, num_epochs, optimizer, scheduler, criterion,
          vocab_swap=None, device=None, epoch_offset=0,
          total_epochs=None, gradient_clip=None, ot_loss_weight=0.0,
          gradient_accumulation_steps=1, amp="off", scaler=None, max_updates=None,
          log_every_updates=25):
    """Accumulate exact global example/token means, including short final groups."""
    device = device or next(model.parameters()).device
    base_model = model.module if isinstance(model, DistributedDataParallel) else model
    losses, epoch_losses, epoch_ot_losses = [], [], []
    if len(train_loader) == 0:
        raise ValueError("Training dataset is empty")
    normalization = getattr(criterion, "normalization", "token")
    ignore = criterion.ignore_index
    distributed = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size() if distributed else 1
    reduction_dtype = torch.float32 if device.type == "mps" else torch.float64
    updates = overflows = batches_seen = attempted_updates = 0
    gradient_samples = []
    ot_diagnostics = {}
    max_sequence = max_visual = 0
    prep_seconds = 0.
    started = perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(num_epochs):
        model.train()
        ce_total = aux_total = ce_weight = examples_total = 0.
        for group in accumulation_groups(train_loader, gradient_accumulation_steps):
            prepared_group, counts = [], []
            prep_started = perf_counter()
            for batch in group:
                if getattr(base_model, "model_family", "legacy") == "qwen3_vl":
                    prepared = base_model.prepare_batch(batch[1], batch[2], batch[3], device)
                    targets = prepared["labels"][:, 1:]
                else:
                    prepared = None
                    targets = base_model.answer_embedding.tokenize(batch[3], base_model.max_answer_tokens)[:, 1:]
                count = targets.size(0) if normalization == "example" else targets.ne(ignore).sum().item()
                prepared_group.append(prepared)
                counts.append((count, len(batch[3])))
            prep_seconds += perf_counter()-prep_started
            denominators = torch.tensor([sum(c[0] for c in counts), sum(c[1] for c in counts)],
                                        device=device, dtype=reduction_dtype)
            if distributed:
                dist.all_reduce(denominators)
            optimizer.zero_grad(set_to_none=True)
            for micro, (batch, prepared, (count, examples)) in enumerate(zip(group, prepared_group, counts)):
                sync = model.no_sync() if isinstance(model, DistributedDataParallel) and micro < len(group)-1 else nullcontext()
                with sync:
                    with autocast_context(device, amp):
                        output = (model(prepared=prepared, return_ot_loss=ot_loss_weight > 0) if prepared is not None
                                  else _forward_batch(model, batch, device, return_ot_loss=ot_loss_weight > 0))
                        logits, targets = output[:2]
                        ce = criterion(logits.transpose(1, 2), targets)
                        aux = output[2] if len(output) > 2 else ce.new_zeros(())
                        loss = ce * (world * count / denominators[0]) + ot_loss_weight * aux * (world * examples / denominators[1])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Training loss is NaN or infinity")
                    if scaler is not None and scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                if getattr(base_model, "model_family", "legacy") == "qwen3_vl":
                    for name, value in base_model.adapter.diagnostics.items():
                        number = value.item()
                        old = ot_diagnostics.get(name, {"min": number, "max": number})
                        ot_diagnostics[name] = {"min": min(old["min"], number), "max": max(old["max"], number)}
                stats = getattr(base_model, "last_batch_stats", {})
                max_sequence = max(max_sequence, stats.get("sequence_tokens", 0))
                max_visual = max(max_visual, max(stats.get("visual_tokens", [0])))
                ce_total += ce.detach().item() * count
                aux_total += aux.detach().item() * examples
                ce_weight += count
                examples_total += examples
                losses.append(ce.detach().item() + ot_loss_weight * aux.detach().item())
                batches_seen += 1
            gradient_stats = {} if getattr(getattr(base_model, "profiler", None), "enabled", False) else None
            success = optimizer_update(model, optimizer, scheduler, scaler, gradient_clip, device, gradient_stats)
            if gradient_stats is not None:
                sample = {"optimizer_update": updates+int(success), "norms": gradient_stats}
                if len(gradient_samples) < 4:
                    gradient_samples.append(sample)
                elif len(gradient_samples) == 4:
                    gradient_samples.append(sample)
                else:
                    gradient_samples[-1] = sample
            updates += int(success)
            overflows += int(not success)
            attempted_updates += 1
            if (log_every_updates and attempted_updates % log_every_updates == 0 and
                    (not distributed or dist.get_rank() == 0)):
                elapsed = perf_counter()-started
                print(f"Epoch {epoch_offset+epoch+1}: optimizer_updates={updates}, "
                      f"microbatches={batches_seen}/{len(train_loader)}, ce={ce.detach().item():.4f}, "
                      f"lr={optimizer.param_groups[0]['lr']:.3g}, overflow_skips={overflows}, "
                      f"seconds/update={elapsed/max(updates, 1):.2f}", flush=True)
            if max_updates is not None and updates >= max_updates:
                break
        totals = torch.tensor([ce_total, aux_total, ce_weight, examples_total], dtype=reduction_dtype, device=device)
        if distributed:
            dist.all_reduce(totals)
        ce_total, aux_total, ce_weight, examples_total = totals.tolist()
        mean_aux = aux_total / max(examples_total, 1)
        epoch_losses.append(ce_total / max(ce_weight, 1) + ot_loss_weight * mean_aux)
        if ot_loss_weight:
            epoch_ot_losses.append(mean_aux)
    telemetry = {"optimizer_updates": updates, "overflow_skips": overflows,
                 "batches_seen": batches_seen, "complete_epoch": batches_seen == len(train_loader)*num_epochs,
                 "gradient_samples": gradient_samples, "ot_diagnostics": ot_diagnostics,
                 "max_sequence_tokens": max_sequence, "max_visual_tokens": max_visual,
                 "prepare_seconds": prep_seconds,
                 "stage_seconds": getattr(getattr(base_model, "profiler", None), "seconds", {}),
                 "seconds": perf_counter()-started, "memory": cuda_memory_report(device)}
    return losses, epoch_losses, epoch_ot_losses, telemetry


def _make_loader(args, split, shuffle, text_model, image_model, rank=0, world_size=1,
                 image_size=256, image_preprocessing="crop", figure_context=False,
                 prompt_version="scivqa-v2", native=False):
    csv_path = getattr(args, f"{split}_csv_path")
    frame = load_dataframe(csv_path)
    image_path = resolve_image_root(
        frame,
        args.img_path,
        split,
        override=getattr(args, f"{split}_img_path"),
    )
    dataset = VQADataset(frame, transform=None if native else build_image_transform(image_size, image_preprocessing), img_path=image_path,
                         dataset_name=resolve_dataset(args.dataset, args.train_csv_path),
                         figure_context=figure_context, prompt_version=prompt_version, native=native)
    if world_size > 1 and split == "train" and len(dataset) % world_size:
        raise ValueError("Distributed training needs a sample count divisible by world_size to avoid duplicated sampler padding")
    sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                  shuffle=shuffle, seed=args.seed)
               if world_size > 1 and split == "train" else None)
    return DataLoader(dataset, collate_fn=native_collate if native else None, batch_size=args.batch_size,
                      shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=args.num_workers,
                      multiprocessing_context="spawn" if args.num_workers > 0 else None,
                      pin_memory=args.device in {"cuda", "auto"} and torch.cuda.is_available())


def _make_train_eval_loader(dataset, args, device):
    if args.train_eval_samples == 0:
        return None
    if 0 < args.train_eval_samples < len(dataset):
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(dataset), generator=generator)[:args.train_eval_samples]
        dataset = Subset(dataset, indices.tolist())
    return DataLoader(dataset, collate_fn=native_collate if args.model_family == "qwen3_vl" else None,
                      batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=device.type == "cuda",
                      multiprocessing_context="spawn" if args.num_workers > 0 else None)


def _build_optimizer(model, args, resume=None):
    lr = args.lr if args.lr is not None else Config.lr
    if resume is None or resume.get("format_version", 0) >= 5:
        grouped = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if getattr(model, "model_family", "legacy") == "qwen3_vl":
                category = "lora" if "lora_" in name else "ot"
                rate = args.lora_lr if category == "lora" else lr
            else:
                category = "encoder" if name.startswith(("question_encoder.", "answer_embedding.")) else "decoder"
                rate = args.encoder_lr if category == "encoder" and args.encoder_lr is not None else lr
            decay = 0.0 if parameter.ndim < 2 or name.endswith("bias") else args.weight_decay
            key = (category, rate, decay)
            grouped.setdefault(key, []).append(parameter)
        if not grouped:
            raise ValueError("No trainable parameters; native control requires LoRA for training")
        return optim.AdamW([{"params": params, "lr": rate, "weight_decay": decay,
                             "group_name": category + ("_no_decay" if decay == 0 else "")}
                            for (category, rate, decay), params in grouped.items()], lr=lr)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    saved = (resume or {}).get("optimizer_state_dict")
    saved_groups = saved["param_groups"] if saved else []
    separate = args.encoder_lr is not None or any(
        group.get("group_name") == "encoder" for group in saved_groups)
    if separate and saved_groups and (len(saved_groups) not in {1, 2} or
                                     (len(saved_groups) == 1 and
                                      saved_groups[0].get("group_name") != "decoder")):
        raise ValueError("--encoder_lr requires a fresh run when resuming a single-LR checkpoint")
    if separate:
        encoder_ids = {id(parameter) for module in
                       (model.question_encoder, model.answer_embedding)
                       for parameter in module.parameters()}
        groups = [{"params": [p for p in trainable if id(p) not in encoder_ids],
                   "lr": lr, "group_name": "decoder"}]
        encoder_params = [p for p in trainable if id(p) in encoder_ids]
        if encoder_params:
            groups.append({"params": encoder_params,
                           "lr": args.encoder_lr if args.encoder_lr is not None else lr,
                           "group_name": "encoder"})
    else:
        groups = trainable
    return optim.AdamW(groups, lr=lr, weight_decay=args.weight_decay)


def _validation_selection(validation, epoch, metric, best):
    value = validation["loss"] if metric == "loss" else validation["metrics"][metric]
    if not math.isfinite(value):
        raise FloatingPointError(f"Validation {metric} is NaN or infinity")
    current = {"loss": validation["loss"], "epoch": epoch,
               "selection_metric": metric, "value": value}
    if best is None:
        return current, True
    if best.get("selection_metric", "loss") != metric:
        raise ValueError("--selection_metric does not match the resume checkpoint")
    best_value = best.get("value", best["loss"])
    return current, value < best_value if metric == "loss" else value > best_value


def _training_device(requested):
    """Initialize one CUDA worker per process when launched with torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return resolve_device(requested), 0, 1
    if requested not in {"auto", "cuda"}:
        raise ValueError("Multi-GPU training requires --device cuda or auto")
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU training requires CUDA")
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {local_rank} is unavailable")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://",
                            timeout=timedelta(hours=2))
    return torch.device("cuda", local_rank), dist.get_rank(), world_size


def _csv_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_training(args, device, rank, world_size):
    resume = read_checkpoint(args.resume, device) if args.resume else None
    saved_training = (resume or {}).get("training_config", {})
    saved_epochs = saved_training.get("arguments", {}).get("epochs")
    extend_schedule = (resume is not None and saved_epochs is not None
                       and "--epochs" in args.explicit_args and args.epochs > saved_epochs)
    if resume is not None:
        stored_destination = saved_training.get("arguments", {}).get("model_path", str(Path(args.resume).parent))
        if "--model_path" in args.explicit_args and Path(args.model_path).resolve() != Path(stored_destination).resolve():
            raise ValueError("Exact resume requires the original --model_path")
        args.model_path = stored_destination
    if resume is not None and resume.get("format_version") == 5:
        if not saved_training.get("complete_epoch", True):
            raise ValueError("Pilot checkpoint is mid-epoch; start a fresh full run")
        saved_args = saved_training.get("arguments", {})
        if saved_epochs is not None:
            if "--epochs" not in args.explicit_args:
                args.epochs = saved_epochs
            elif args.epochs < saved_epochs:
                raise ValueError("--epochs cannot decrease the saved total epoch target")
        contract = ("dataset", "compute_bertscore", "bertscore_model", "bertscore_rescale",
                    "bertscore_device", "bertscore_batch_size", "train_eval_samples",
                    "selection_metric", "early_stopping_patience", "batch_size", "gradient_accumulation_steps", "amp", "amp_init_scale", "warmup_ratio",
                    "seed", "weight_decay", "label_smoothing", "gradient_clip", "lr", "encoder_lr",
                    "lora_lr", "ot_loss_weight", "loss_normalization", "model_family",
                    "prompt_version", "decode_version", "output_drop_prob", "train_csv_path", "dev_csv_path",
                    "vlm_model", "vlm_revision", "query_revision", "visual_tokens", "total_sequence_tokens",
                    "query_tokens", "cost_dim", "quantize_4bit", "lora_rank", "gradient_checkpointing",
                    "text_model", "image_model", "train_img_path", "dev_img_path", "img_path",
                    "adapter_control", "plan_control", "num_workers", "max_updates",
                    *MODEL_CONFIG_KEYS)
        for name in contract:
            if name not in saved_args:
                continue
            if (f"--{name}" in args.explicit_args or f"--no-{name}" in args.explicit_args) and getattr(args, name) != saved_args[name]:
                raise ValueError(f"--{name} changes the saved resume recipe; start a fresh run")
            if hasattr(args, name):
                setattr(args, name, saved_args[name])
        if saved_training.get("world_size", 1) != world_size:
            raise ValueError("Exact resume requires the saved world size")
        for split in ("train", "dev"):
            if saved_training.get(f"{split}_csv_sha256") != _csv_digest(getattr(args, f"{split}_csv_path")):
                raise ValueError(f"{split} CSV changed since checkpoint")
    elif resume is not None and args.ot_loss_weight is None:
        raise ValueError("Historical checkpoint lacks auxiliary objective config; supply --ot_loss_weight explicitly")
    if args.ot_loss_weight is None:
        args.ot_loss_weight = 0.0
    if not math.isfinite(args.ot_loss_weight) or args.ot_loss_weight < 0:
        raise ValueError("ot_loss_weight must be finite and nonnegative")
    if args.gradient_accumulation_steps < 1 or (args.max_updates is not None and args.max_updates < 1):
        raise ValueError("Accumulation and max_updates must be positive")
    if not math.isfinite(args.amp_init_scale) or args.amp_init_scale <= 0:
        raise ValueError("amp_init_scale must be finite and positive")
    autocast_context(device, args.amp)
    destination = Path(args.model_path)
    if resume is None and destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Fresh run requires an empty --model_path: {destination}")
    dataset_name = resolve_dataset(args.dataset, args.train_csv_path)
    metric_names = metrics_for_dataset(dataset_name)
    if not args.compute_bertscore:
        metric_names = tuple(n for n in metric_names if not n.startswith("bertscore"))
    if args.batch_size < 1 or args.epochs < 1:
        raise ValueError("batch_size and epochs must be positive")
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        raise ValueError("lr must be finite and positive")
    if args.encoder_lr is not None and (not math.isfinite(args.encoder_lr) or args.encoder_lr <= 0):
        raise ValueError("encoder_lr must be finite and positive")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if args.log_every_updates < 0:
        raise ValueError("log_every_updates cannot be negative")
    if args.train_eval_samples < -1 or args.num_workers < 0:
        raise ValueError("train_eval_samples must be >= -1 and num_workers must be >= 0")
    if args.max_answer_tokens is not None and args.max_answer_tokens < 2:
        raise ValueError("max_answer_tokens must be at least 2")
    if args.max_question_tokens is not None and args.max_question_tokens < 2:
        raise ValueError("max_question_tokens must be at least 2")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience cannot be negative")
    if min(args.d_model, args.ffn_hidden, args.num_layers, args.num_heads) < 1:
        raise ValueError("model dimensions, layers, and heads must be positive")
    if args.d_model % args.num_heads:
        raise ValueError("d_model must be divisible by num_heads")
    if not 0.0 <= args.drop_prob < 1.0:
        raise ValueError("drop_prob must be in [0, 1)")
    if args.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    if args.gradient_clip < 0:
        raise ValueError("gradient_clip cannot be negative")
    if dataset_name != "gqa" and args.bertscore_batch_size < 1:
        raise ValueError("bertscore_batch_size must be positive")
    seed_everything(args.seed + rank)
    if rank == 0:
        if world_size == 1:
            print(f"Training on device: {device}")
        else:
            print(f"Training on {world_size} GPUs; primary device: {device}")
        print(f"Dataset: {dataset_name}; generated-answer metrics: {', '.join(metric_names)}")

    saved_normalization = (resume or {}).get("training_config", {}).get("loss_normalization", "token")
    loss_normalization = args.loss_normalization or saved_normalization
    if resume is not None and loss_normalization != saved_normalization:
        raise ValueError("--loss_normalization does not match the resume checkpoint; start a fresh run")
    resumed_best = (resume or {}).get("best_metric") or {}
    default_selection = "rouge_l_f1" if dataset_name == "scivqa" and resume is None else "loss"
    selection_metric = args.selection_metric or resumed_best.get("selection_metric", default_selection)
    if selection_metric != "loss" and selection_metric not in metric_names:
        raise ValueError(f"Selection metric {selection_metric} is unavailable for {dataset_name}")
    if resume is not None and selection_metric != resumed_best.get("selection_metric", "loss"):
        raise ValueError("--selection_metric does not match the resume checkpoint")
    native = args.model_family == "qwen3_vl" or (resume or {}).get("model_family") == "qwen3_vl"
    required_context = ((resume or {}).get("model_config", {}).get("figure_context", False) if resume
                        else bool(args.figure_context) if args.figure_context is not None else native)
    data_preflight = validate_training_data(args, world_size, required_context)
    if rank == 0:
        print("Data preflight: " + json.dumps(data_preflight), flush=True)
    if native:
        from model.factory import build_native
        args.model_family = "qwen3_vl"
        if args.fgw_alpha or args.unbalanced_tau is not None or (args.ot_num_heads not in {None, 1}):
            raise ValueError("Native prototype supports single-head partial OT without FGW/UOT")
        model = build_native(args, device, resume)
        text_model, image_model = model.model_config["text_model"], model.model_config["vlm_model"]
    else:
        if resume is not None:
            if resume["format_version"] not in {3, 4, 5}:
                raise ValueError("Training can resume only from a version-3/4 checkpoint")
            text_model, image_model = resume["text_model"], resume["image_model"]
            stored_config = dict(resume["model_config"])
            for name, default in (("image_size", 256), ("image_preprocessing", "crop"),
                                  ("freeze_question_encoder", False),
                                  ("ot_visual_memory", False), ("figure_context", False),
                                  ("tie_weights", False), ("ot_num_heads", 1),
                                  ("fgw_alpha", 0.0), ("unbalanced_tau", None)):
                value = getattr(args, name)
                if value is not None and value != stored_config.get(name, default):
                    raise ValueError(f"--{name} does not match the resume checkpoint; start a fresh run")
            stored_answer_length = stored_config.get(
                "max_answer_tokens", resume.get("preprocessing", {}).get(
                    "max_answer_length", LEGACY_MAX_ANSWER_TOKENS)
            )
            if args.max_answer_tokens is not None and args.max_answer_tokens != stored_answer_length:
                raise ValueError("--max_answer_tokens does not match the resume checkpoint")
            stored_question_length = stored_config.get(
                "max_question_tokens", resume.get("preprocessing", {}).get(
                    "max_question_length", LEGACY_MAX_QUESTION_TOKENS)
            )
            if args.max_question_tokens is not None and args.max_question_tokens != stored_question_length:
                raise ValueError("--max_question_tokens does not match the resume checkpoint")
            ot_defaults = {"ot_epsilon": 0.05, "ot_iterations": 20,
                           "ot_dustbin_mass": 0.2, "ot_dustbin_cost": 1.0}
            for name, default in ot_defaults.items():
                value = getattr(args, name)
                if value is not None and value != stored_config.get(name, default):
                    raise ValueError(f"--{name} does not match the resume checkpoint")
            model_config = {
                key: value for key, value in stored_config.items() if key in MODEL_CONFIG_KEYS
            }
            model_config.setdefault("max_answer_tokens", stored_answer_length)
            model_config.setdefault("max_question_tokens", stored_question_length)
            model_config.setdefault("prompt_version", "scivqa-v1")
            model_config.setdefault("decode_version", "task-v1")
            model = VQAModel(text_model=text_model, image_model=image_model,
                             **model_config).to(device)
        else:
            text_model, image_model = args.text_model, args.image_model
            model = VQAModel(text_model=text_model, image_model=image_model,
                             prompt_version=args.prompt_version or "scivqa-v2",
                             decode_version=args.decode_version or "task-v2",
                             output_drop_prob=args.output_drop_prob,
                             output_size=args.d_model, d_model=args.d_model,
                             ffn_hidden=args.ffn_hidden, num_layers=args.num_layers,
                             num_heads=args.num_heads, drop_prob=args.drop_prob,
                             freeze_answer_embeddings=args.freeze_answer_embeddings,
                             freeze_question_encoder=bool(args.freeze_question_encoder),
                             ot_epsilon=args.ot_epsilon if args.ot_epsilon is not None else 0.05,
                             ot_iterations=args.ot_iterations if args.ot_iterations is not None else 20,
                             ot_dustbin_mass=(args.ot_dustbin_mass if args.ot_dustbin_mass is not None
                                              else 0.2),
                             ot_dustbin_cost=(args.ot_dustbin_cost if args.ot_dustbin_cost is not None
                                              else 1.0),
                             ot_num_heads=args.ot_num_heads if args.ot_num_heads is not None else 1,
                             max_answer_tokens=(args.max_answer_tokens if args.max_answer_tokens is not None
                                                else Config.MAX_LEN_ANS),
                             max_question_tokens=(args.max_question_tokens if args.max_question_tokens is not None
                                                  else Config.MAX_LEN_QUES),
                             image_size=args.image_size if args.image_size is not None else 256,
                             image_preprocessing=args.image_preprocessing or "crop",
                             ot_visual_memory=bool(args.ot_visual_memory),
                             figure_context=bool(args.figure_context),
                             tie_weights=bool(args.tie_weights),
                             fgw_alpha=args.fgw_alpha or 0.0,
                             unbalanced_tau=getattr(args, "unbalanced_tau", None)).to(device)

    trainable_summary = validate_trainable_parameters(model, device, args.amp)
    if rank == 0:
        print("Trainable parameter groups: " + json.dumps(trainable_summary), flush=True)
    train_loader = _make_loader(args, "train", True, text_model, image_model,
                                rank=rank, world_size=world_size,
                                image_size=model.image_size, image_preprocessing=model.image_preprocessing,
                                figure_context=model.figure_context, prompt_version=model.prompt_version, native=native)
    dev_loader = (_make_loader(args, "dev", False, text_model, image_model,
                               image_size=model.image_size, image_preprocessing=model.image_preprocessing,
                               figure_context=model.figure_context, prompt_version=model.prompt_version, native=native)
                  if rank == 0 else None)
    if not len(train_loader) or (rank == 0 and not len(dev_loader)):
        raise ValueError("Training and development datasets must be non-empty")
    train_eval_loader = (_make_train_eval_loader(train_loader.dataset, args, device)
                         if rank == 0 else None)
    train_criterion = AnswerCrossEntropyLoss(
        ignore_index=model.target_ignore_index, label_smoothing=args.label_smoothing,
        normalization=loss_normalization,
    )
    validation_criterion = nn.CrossEntropyLoss(ignore_index=model.target_ignore_index)
    bert_scorer = build_bertscore_scorer(
        model_type=args.bertscore_model,
        device=str(resolve_device(args.bertscore_device)),
        batch_size=args.bertscore_batch_size,
        rescale_with_baseline=args.bertscore_rescale,
    ) if rank == 0 and dataset_name != "gqa" and args.compute_bertscore else None
    trainable_parameters = [parameter for parameter in model.parameters()
                            if parameter.requires_grad]
    optimizer = _build_optimizer(model, args, resume)
    if rank == 0:
        print(f"Trainable parameters: {sum(parameter.numel() for parameter in trainable_parameters):,}")
    total_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16", init_scale=args.amp_init_scale)
    warmup_steps = math.ceil(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    start_epoch = global_step = 0
    best_metric = None
    best_val_loss = float("inf")
    best_val_rouge = float("-inf")
    epochs_without_improvement = 0
    if resume is not None:
        start_epoch, global_step, best_metric = restore_training_state(
            resume, model, optimizer, scheduler, scaler=scaler, rank=rank
        )
        epochs_without_improvement = int(resume.get("epochs_without_improvement", 0))
        if best_metric:
            if best_metric.get("selection_metric") == "loss":
                best_val_loss = float(best_metric.get("value", best_metric.get("loss", float("inf"))))
            elif best_metric.get("selection_metric") == "rouge_l_f1":
                best_val_rouge = float(best_metric.get("value", float("-inf")))
        bookkeeping = resume.get("bookkeeping", {})
        best_val_loss = bookkeeping.get("best_val_loss", best_val_loss)
        best_val_rouge = bookkeeping.get("best_val_rouge", best_val_rouge)
        if resume.get("format_version", 0) < 5 and rank == 0:
            print("Historical resume lacks per-rank RNG/scaler/full schedule: continuation is not exact.")
    if start_epoch >= args.epochs:
        raise ValueError("Resume checkpoint has already reached the requested epoch count")
    if extend_schedule:
        # The restored optimizer may have LR=0 at the old schedule's end.
        # Recreate the linear schedule at its saved update index so the very
        # first resumed update uses the enlarged horizon, without resetting Adam.
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
            last_epoch=scheduler.last_epoch - 1,
        )
        if rank == 0:
            print(f"Extending total epochs: {saved_epochs} -> {args.epochs}; "
                  f"recomputed LR schedule at optimizer update {scheduler.last_epoch} "
                  f"with {warmup_steps} warmup / {total_steps} total steps. "
                  "Optimizer moments and early-stopping state retained.", flush=True)
    if rank == 0:
        print(f"Global batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}; "
              f"warmup steps: {warmup_steps}; checkpoint selection: {selection_metric}")
        print(f"Training loss normalization: {loss_normalization}; validation loss normalization: token")
        print("Peak learning rates: " + ", ".join(
            f"{group.get('group_name', 'all')}={group['initial_lr']:g}"
            for group in optimizer.param_groups))
    training_model = (DistributedDataParallel(
        model, device_ids=[device.index], output_device=device.index,
        find_unused_parameters=True, broadcast_buffers=False,
    ) if world_size > 1 else model)

    if rank == 0:
        destination.mkdir(parents=True, exist_ok=True)
    if rank == 0 and resume is None:
        manifest = {
            **source_manifest(destination),
            "model_config": model.model_config,
            "arguments": vars(args),
            "data_preflight": data_preflight,
            "trainable_parameter_groups": trainable_summary,
            "train_examples": len(train_loader.dataset),
            "dev_examples": len(dev_loader.dataset),
            "train_csv_sha256": _csv_digest(args.train_csv_path),
            "dev_csv_sha256": _csv_digest(args.dev_csv_path),
            "torch_version": torch.__version__,
            "dataset": dataset_name,
            "generated_metrics": list(metric_names),
            "selection_metric": selection_metric,
            "world_size": world_size,
            "global_batch_size": args.batch_size * world_size * args.gradient_accumulation_steps,
            "warmup_steps": warmup_steps,
            "train_eval_examples": len(train_eval_loader.dataset) if train_eval_loader else 0,
            "answer_postprocessing": ANSWER_POSTPROCESSING,
            "loss_normalization": loss_normalization,
        }
        if bert_scorer is not None:
            manifest["bertscore_hash"] = bert_scorer.hash
        if dataset_name == "scivqa":
            manifest["rouge"] = {"implementation": "rouge-score", "use_stemmer": True,
                                 "aggregation": "mean of per-answer scores"}
        (destination / "run_config.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
        )
    metrics_path = destination / "metrics.jsonl"
    if rank == 0 and best_metric is not None and "epoch" not in best_metric:
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
                if old_row.get("val_loss") == best_metric.get("loss"):
                    best_metric["epoch"] = old_row.get("epoch", "unknown")
    if rank == 0 and resume is None:
        metrics_path.write_text("", encoding="utf-8")
    history = []
    from test import evaluation
    for epoch in range(start_epoch, args.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        train_output = train(
            training_model, train_loader, 1, optimizer, scheduler, train_criterion,
            device=device,
            epoch_offset=epoch, total_epochs=args.epochs,
            gradient_clip=args.gradient_clip or None,
            ot_loss_weight=args.ot_loss_weight,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            amp=args.amp, scaler=scaler, log_every_updates=args.log_every_updates,
            max_updates=None if args.max_updates is None else args.max_updates-global_step,
        )
        epoch_losses = train_output[1]
        epoch_ot_losses = train_output[2] if len(train_output) > 2 else []
        train_loss = epoch_losses[-1]
        telemetry = train_output[3]
        global_step += telemetry["optimizer_updates"]
        rank_rng = gather_rng()
        rank_telemetry = [telemetry]
        if world_size > 1:
            rank_telemetry = [None] * world_size
            dist.all_gather_object(rank_telemetry, telemetry)
        stop = False
        if rank == 0:
            train_scores = {}
            if train_eval_loader is not None:
                training = evaluation(
                    model, train_eval_loader, validation_criterion, device=device,
                    bert_scorer=bert_scorer, dataset=dataset_name, compute_bertscore=args.compute_bertscore,
                )
                train_scores = training["metrics"]
            train_msg = _format_epoch_metrics(epoch + 1, args.epochs, "Train", train_loss,
                                               train_scores)
            if epoch_ot_losses and getattr(args, "ot_loss_weight", 0.0):
                train_msg += f", ot_loss={epoch_ot_losses[-1]:.4f}"
            print(train_msg)
            validation = evaluation(
                model, dev_loader, validation_criterion, device=device, measure_performance=True,
                bert_scorer=bert_scorer, dataset=dataset_name, compute_bertscore=args.compute_bertscore,
            )
            val_loss, val_scores = validation["loss"], validation["metrics"]
            current, improved = _validation_selection(
                validation, epoch + 1, selection_metric, best_metric)
            if improved:
                best_metric = current
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            restore_rng(rank_rng[rank])
            # Preserve independent bests in every checkpoint, including last.pt.
            old_loss, old_rouge = best_val_loss, best_val_rouge
            best_val_loss = min(best_val_loss, val_loss)
            best_val_rouge = max(best_val_rouge, val_scores.get("rouge_l_f1", float("-inf")))
            checkpoint_args = dict(
                model=model, text_model=text_model, image_model=image_model,
                optimizer=optimizer, scheduler=scheduler,
                epoch=epoch + 1, global_step=global_step, best_metric=best_metric,
                epochs_without_improvement=epochs_without_improvement,
                scaler=scaler, rng_by_rank=rank_rng,
                bookkeeping={"best_val_loss": best_val_loss, "best_val_rouge": best_val_rouge},
                training_config={"loss_normalization": loss_normalization, "arguments": vars(args),
                                 "world_size": world_size, "complete_epoch": telemetry["complete_epoch"],
                                 "train_csv_sha256": _csv_digest(args.train_csv_path),
                                 "dev_csv_sha256": _csv_digest(args.dev_csv_path)},
            )
            save_checkpoint(destination / "last.pt", **checkpoint_args)
            if args.save_every_epoch:
                save_checkpoint(destination / f"epoch_{epoch + 1:04d}.pt", **checkpoint_args)
            if improved:
                save_checkpoint(destination / "best.pt", **checkpoint_args)

            improved_loss = False
            if val_loss < old_loss:
                best_val_loss = val_loss
                improved_loss = True
                loss_checkpoint_args = dict(checkpoint_args)
                loss_checkpoint_args["best_metric"] = {
                    "loss": val_loss, "epoch": epoch + 1,
                    "selection_metric": "loss", "value": val_loss,
                }
                save_checkpoint(destination / "best_loss.pt", **loss_checkpoint_args)

            improved_rouge = False
            current_rouge = val_scores.get("rouge_l_f1")
            if current_rouge is not None and current_rouge > old_rouge:
                best_val_rouge = current_rouge
                improved_rouge = True
                rouge_checkpoint_args = dict(checkpoint_args)
                rouge_checkpoint_args["best_metric"] = {
                    "loss": val_loss, "epoch": epoch + 1,
                    "selection_metric": "rouge_l_f1", "value": current_rouge,
                }
                save_checkpoint(destination / "best_rouge.pt", **rouge_checkpoint_args)

            row = {"epoch": epoch + 1, "global_step": global_step, "runtime_by_rank": rank_telemetry, "train_loss": train_loss,
                   "train_loss_normalization": loss_normalization,
                   **{f"train_{name}": score for name, score in train_scores.items()},
                   "val_loss": val_loss, "validation_performance": validation.get("performance"),
                   "validation_memory": cuda_memory_report(device),
                   "val_guardrails": validation.get("guardrails"),
                   "val_metrics_by_qa_pair_type": validation.get("metrics_by_qa_pair_type", {}),
                   **{f"val_{name}": val_scores[name] for name in metric_names},
                   "learning_rate": scheduler.get_last_lr()[0],
                   "learning_rates": {group.get("group_name", "all"): group["lr"]
                                      for group in optimizer.param_groups},
                   "selection_metric": selection_metric,
                   "selection_value": current["value"],
                   "train_eval_examples": len(train_eval_loader.dataset) if train_eval_loader else 0,
                   "improved": improved,
                   "improved_loss": improved_loss,
                   "improved_rouge": improved_rouge,
                   "epochs_without_improvement": epochs_without_improvement}
            if epoch_ot_losses and getattr(args, "ot_loss_weight", 0.0):
                row["train_ot_loss"] = epoch_ot_losses[-1]
            history.append(row)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            extra_tags = []
            if improved_loss and selection_metric != "loss":
                extra_tags.append(f"best val_loss={val_loss:.4f} -> best_loss.pt")
            if improved_rouge and selection_metric != "rouge_l_f1":
                extra_tags.append(f"best rouge_l={current_rouge:.4f} -> best_rouge.pt")
            tag_str = f" [{', '.join(extra_tags)}]" if extra_tags else ""
            print(_format_epoch_metrics(epoch + 1, args.epochs, "Validation", val_loss,
                                        val_scores) +
                  f", best epoch={best_metric['epoch']}, "
                  f"selection={selection_metric}, "
                  f"patience={epochs_without_improvement}/{args.early_stopping_patience or 'off'}" +
                  tag_str)
            if (args.early_stopping_patience and
                    epochs_without_improvement >= args.early_stopping_patience):
                print(f"Early stopping at epoch {epoch + 1}; best checkpoint is epoch "
                      f"{best_metric['epoch']} with validation {selection_metric}="
                      f"{best_metric.get('value', best_metric['loss']):.4f}.")
                stop = True
        if args.max_updates is not None and global_step >= args.max_updates:
            stop = True
        if world_size > 1:
            stop_signal = torch.tensor([int(stop)], device=device)
            dist.broadcast(stop_signal, src=0)
            stop = bool(stop_signal.item())
        if stop:
            break

    if rank == 0:
        plt.figure(figsize=(10, 6))
        for metric in metric_names:
            training_rows = [row for row in history if f"train_{metric}" in row]
            if training_rows:
                plt.plot([row["epoch"] for row in training_rows],
                         [row[f"train_{metric}"] for row in training_rows],
                         label=f"train_{metric}", linestyle="--")
            plt.plot([row["epoch"] for row in history],
                     [row[f"val_{metric}"] for row in history], label=f"val_{metric}")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(destination / "evaluation_metrics_plot.png")
        plt.close()


def main():
    args = get_args()
    device, rank, world_size = _training_device(args.device)
    try:
        _run_training(args, device, rank, world_size)
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
