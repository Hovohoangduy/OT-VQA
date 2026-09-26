"""Train SAN or OT VQA and select checkpoints by validation loss."""

from __future__ import annotations

import json
import hashlib
import os
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup

from configs.arg_parser import get_args
from configs.config import Config
from model.vqa_model import VQAModel
from utils.checkpoint import MODEL_CONFIG_KEYS, read_checkpoint, restore_training_state, save_checkpoint
from utils.data_processing import load_dataframe
from utils.device import resolve_device, seed_everything
from utils.metrics import PAPER_METRICS, build_bertscore_scorer
from utils.vqa_dataset import VQADataset, resolve_image_root


def _forward_batch(model, batch, device):
    anno_ids, images, questions, answers = batch
    return model(images.to(device), questions, answers, anno_ids)


def _format_epoch_metrics(epoch, total_epochs, split, loss, scores):
    metrics = ", ".join(f"{name}={scores[name]:.4f}" for name in PAPER_METRICS)
    return f"Epoch {epoch}/{total_epochs} {split}: loss={loss:.4f}, {metrics}"


def train(model, train_loader, num_epochs, optimizer, scheduler, criterion,
          vocab_swap=None, device=None, epoch_offset=0,
          total_epochs=None, gradient_clip=None):
    """Run teacher-forced optimization; score generated answers after this step."""
    device = device or next(model.parameters()).device
    base_model = model.module if isinstance(model, DistributedDataParallel) else model
    losses, epoch_losses = [], []
    if len(train_loader) == 0:
        raise ValueError("Training dataset is empty")
    for epoch in range(num_epochs):
        displayed_epoch = epoch_offset + epoch + 1
        displayed_total = total_epochs if total_epochs is not None else epoch_offset + num_epochs
        model.train()
        total_loss = 0.0
        tokens = 0
        for batch_idx, batch in enumerate(train_loader):
            logits, targets = _forward_batch(model, batch, device)
            loss = criterion(logits.transpose(1, 2), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is NaN or infinity")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    gradient_clip,
                )
            optimizer.step()
            scheduler.step()
            batch_tokens = targets.ne(base_model.pad_token_id).sum().item()
            total_loss += loss.item() * batch_tokens
            tokens += batch_tokens
            losses.append(loss.item())
            if (batch_idx + 1) % 2000 == 0:
                print(f"Epoch {displayed_epoch}/{displayed_total}, "
                      f"batch {batch_idx + 1}: loss={loss.item():.4f}")
        if dist.is_available() and dist.is_initialized():
            totals = torch.tensor([total_loss, tokens], dtype=torch.float64, device=device)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            total_loss, tokens = totals.tolist()
        epoch_losses.append(total_loss / max(tokens, 1))
    return losses, epoch_losses


def _make_loader(args, split, shuffle, text_model, image_model, rank=0, world_size=1):
    csv_path = getattr(args, f"{split}_csv_path")
    frame = load_dataframe(csv_path)
    image_path = resolve_image_root(
        frame,
        args.img_path,
        split,
        override=getattr(args, f"{split}_img_path"),
    )
    dataset = VQADataset(frame, transform=Config.transforms, img_path=image_path)
    sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                  shuffle=shuffle, seed=args.seed)
               if world_size > 1 and split == "train" else None)
    return DataLoader(dataset, batch_size=args.batch_size,
                      shuffle=shuffle and sampler is None, sampler=sampler)


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
    if args.batch_size < 1 or args.epochs < 1:
        raise ValueError("batch_size and epochs must be positive")
    if args.max_answer_tokens is not None and args.max_answer_tokens < 2:
        raise ValueError("max_answer_tokens must be at least 2")
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
    if args.bertscore_batch_size < 1:
        raise ValueError("bertscore_batch_size must be positive")
    seed_everything(args.seed + rank)
    if rank == 0:
        if world_size == 1:
            print(f"Training on device: {device}")
        else:
            print(f"Training on {world_size} GPUs; primary device: {device}")

    resume = read_checkpoint(args.resume, device) if args.resume else None
    if resume is not None:
        if resume["format_version"] not in {3, 4}:
            raise ValueError("Training can resume only from a version-3/4 checkpoint")
        text_model, image_model = resume["text_model"], resume["image_model"]
        stored_config = dict(resume["model_config"])
        stored_answer_length = stored_config.get("max_answer_tokens", Config.MAX_LEN_ANS)
        if args.max_answer_tokens is not None and args.max_answer_tokens != stored_answer_length:
            raise ValueError("--max_answer_tokens does not match the resume checkpoint")
        fusion = stored_config.get("fusion", "san")
        if args.fusion is not None and args.fusion != fusion:
            raise ValueError("--fusion does not match the resume checkpoint")
        ot_defaults = {"ot_epsilon": 0.05, "ot_iterations": 20,
                       "ot_dustbin_mass": 0.2, "ot_dustbin_cost": 1.0}
        for name, default in ot_defaults.items():
            value = getattr(args, name)
            if value is not None and value != stored_config.get(name, default):
                raise ValueError(f"--{name} does not match the resume checkpoint")
        model_config = {
            key: value for key, value in stored_config.items() if key in MODEL_CONFIG_KEYS
        }
        model = VQAModel(text_model=text_model, image_model=image_model,
                         **model_config).to(device)
    else:
        text_model, image_model = args.text_model, args.image_model
        model = VQAModel(text_model=text_model, image_model=image_model,
                         output_size=args.d_model, d_model=args.d_model,
                         ffn_hidden=args.ffn_hidden, num_layers=args.num_layers,
                         num_heads=args.num_heads, drop_prob=args.drop_prob,
                         freeze_answer_embeddings=args.freeze_answer_embeddings,
                         fusion=args.fusion or "ot",
                         ot_epsilon=args.ot_epsilon if args.ot_epsilon is not None else 0.05,
                         ot_iterations=args.ot_iterations if args.ot_iterations is not None else 20,
                         ot_dustbin_mass=(args.ot_dustbin_mass if args.ot_dustbin_mass is not None
                                          else 0.2),
                         ot_dustbin_cost=(args.ot_dustbin_cost if args.ot_dustbin_cost is not None
                                          else 1.0),
                         max_answer_tokens=(args.max_answer_tokens if args.max_answer_tokens is not None
                                            else Config.MAX_LEN_ANS)).to(device)

    train_loader = _make_loader(args, "train", True, text_model, image_model,
                                rank=rank, world_size=world_size)
    dev_loader = (_make_loader(args, "dev", False, text_model, image_model)
                  if rank == 0 else None)
    if not len(train_loader) or (rank == 0 and not len(dev_loader)):
        raise ValueError("Training and development datasets must be non-empty")
    train_eval_loader = (DataLoader(train_loader.dataset, batch_size=args.batch_size,
                                    shuffle=False) if rank == 0 else None)
    train_criterion = nn.CrossEntropyLoss(
        ignore_index=model.pad_token_id, label_smoothing=args.label_smoothing
    )
    validation_criterion = nn.CrossEntropyLoss(ignore_index=model.pad_token_id)
    bert_scorer = build_bertscore_scorer(
        model_type=args.bertscore_model,
        device=str(resolve_device(args.bertscore_device)),
        batch_size=args.bertscore_batch_size,
        rescale_with_baseline=args.bertscore_rescale,
    ) if rank == 0 else None
    trainable_parameters = [parameter for parameter in model.parameters()
                            if parameter.requires_grad]
    optimizer = optim.AdamW(
        trainable_parameters,
        lr=args.lr if args.lr is not None else Config.lr,
        weight_decay=args.weight_decay,
    )
    if rank == 0:
        print(f"Trainable parameters: {sum(parameter.numel() for parameter in trainable_parameters):,}")
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * args.epochs
    )
    start_epoch = global_step = 0
    best_metric = None
    epochs_without_improvement = 0
    if resume is not None:
        start_epoch, global_step, best_metric = restore_training_state(
            resume, model, optimizer, scheduler
        )
        epochs_without_improvement = int(resume.get("epochs_without_improvement", 0))
    if start_epoch >= args.epochs:
        raise ValueError("Resume checkpoint has already reached the requested epoch count")
    training_model = (DistributedDataParallel(
        model, device_ids=[device.index], output_device=device.index,
        find_unused_parameters=True,
    ) if world_size > 1 else model)

    destination = Path(args.model_path)
    if rank == 0:
        destination.mkdir(parents=True, exist_ok=True)
    if rank == 0 and resume is None:
        manifest = {
            "model_config": model.model_config,
            "arguments": vars(args),
            "train_examples": len(train_loader.dataset),
            "dev_examples": len(dev_loader.dataset),
            "train_csv_sha256": _csv_digest(args.train_csv_path),
            "dev_csv_sha256": _csv_digest(args.dev_csv_path),
            "torch_version": torch.__version__,
            "bertscore_hash": bert_scorer.hash,
        }
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
        _, epoch_losses = train(
            training_model, train_loader, 1, optimizer, scheduler, train_criterion,
            device=device,
            epoch_offset=epoch, total_epochs=args.epochs,
            gradient_clip=args.gradient_clip or None,
        )
        train_loss = epoch_losses[-1]
        global_step += len(train_loader)
        stop = False
        if rank == 0:
            training = evaluation(
                model, train_eval_loader, validation_criterion, device=device,
                bert_scorer=bert_scorer,
            )
            train_scores = training["metrics"]
            print(_format_epoch_metrics(epoch + 1, args.epochs, "Train", train_loss,
                                        train_scores))
            validation = evaluation(
                model, dev_loader, validation_criterion, device=device,
                bert_scorer=bert_scorer,
            )
            val_loss, val_scores = validation["loss"], validation["metrics"]
            current = {"loss": val_loss, "epoch": epoch + 1}
            improved = best_metric is None or val_loss < best_metric["loss"]
            if improved:
                best_metric = current
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            checkpoint_args = dict(
                model=model, text_model=text_model, image_model=image_model,
                optimizer=optimizer, scheduler=scheduler,
                epoch=epoch + 1, global_step=global_step, best_metric=best_metric,
                epochs_without_improvement=epochs_without_improvement,
            )
            save_checkpoint(destination / "last.pt", **checkpoint_args)
            if args.save_every_epoch:
                save_checkpoint(destination / f"epoch_{epoch + 1:04d}.pt", **checkpoint_args)
            if improved:
                save_checkpoint(destination / "best.pt", **checkpoint_args)
            row = {"epoch": epoch + 1, "train_loss": train_loss,
                   **{f"train_{name}": train_scores[name] for name in PAPER_METRICS},
                   "val_loss": val_loss,
                   **{f"val_{name}": val_scores[name] for name in PAPER_METRICS},
                   "learning_rate": scheduler.get_last_lr()[0],
                   "improved": improved,
                   "epochs_without_improvement": epochs_without_improvement}
            history.append(row)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(_format_epoch_metrics(epoch + 1, args.epochs, "Validation", val_loss,
                                        val_scores) +
                  f", best epoch={best_metric['epoch']}, "
                  f"patience={epochs_without_improvement}/{args.early_stopping_patience or 'off'}")
            if (args.early_stopping_patience and
                    epochs_without_improvement >= args.early_stopping_patience):
                print(f"Early stopping at epoch {epoch + 1}; best checkpoint is epoch "
                      f"{best_metric['epoch']} with validation loss={best_metric['loss']:.4f}.")
                stop = True
        if world_size > 1:
            stop_signal = torch.tensor([int(stop)], device=device)
            dist.broadcast(stop_signal, src=0)
            stop = bool(stop_signal.item())
        if stop:
            break

    if rank == 0:
        plt.figure(figsize=(10, 6))
        for metric in PAPER_METRICS:
            plt.plot([row["epoch"] for row in history],
                     [row[f"train_{metric}"] for row in history],
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
