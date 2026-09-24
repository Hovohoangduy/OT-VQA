"""Train SAN or OT VQA and select checkpoints by generated F1."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup

from configs.arg_parser import get_args
from configs.config import Config
from model.vqa_model import VQAModel
from utils.checkpoint import MODEL_CONFIG_KEYS, read_checkpoint, restore_training_state, save_checkpoint
from utils.data_processing import load_dataframe
from utils.device import resolve_device, seed_everything
from utils.metrics import compute_em_and_f1
from utils.vqa_dataset import VQADataset, resolve_image_root


def _forward_batch(model, batch, device):
    anno_ids, images, questions, answers = batch
    return model(images.to(device), questions, answers, anno_ids), answers


def train(model, train_loader, num_epochs, optimizer, scheduler, criterion,
          vocab_swap=None, device=None, epoch_offset=0,
          total_epochs=None, gradient_clip=None):
    """Run teacher-forced optimization; generation is reserved for validation."""
    device = device or next(model.parameters()).device
    losses, em_scores, f1_scores = [], [], []
    if len(train_loader) == 0:
        raise ValueError("Training dataset is empty")
    for epoch in range(num_epochs):
        displayed_epoch = epoch_offset + epoch + 1
        displayed_total = total_epochs if total_epochs is not None else epoch_offset + num_epochs
        model.train()
        total_loss = total_em = total_f1 = 0.0
        examples = tokens = 0
        for batch_idx, batch in enumerate(train_loader):
            (logits, targets), answers = _forward_batch(model, batch, device)
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
            batch_tokens = targets.ne(model.pad_token_id).sum().item()
            total_loss += loss.item() * batch_tokens
            tokens += batch_tokens
            losses.append(loss.item())
            hypotheses = model.answers_from_ids(logits.detach().argmax(-1))
            em, f1 = compute_em_and_f1(answers, hypotheses)
            count = len(answers)
            total_em += em * count
            total_f1 += f1 * count
            examples += count
            if (batch_idx + 1) % 2000 == 0:
                print(f"Epoch {displayed_epoch}, batch {batch_idx + 1}: loss={loss.item():.4f}")
        em_scores.append(total_em / examples)
        f1_scores.append(total_f1 / examples)
        message = (f"Epoch {displayed_epoch}/{displayed_total}: "
                   f"loss={total_loss / max(tokens, 1):.4f}, "
                   f"teacher-forced EM={em_scores[-1]:.4f}, F1={f1_scores[-1]:.4f}")
        print(message)
    return losses, em_scores, f1_scores


def _make_loader(args, split, shuffle, text_model, image_model):
    csv_path = getattr(args, f"{split}_csv_path")
    frame = load_dataframe(csv_path)
    image_path = resolve_image_root(
        frame,
        args.img_path,
        split,
        override=getattr(args, f"{split}_img_path"),
    )
    dataset = VQADataset(frame, transform=Config.transforms, img_path=image_path)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)


def _csv_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if not 0.0 <= args.drop_prob < 1.0:
        raise ValueError("drop_prob must be in [0, 1)")
    if args.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    if args.gradient_clip < 0:
        raise ValueError("gradient_clip cannot be negative")
    if args.train_eval_samples < 0:
        raise ValueError("train_eval_samples cannot be negative")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"Training on device: {device}")

    resume = read_checkpoint(args.resume, device) if args.resume else None
    if resume is not None:
        if resume["format_version"] not in {3, 4}:
            raise ValueError("Training can resume only from a version-3/4 checkpoint")
        text_model, image_model = resume["text_model"], resume["image_model"]
        stored_config = dict(resume["model_config"])
        fusion = stored_config.get("fusion", "san")
        if args.fusion is not None and args.fusion != fusion:
            raise ValueError("--fusion does not match the resume checkpoint")
        if (args.freeze_text_encoder is not None and
                args.freeze_text_encoder != stored_config.get("freeze_text_encoder", False)):
            raise ValueError("--freeze_text_encoder does not match the resume checkpoint")
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
                         freeze_text_encoder=bool(args.freeze_text_encoder),
                         fusion=args.fusion or "ot",
                         ot_epsilon=args.ot_epsilon if args.ot_epsilon is not None else 0.05,
                         ot_iterations=args.ot_iterations if args.ot_iterations is not None else 20,
                         ot_dustbin_mass=(args.ot_dustbin_mass if args.ot_dustbin_mass is not None
                                          else 0.2),
                         ot_dustbin_cost=(args.ot_dustbin_cost if args.ot_dustbin_cost is not None
                                          else 1.0)).to(device)

    train_loader = _make_loader(args, "train", True, text_model, image_model)
    dev_loader = _make_loader(args, "dev", False, text_model, image_model)
    if not len(train_loader) or not len(dev_loader):
        raise ValueError("Training and development datasets must be non-empty")
    train_eval_loader = None
    if args.train_eval_samples:
        sample_count = min(args.train_eval_samples, len(train_loader.dataset))
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(train_loader.dataset), generator=generator)[:sample_count].tolist()
        train_eval_loader = DataLoader(
            Subset(train_loader.dataset, indices), batch_size=args.batch_size,
            shuffle=False, generator=torch.Generator().manual_seed(args.seed),
        )
        print(f"Generated training evaluation: {sample_count} fixed examples")
    train_criterion = nn.CrossEntropyLoss(
        ignore_index=model.pad_token_id, label_smoothing=args.label_smoothing
    )
    validation_criterion = nn.CrossEntropyLoss(ignore_index=model.pad_token_id)
    trainable_parameters = [parameter for parameter in model.parameters()
                            if parameter.requires_grad]
    optimizer = optim.AdamW(
        trainable_parameters,
        lr=args.lr if args.lr is not None else Config.lr,
        weight_decay=args.weight_decay,
    )
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

    destination = Path(args.model_path)
    destination.mkdir(parents=True, exist_ok=True)
    if resume is None:
        manifest = {
            "model_config": model.model_config,
            "arguments": vars(args),
            "train_examples": len(train_loader.dataset),
            "dev_examples": len(dev_loader.dataset),
            "train_csv_sha256": _csv_digest(args.train_csv_path),
            "dev_csv_sha256": _csv_digest(args.dev_csv_path),
            "torch_version": torch.__version__,
        }
        (destination / "run_config.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
        )
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
    if resume is None:
        metrics_path.write_text("", encoding="utf-8")
    history = []
    from test import evaluation
    for epoch in range(start_epoch, args.epochs):
        losses, train_em, train_f1 = train(
            model, train_loader, 1, optimizer, scheduler, train_criterion,
            device=device,
            epoch_offset=epoch, total_epochs=args.epochs,
            gradient_clip=args.gradient_clip or None,
        )
        global_step += len(train_loader)
        train_generated = (evaluation(
            model, train_eval_loader, validation_criterion, device=device,
        ) if train_eval_loader is not None else None)
        validation = evaluation(
            model, dev_loader, validation_criterion, device=device,
        )
        val_loss, val_em, val_f1 = validation[:3]
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
            optimizer=optimizer, scheduler=scheduler,
            epoch=epoch + 1, global_step=global_step, best_metric=best_metric,
            epochs_without_improvement=epochs_without_improvement,
        )
        save_checkpoint(destination / "last.pt", **checkpoint_args)
        if improved:
            save_checkpoint(destination / "best.pt", **checkpoint_args)
        row = {"epoch": epoch + 1, "train_loss": sum(losses) / len(losses),
               "train_em": train_em[-1], "train_f1": train_f1[-1],
               "val_loss": val_loss, "val_em": val_em, "val_f1": val_f1,
               "learning_rate": scheduler.get_last_lr()[0],
               "improved": improved,
               "epochs_without_improvement": epochs_without_improvement}
        if train_generated is not None:
            row.update(train_generated_em=train_generated[1],
                       train_generated_f1=train_generated[2],
                       train_eval_examples=len(train_eval_loader.dataset))
        history.append(row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"Validation: loss={val_loss:.4f}, generated EM={val_em:.4f}, "
              f"F1={val_f1:.4f}, best epoch={best_metric['epoch']}, "
              f"patience={epochs_without_improvement}/{args.early_stopping_patience or 'off'}")
        if train_generated is not None:
            print(f"Train generated ({len(train_eval_loader.dataset)} fixed examples): "
                  f"EM={train_generated[1]:.4f}, F1={train_generated[2]:.4f}; "
                  f"generated F1 gap={train_generated[2] - val_f1:.4f}")
        if (args.early_stopping_patience and
                epochs_without_improvement >= args.early_stopping_patience):
            print(f"Early stopping at epoch {epoch + 1}; best checkpoint is epoch "
                  f"{best_metric['epoch']} with generated F1={best_metric['f1']:.4f}.")
            break

    plt.figure(figsize=(10, 6))
    plt.plot([row["epoch"] for row in history], [row["val_em"] for row in history], label="Generated EM")
    plt.plot([row["epoch"] for row in history], [row["val_f1"] for row in history], label="Generated F1")
    if train_eval_loader is not None:
        plt.plot([row["epoch"] for row in history],
                 [row["train_generated_f1"] for row in history],
                 label="Train generated F1 (fixed subset)")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination / "evaluation_metrics_plot.png")
    plt.close()


if __name__ == "__main__":
    main()
