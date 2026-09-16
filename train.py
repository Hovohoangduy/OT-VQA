"""Train SAN or Optimal-Transport VQA and select checkpoints by generated F1."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from configs.arg_parser import get_args
from configs.config import Config
from model.optimal_transport import OTConfig
from model.vqa_model import VQAModel
from utils.checkpoint import read_checkpoint, restore_training_state, save_checkpoint
from utils.data_processing import load_dataframe
from utils.device import resolve_device, seed_everything
from utils.feature_cache import FeatureCacheDataset, collate_feature_cache
from utils.metrics import compute_em_and_f1
from utils.ViTextVQA_dataset import ViTextVQA_Dataset


def _forward_batch(model, batch, device, diagnostics=False):
    if isinstance(batch, dict):
        answers = batch["answers"]
        result = model.forward_from_features(
            batch["image_features"].to(device),
            batch["question_features"].to(device),
            batch["question_padding_mask"].to(device),
            answers, return_diagnostics=diagnostics,
        )
    else:
        anno_ids, images, questions, answers = batch
        result = model(images.to(device), questions, answers, anno_ids,
                       return_diagnostics=diagnostics)
    return result, answers


def train(model, train_loader, num_epochs, optimizer, scheduler, criterion,
          vocab_swap=None, device=None, diagnostics=False, epoch_offset=0,
          total_epochs=None):
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
        diagnostic_rows = []
        for batch_idx, batch in enumerate(train_loader):
            result, answers = _forward_batch(model, batch, device, diagnostics)
            if diagnostics:
                logits, targets, transport = result
                if transport is not None:
                    diagnostic_rows.append({
                        "matched_mass": transport.matched_mass.detach().mean().item(),
                        "entropy": transport.entropy.detach().mean().item(),
                        "residual": transport.residual.detach().mean().item(),
                        "iterations": transport.iterations.float().mean().item(),
                        "convergence": transport.converged.float().mean().item(),
                    })
            else:
                logits, targets = result
            loss = criterion(logits.transpose(1, 2), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is NaN or infinity")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
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
        if diagnostic_rows:
            means = {key: sum(row[key] for row in diagnostic_rows) / len(diagnostic_rows)
                     for key in diagnostic_rows[0]}
            message += ", OT " + ", ".join(f"{key}={value:.4g}" for key, value in means.items())
        print(message)
    return losses, em_scores, f1_scores


def _make_loader(args, split, shuffle, language, text_model, image_model):
    csv_path = getattr(args, f"{split}_csv_path")
    if args.feature_cache:
        candidate = Path(args.feature_cache) / split
        cache_path = candidate if (candidate / "manifest.json").is_file() else Path(args.feature_cache)
        cache = FeatureCacheDataset(cache_path, csv_path, text_model, image_model)
        return DataLoader(cache, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=collate_feature_cache)
    frame = load_dataframe(csv_path, language)
    image_path = getattr(args, f"{split}_img_path") or args.img_path
    dataset = ViTextVQA_Dataset(frame, transform=Config.transforms, img_path=image_path)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)


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
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"Training on device: {device}")

    resume = read_checkpoint(args.resume, device) if args.resume else None
    if resume is not None:
        if resume["format_version"] != 3:
            raise ValueError("Training can resume only from a version-3 checkpoint")
        text_model, image_model = resume["text_model"], resume["image_model"]
        language = resume.get("language", args.language)
        model = VQAModel(text_model=text_model, image_model=image_model,
                         **resume["model_config"]).to(device)
    else:
        text_model, image_model, language = args.text_model, args.image_model, args.language
        ot_config = OTConfig.from_json(args.ot_profile) if args.ot_profile else OTConfig()
        model = VQAModel(text_model=text_model, image_model=image_model,
                         output_size=args.d_model, d_model=args.d_model,
                         ffn_hidden=args.ffn_hidden, num_layers=args.num_layers,
                         num_heads=args.num_heads, drop_prob=args.drop_prob,
                         freeze_answer_embeddings=args.freeze_answer_embeddings,
                         fusion=args.fusion, ot_config=ot_config).to(device)
    if args.feature_cache and model.fusion_type == "san":
        raise ValueError("Feature caches contain token features and require OT fusion")

    train_loader = _make_loader(args, "train", True, language, text_model, image_model)
    dev_loader = _make_loader(args, "dev", False, language, text_model, image_model)
    if not len(train_loader) or not len(dev_loader):
        raise ValueError("Training and development datasets must be non-empty")
    train_criterion = nn.CrossEntropyLoss(
        ignore_index=model.pad_token_id, label_smoothing=args.label_smoothing
    )
    validation_criterion = nn.CrossEntropyLoss(ignore_index=model.pad_token_id)
    optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr if args.lr is not None else Config.lr)
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
            device=device, diagnostics=args.diagnostics and model.fusion_type != "san",
            epoch_offset=epoch, total_epochs=args.epochs,
        )
        global_step += len(train_loader)
        validation = evaluation(
            model, dev_loader, validation_criterion, device=device,
            diagnostics=args.diagnostics and model.fusion_type != "san",
        )
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
            language=language, optimizer=optimizer, scheduler=scheduler,
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
        row.update({f"val_ot_{key}": value for key, value in val_diagnostics.items()})
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
                f"OT convergence={val_diagnostics.get('convergence_rate', 0):.3f}, "
                f"residual={val_diagnostics.get('residual', 0):.5f}"
            )
        if (args.early_stopping_patience and
                epochs_without_improvement >= args.early_stopping_patience):
            print(f"Early stopping at epoch {epoch + 1}; best checkpoint is epoch "
                  f"{best_metric['epoch']} with generated F1={best_metric['f1']:.4f}.")
            break

    plt.figure(figsize=(10, 6))
    plt.plot([row["epoch"] for row in history], [row["val_em"] for row in history], label="Generated EM")
    plt.plot([row["epoch"] for row in history], [row["val_f1"] for row in history], label="Generated F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination / "evaluation_metrics_plot.png")
    plt.close()


if __name__ == "__main__":
    main()
