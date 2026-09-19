"""Evaluate teacher-forced loss and answer quality from autoregressive generation."""

from pathlib import Path
import time
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader

from configs.arg_parser import get_args
from configs.config import Config
from utils.checkpoint import load_model
from utils.data_processing import load_dataframe
from utils.device import resolve_device
from utils.feature_cache import FeatureCacheDataset, collate_feature_cache
from utils.metrics import compute_em_and_f1
from utils.vqa_dataset import VQADataset, resolve_image_root


def evaluation(model, test_loader, criterion, vocab_swap=None, device=None,
               diagnostics=False):
    model.eval()
    device = device or next(model.parameters()).device
    total_loss = total_em = total_f1 = 0.0
    examples = tokens = 0
    all_hypotheses = []
    diagnostic_rows = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                answers = batch["answers"]
                image_features = batch["image_features"].to(device)
                question_features = batch["question_features"].to(device)
                question_mask = batch["question_padding_mask"].to(device)
                result = model.forward_from_features(
                    image_features, question_features, question_mask, answers,
                    return_diagnostics=diagnostics,
                )
                generated = model.generate_from_features(
                    image_features, question_features, question_mask,
                    return_diagnostics=diagnostics,
                )
            else:
                anno_ids, images, questions, answers = batch
                images = images.to(device)
                result = model(images, questions, answers, anno_ids,
                               return_diagnostics=diagnostics)
                generated = model.generate(images, questions, anno_ids,
                                           return_diagnostics=diagnostics)
            if diagnostics:
                logits, targets, transport = result
                ids = generated.generated_ids
                row = {"_count": len(answers)}
                if transport is not None:
                    row.update({
                        "ot_transport_cost": transport.transport_cost.mean().item(),
                        "ot_entropy": transport.entropy.mean().item(),
                        "ot_matched_mass": transport.matched_mass.mean().item(),
                        "ot_unmatched_mass": transport.unmatched_mass.mean().item(),
                        "ot_residual": transport.residual.mean().item(),
                        "ot_iterations": transport.iterations.float().mean().item(),
                        "ot_convergence_rate": transport.converged.float().mean().item(),
                    })
                    if generated.ot_san is not None:
                        row.update({
                            "fusion_ot_san_gate": generated.ot_san.gate.item(),
                            "fusion_ot_san_summary_norm": generated.ot_san.summary_norm.mean().item(),
                            "fusion_ot_san_attention_entropy": (
                                generated.ot_san.attention_entropy.mean().item()
                            ),
                        })
                if (generated.fusion_output is not None and
                        generated.fusion_output.diagnostics is not None):
                    for key, value in generated.fusion_output.diagnostics.items():
                        row[f"fusion_{key}"] = value.detach().float().mean().item()
                diagnostic_rows.append(row)
            else:
                logits, targets = result
                ids = generated
            loss = criterion(logits.transpose(1, 2), targets)
            count_tokens = targets.ne(model.pad_token_id).sum().item()
            total_loss += loss.item() * count_tokens
            tokens += count_tokens
            hypotheses = model.answers_from_ids(ids)
            all_hypotheses.extend(hypotheses)
            em, f1 = compute_em_and_f1(answers, hypotheses)
            count = len(answers)
            examples += count
            total_em += em * count
            total_f1 += f1 * count
    if not examples:
        raise ValueError("Evaluation dataset is empty")
    scores = (total_loss / max(tokens, 1), total_em / examples, total_f1 / examples)
    if not diagnostics:
        return scores
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if diagnostic_rows:
        diagnostic_examples = sum(row["_count"] for row in diagnostic_rows)
        summary = {
            key: sum(row[key] * row["_count"] for row in diagnostic_rows) / diagnostic_examples
            for key in diagnostic_rows[0] if key != "_count"
        }
    else:
        summary = {}
    prediction_counts = Counter(all_hypotheses)
    summary["unique_predictions"] = len(prediction_counts)
    summary["top_prediction_fraction"] = (
        max(prediction_counts.values()) / examples if prediction_counts else 0.0
    )
    summary["latency_ms_per_example"] = elapsed * 1000 / examples
    if device.type == "cuda":
        summary["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return (*scores, summary)


def main():
    args = get_args()
    device = resolve_device(args.device)
    print(f"Evaluating on device: {device}")
    default = Path(args.model_path) / "best.pt"
    checkpoint = Path(args.checkpoint) if args.checkpoint else default
    model = load_model(checkpoint, device)
    csv_path = args.dev_csv_path if args.split == "dev" else args.test_csv_path
    if args.feature_cache:
        if model.fusion_type == "san":
            raise ValueError("Feature caches require a token-level fusion checkpoint")
        dataset = FeatureCacheDataset(
            args.feature_cache, csv_path, model.text_model_name, model.image_model_name
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_feature_cache)
    else:
        frame = load_dataframe(csv_path)
        split_image_path = resolve_image_root(
            frame,
            args.img_path,
            args.split,
            override=(args.dev_img_path if args.split == "dev" else args.test_img_path),
        )
        dataset = VQADataset(
            frame, transform=Config.transforms,
            img_path=split_image_path,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    result = evaluation(
        model, loader, nn.CrossEntropyLoss(ignore_index=model.pad_token_id),
        device=device, diagnostics=args.diagnostics,
    )
    loss, em, f1 = result[:3]
    print(f"{args.split} loss: {loss:.4f}, generated EM: {em:.4f}, F1: {f1:.4f}")
    if args.diagnostics:
        print("Fusion diagnostics:", result[3])


if __name__ == "__main__":
    main()
