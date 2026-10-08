"""Evaluate teacher-forced loss and generated VQA answer quality."""

import json
import hashlib
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from configs.arg_parser import get_args
from configs.config import Config
from utils.checkpoint import load_model
from utils.data_processing import load_dataframe
from utils.device import resolve_device
from utils.metrics import (build_bertscore_scorer, gqa_score_pairs, mean_scores,
                           metrics_for_dataset, resolve_dataset, score_pairs)
from utils.vqa_dataset import VQADataset, resolve_image_root


def evaluation(model, test_loader, criterion, vocab_swap=None, device=None,
               measure_performance=False, predictions=None, bert_scorer=None,
               dataset=None):
    dataset_name = resolve_dataset(
        dataset if dataset is not None else
        getattr(test_loader.dataset, "dataset_name", "plantexpert")
    )
    model.eval()
    device = device or next(model.parameters()).device
    total_loss = 0.0
    examples = tokens = 0
    references, generated, annotations, question_texts = [], [], [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            anno_ids, images, questions, answers = batch
            images = images.to(device)
            logits, targets, ids = model.evaluate_batch(images, questions, answers, anno_ids)
            loss = criterion(logits.transpose(1, 2), targets)
            count_tokens = targets.ne(model.pad_token_id).sum().item()
            total_loss += loss.item() * count_tokens
            tokens += count_tokens
            hypotheses = model.answers_from_ids(ids)
            references.extend(answers)
            generated.extend(hypotheses)
            if predictions is not None:
                annotation_values = (anno_ids.tolist() if torch.is_tensor(anno_ids)
                                     else anno_ids)
                annotations.extend(annotation_values)
                question_texts.extend(questions)
            count = len(answers)
            examples += count
    if not examples:
        raise ValueError("Evaluation dataset is empty")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_elapsed = perf_counter() - started
    if dataset_name == "gqa":
        rows = gqa_score_pairs(references, generated)
    else:
        if bert_scorer is None:
            bert_scorer = build_bertscore_scorer()
        rows = score_pairs(references, generated, bert_scorer)
    scores = mean_scores(rows, metrics_for_dataset(dataset_name))
    if predictions is not None:
        for annotation, question, reference, hypothesis, row in zip(
                annotations, question_texts, references, generated, rows):
            predictions.append({
                "anno_id": str(annotation), "question": question,
                "reference": reference, "prediction": hypothesis, **row,
            })
    result = {"loss": total_loss / max(tokens, 1), "metrics": scores,
              "examples": examples}
    if measure_performance:
        result["performance"] = {
            "examples": examples,
            "elapsed_seconds": model_elapsed,
            "examples_per_second": examples / model_elapsed,
            "milliseconds_per_example": 1000 * model_elapsed / examples,
            "peak_cuda_bytes": (torch.cuda.max_memory_allocated(device)
                                if device.type == "cuda" else None),
        }
    return result


def main():
    args = get_args()
    device = resolve_device(args.device)
    print(f"Evaluating on device: {device}")
    default = Path(args.model_path) / "best.pt"
    checkpoint = Path(args.checkpoint) if args.checkpoint else default
    model = load_model(checkpoint, device)
    if args.fusion is not None and args.fusion != model.fusion:
        raise ValueError(f"--fusion {args.fusion} does not match checkpoint fusion {model.fusion}")
    print(f"Fusion method: {model.fusion}")
    csv_path = args.dev_csv_path if args.split == "dev" else args.test_csv_path
    dataset_name = resolve_dataset(args.dataset, csv_path)
    print(f"Dataset: {dataset_name}")
    frame = load_dataframe(csv_path)
    split_image_path = resolve_image_root(
        frame, args.img_path, args.split,
        override=(args.dev_img_path if args.split == "dev" else args.test_img_path),
    )
    dataset = VQADataset(frame, transform=Config.transforms, img_path=split_image_path,
                         dataset_name=dataset_name)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    predictions = [] if args.predictions_csv else None
    scorer = build_bertscore_scorer(
        model_type=args.bertscore_model,
        device=str(resolve_device(args.bertscore_device)),
        batch_size=args.bertscore_batch_size,
        rescale_with_baseline=args.bertscore_rescale,
    ) if dataset_name != "gqa" else None
    result = evaluation(model, loader, nn.CrossEntropyLoss(ignore_index=model.pad_token_id),
                        device=device, measure_performance=True, predictions=predictions,
                        bert_scorer=scorer)
    print(f"{args.split} loss: {result['loss']:.4f}, " + ", ".join(
        f"{name}={value:.4f}" for name, value in result["metrics"].items()))
    performance = result["performance"]
    print(f"Throughput: {performance['examples_per_second']:.2f} examples/s; "
          f"latency: {performance['milliseconds_per_example']:.1f} ms/example; "
          f"peak CUDA memory: {performance['peak_cuda_bytes']}")
    if predictions is not None:
        output = Path(args.predictions_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        prediction_frame = pd.DataFrame(predictions)
        if "question_type" in frame:
            prediction_frame["question_type"] = frame["question_type"].tolist()
        prediction_frame.to_csv(output, index=False)
        print(f"Wrote predictions to {output}")
    if args.report_json:
        output = Path(args.report_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with Path(csv_path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        report = {
            "checkpoint": str(checkpoint), "fusion": model.fusion, "split": args.split,
            "dataset": dataset_name,
            "evaluation_csv_sha256": digest.hexdigest(),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "loss": result["loss"], "generated_metrics": result["metrics"],
            "performance": performance,
        }
        if scorer is not None:
            report["bertscore"] = {
                "model": args.bertscore_model, "model_hash": scorer.hash,
                "rescale_with_baseline": args.bertscore_rescale,
                "batch_size": args.bertscore_batch_size,
                "device": str(resolve_device(args.bertscore_device)),
            }
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote evaluation report to {output}")


if __name__ == "__main__":
    main()
