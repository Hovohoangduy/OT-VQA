"""Diagnose VQA overfitting, output collapse, and image/question reliance."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from configs.config import Config
from utils.checkpoint import load_model
from utils.data_processing import load_dataframe
from utils.device import resolve_device, seed_everything
from utils.metrics import compute_em_and_f1, normalize_text
from utils.vqa_dataset import VQADataset


def _latest_run(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    runs: list[list[dict]] = [[]]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if runs[-1] and row.get("epoch", 0) <= runs[-1][-1].get("epoch", 0):
            runs.append([])
        runs[-1].append(row)
    return runs[-1]


def _history_report(rows: list[dict]) -> dict:
    if not rows:
        return {}
    best = max(rows, key=lambda row: (row["val_f1"], -row["val_loss"]))
    last = rows[-1]
    return {
        "epochs_logged": len(rows),
        "best_epoch": best["epoch"],
        "best_val_f1": best["val_f1"],
        "best_val_loss": best["val_loss"],
        "last_epoch": last["epoch"],
        "last_train_f1": last["train_f1"],
        "last_val_f1": last["val_f1"],
        "last_val_loss": last["val_loss"],
        "train_validation_f1_gap": last["train_f1"] - last["val_f1"],
        "val_loss_increase_after_best": last["val_loss"] - best["val_loss"],
        "overfitting_detected": (
            last["epoch"] > best["epoch"] and
            last["train_f1"] > best.get("train_f1", 0) and
            last["val_f1"] <= best["val_f1"]
        ),
    }


def _prediction_summary(answers: list[str], predictions: list[str]) -> dict:
    em, f1 = compute_em_and_f1(answers, predictions)
    counts = Counter(predictions)
    return {
        "em": em,
        "f1": f1,
        "unique_predictions": len(counts),
        "top_predictions": counts.most_common(10),
        "top_prediction_fraction": max(counts.values()) / len(predictions),
    }


def _dataset_report(train_frame, dev_frame) -> dict:
    train_answers = [normalize_text(value) for value in train_frame["answer"]]
    dev_answers = [normalize_text(value) for value in dev_frame["answer"]]
    train_vocabulary = set(train_answers)
    unseen = sum(answer not in train_vocabulary for answer in dev_answers)
    majority_answer, majority_count = Counter(train_answers).most_common(1)[0]
    majority_em, majority_f1 = compute_em_and_f1(
        dev_answers, [majority_answer] * len(dev_answers)
    )
    return {
        "train_examples": len(train_answers),
        "validation_examples": len(dev_answers),
        "unique_train_answers": len(train_vocabulary),
        "unique_validation_answers": len(set(dev_answers)),
        "validation_examples_with_unseen_answer": unseen,
        "validation_unseen_answer_fraction": unseen / len(dev_answers),
        "train_majority_answer": majority_answer,
        "train_majority_fraction": majority_count / len(train_answers),
        "validation_majority_baseline_em": majority_em,
        "validation_majority_baseline_f1": majority_f1,
    }


def _roll(values):
    return values[-1:] + values[:-1]


def _reliance_report(model, dataset, samples: int, batch_size: int,
                     device: torch.device) -> dict:
    count = min(samples, len(dataset))
    if count < 2:
        raise ValueError("At least two validation examples are required")
    records = [dataset[index] for index in range(count)]
    original_predictions: list[str] = []
    image_shuffled_predictions: list[str] = []
    question_shuffled_predictions: list[str] = []
    answers: list[str] = []
    for start in range(0, count, batch_size):
        batch = records[start:start + batch_size]
        if len(batch) == 1:
            # Pair a final singleton with the preceding image/question.
            image_source = [records[(start - 1) % count]]
            question_source = image_source
        else:
            image_source = _roll(batch)
            question_source = _roll(batch)
        anno_ids = [row[0] for row in batch]
        images = torch.stack([row[1] for row in batch]).to(device)
        shuffled_images = torch.stack([row[1] for row in image_source]).to(device)
        questions = [row[2] for row in batch]
        shuffled_questions = [row[2] for row in question_source]
        answers.extend(row[3] for row in batch)
        with torch.no_grad():
            original = model.generate(images, questions, anno_ids)
            image_shuffled = model.generate(shuffled_images, questions, anno_ids)
            question_shuffled = model.generate(images, shuffled_questions, anno_ids)
        original_predictions.extend(model.answers_from_ids(original))
        image_shuffled_predictions.extend(model.answers_from_ids(image_shuffled))
        question_shuffled_predictions.extend(model.answers_from_ids(question_shuffled))
    result = {
        "samples": count,
        "original": _prediction_summary(answers, original_predictions),
        "image_shuffled": _prediction_summary(answers, image_shuffled_predictions),
        "question_shuffled": _prediction_summary(answers, question_shuffled_predictions),
        "prediction_change_fraction": {
            "image_shuffled": sum(a != b for a, b in zip(
                original_predictions, image_shuffled_predictions)) / count,
            "question_shuffled": sum(a != b for a, b in zip(
                original_predictions, question_shuffled_predictions)) / count,
        },
    }
    return result


def _transport_report(model, dataset, samples: int, batch_size: int,
                      device: torch.device) -> dict:
    """Summarize how much mass OT assigns to local matches and dustbins."""
    count = min(samples, len(dataset))
    totals = {"matched_mass": 0.0, "image_to_dustbin_mass": 0.0,
              "dustbin_to_question_mass": 0.0, "dustbin_to_dustbin_mass": 0.0}
    max_row_residual = max_column_residual = 0.0
    with torch.no_grad():
        for start in range(0, count, batch_size):
            records = [dataset[index] for index in range(start, min(start + batch_size, count))]
            images = torch.stack([row[1] for row in records]).to(device)
            questions = [row[2] for row in records]
            _, _, diagnostics = model.encode(images, questions, return_transport=True)
            for key in totals:
                totals[key] += diagnostics[key].sum().item()
            max_row_residual = max(max_row_residual,
                                   diagnostics["row_residual"].max().item())
            max_column_residual = max(max_column_residual,
                                      diagnostics["column_residual"].max().item())
    return {**{key: value / count for key, value in totals.items()},
            "max_row_residual": max_row_residual,
            "max_column_residual": max_column_residual}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dev_csv_path", required=True)
    parser.add_argument("--dev_img_path", required=True)
    parser.add_argument("--train_csv_path", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=1105)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.samples < 2 or args.batch_size < 1:
        raise ValueError("samples must be at least 2 and batch_size must be positive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint = Path(args.checkpoint)
    metrics = Path(args.metrics) if args.metrics else checkpoint.parent / "metrics.jsonl"
    model = load_model(checkpoint, device)
    frame = load_dataframe(args.dev_csv_path)
    dataset = VQADataset(frame, Config.transforms, args.dev_img_path)
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    report = {
        "checkpoint": str(checkpoint),
        "device": str(device),
        "history": _history_report(_latest_run(metrics)),
        "parameters": {
            "trainable": trainable,
            "total": sum(parameter.numel() for parameter in model.parameters()),
        },
        "reliance": _reliance_report(
            model, dataset, args.samples, args.batch_size, device
        ),
    }
    if model.fusion == "ot":
        report["transport"] = _transport_report(
            model, dataset, args.samples, args.batch_size, device
        )
    if args.train_csv_path:
        train_frame = load_dataframe(args.train_csv_path)
        report["dataset"] = _dataset_report(train_frame, frame)
    output = Path(args.output) if args.output else checkpoint.parent / "bottleneck_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
