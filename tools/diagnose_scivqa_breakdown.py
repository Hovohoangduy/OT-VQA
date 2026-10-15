"""Detailed SciVQA Diagnostic Breakdown by Question Category and Failure Mode Analysis.

Evaluates a checkpoint on SciVQA validation data and breaks down performance across
the 7 qa_pair_type categories, specifically isolating the infinite numerical answer bottleneck.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from utils.checkpoint import load_model
from utils.data_processing import load_dataframe
from utils.device import resolve_device, seed_everything
from utils.metrics import (build_bertscore_scorer, mean_scores, metrics_for_dataset,
                           scivqa_score_pairs)
from utils.vqa_dataset import VQADataset, resolve_image_root


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose SciVQA performance by category")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--dev_csv_path", default="data/scivqa_dataset/val.csv",
                        help="Path to SciVQA val.csv")
    parser.add_argument("--img_path", default="data/scivqa_dataset/images",
                        help="Path to images directory")
    parser.add_argument("--device", default="auto", help="Execution device (auto, cuda, mps, cpu)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--samples", type=int, default=-1,
                        help="Number of samples to evaluate (-1 for all)")
    parser.add_argument("--seed", type=int, default=1105, help="Random seed for sampling")
    parser.add_argument("--compute_bertscore", action="store_true",
                        help="Compute BERTScore (slower)")
    parser.add_argument("--output_json", default=None,
                        help="Optional destination path for JSON report")
    return parser.parse_args()


def run_breakdown_diagnosis(args) -> dict[str, Any]:
    device = resolve_device(args.device)
    seed_everything(args.seed)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_model(checkpoint_path, device)
    model.eval()

    dev_csv = Path(args.dev_csv_path)
    if not dev_csv.is_file():
        raise FileNotFoundError(f"Validation CSV not found: {dev_csv}")

    frame = load_dataframe(dev_csv)
    total_available = len(frame)
    if 0 < args.samples < total_available:
        frame = frame.sample(n=args.samples, random_state=args.seed).reset_index(drop=True)
        print(f"Sampled {len(frame)} / {total_available} validation examples (seed {args.seed}).")
    else:
        print(f"Evaluating on all {len(frame)} validation examples.")

    resolved_img_root = resolve_image_root(frame, args.img_path, "dev")
    dataset = VQADataset(
        frame,
        transform=model.image_transform,
        img_path=resolved_img_root,
        dataset_name="scivqa",
        figure_context=model.figure_context,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    bert_scorer = None
    if args.compute_bertscore:
        bert_scorer = build_bertscore_scorer(device=str(device))

    references: list[str] = []
    generated: list[str] = []
    annotations: list[Any] = []
    questions: list[str] = []

    print("Running greedy generation...")
    with torch.no_grad():
        for batch in loader:
            batch_anno_ids, batch_images, batch_questions, batch_answers = batch
            batch_images = batch_images.to(device)
            hypotheses = model.generate(batch_images, batch_questions, batch_anno_ids)
            answers = model.answers_from_ids(hypotheses)

            references.extend(batch_answers)
            generated.extend(answers)
            questions.extend(batch_questions)
            if torch.is_tensor(batch_anno_ids):
                annotations.extend(batch_anno_ids.tolist())
            else:
                annotations.extend(batch_anno_ids)

    print("Scoring predictions against references...")
    if bert_scorer is not None:
        rows = scivqa_score_pairs(references, generated, bert_scorer=bert_scorer)
        metric_keys = metrics_for_dataset("scivqa")
    else:
        from rouge_score.rouge_scorer import RougeScorer
        rouge = RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        rows = []
        for ref, hyp in zip(references, generated):
            scores = rouge.score(str(ref), str(hyp))
            rows.append({
                "rouge_1_f1": scores["rouge1"].fmeasure,
                "rouge_1_precision": scores["rouge1"].precision,
                "rouge_1_recall": scores["rouge1"].recall,
                "rouge_l_f1": scores["rougeL"].fmeasure,
                "rouge_l_precision": scores["rougeL"].precision,
                "rouge_l_recall": scores["rougeL"].recall,
            })
        metric_keys = ["rouge_1_f1", "rouge_l_f1"]
    overall_scores = mean_scores(rows, metric_keys)

    # Group by qa_pair_type
    qa_pair_types = frame["qa_pair_type"].tolist() if "qa_pair_type" in frame else ["unknown"] * len(rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for i, (qtype, row, ref, hyp, q) in enumerate(zip(qa_pair_types, rows, references, generated, questions)):
        item = {
            "index": i,
            "qa_pair_type": qtype,
            "question": q,
            "reference": ref,
            "prediction": hyp,
            **row,
        }
        grouped.setdefault(qtype, []).append(item)

    category_summary: dict[str, dict[str, Any]] = {}
    for cat_name, items in sorted(grouped.items()):
        category_summary[cat_name] = {
            "count": len(items),
            "metrics": mean_scores(items, metric_keys),
        }

    # Error analysis on Infinite Answer Set
    infinite_errors = []
    unanswerable_phrase = "not possible to answer"
    for cat_name, items in grouped.items():
        if "infinite" in cat_name.lower():
            for item in items:
                pred = item["prediction"].strip()
                ref = item["reference"].strip()
                # classify error type
                if unanswerable_phrase in pred.lower():
                    error_type = "false_unanswerable_rejection"
                elif pred == "":
                    error_type = "empty_prediction"
                else:
                    error_type = "numerical_or_text_mismatch"
                infinite_errors.append({
                    "qa_pair_type": cat_name,
                    "question": item["question"][:120],
                    "reference": ref,
                    "prediction": pred,
                    "rouge_l_f1": item.get("rouge_l_f1", 0.0),
                    "error_type": error_type,
                })

    # Sort worst first
    infinite_errors.sort(key=lambda x: x["rouge_l_f1"])

    report = {
        "checkpoint": str(checkpoint_path),
        "total_examples": len(rows),
        "overall_metrics": overall_scores,
        "categories": category_summary,
        "infinite_answer_errors_count": len(infinite_errors),
        "sample_infinite_errors": infinite_errors[:15],
    }

    # Print Markdown Summary Table
    print("\n" + "=" * 80)
    print("🎯 SCI-VQA CATEGORY BREAKDOWN REPORT")
    print("=" * 80)
    print(f"{'Category':<50} | {'Count':<6} | {'ROUGE-1':<8} | {'ROUGE-L':<8}")
    print("-" * 80)
    for cat, data in category_summary.items():
        m = data["metrics"]
        r1 = f"{m.get('rouge_1_f1', 0.0):.4f}"
        rl = f"{m.get('rouge_l_f1', 0.0):.4f}"
        print(f"{cat:<50} | {data['count']:<6} | {r1:<8} | {rl:<8}")
    print("-" * 80)
    ov_r1 = f"{overall_scores.get('rouge_1_f1', 0.0):.4f}"
    ov_rl = f"{overall_scores.get('rouge_l_f1', 0.0):.4f}"
    print(f"{'OVERALL':<50} | {len(rows):<6} | {ov_r1:<8} | {ov_rl:<8}")
    print("=" * 80)

    if infinite_errors:
        print("\n🔍 SAMPLE INFINITE ANSWER SET ERRORS (Numerical reading bottleneck):")
        for idx, err in enumerate(infinite_errors[:8], 1):
            print(f"[{idx}] Type: {err['error_type']} | Ground Truth: '{err['reference']}' | Pred: '{err['prediction']}'")
            print(f"    Q: {err['question']}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote full diagnostic report to: {out_path}")

    return report


def main():
    args = parse_args()
    run_breakdown_diagnosis(args)


if __name__ == "__main__":
    main()
