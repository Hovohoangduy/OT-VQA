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
from utils.checkpoint import load_model
from utils.data_processing import load_dataframe
from utils.device import resolve_device
from utils.metrics import (bertscore_defaults, build_bertscore_scorer, gqa_score_pairs, mean_scores,
                           metrics_for_dataset, resolve_dataset, score_pairs,
                           scivqa_score_pairs)
from utils.vqa_dataset import VQADataset, resolve_image_root, native_collate
from utils.answer_text import ANSWER_POSTPROCESSING


def evaluation(model, test_loader, criterion, vocab_swap=None, device=None,
               measure_performance=False, predictions=None, bert_scorer=None,
               dataset=None, compute_bertscore=True):
    dataset_name = resolve_dataset(
        dataset if dataset is not None else
        getattr(test_loader.dataset, "dataset_name", "plantexpert")
    )
    metric_names = metrics_for_dataset(dataset_name)
    if not compute_bertscore:
        if dataset_name not in {"scivqa", "gqa"}:
            raise ValueError("ROUGE-only mode is supported for SciVQA")
        metric_names = tuple(n for n in metric_names if not n.startswith("bertscore"))
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
            if torch.is_tensor(images):
                images = images.to(device)
            logits, targets, ids = model.evaluate_batch(images, questions, answers, anno_ids)
            loss = criterion(logits.transpose(1, 2), targets)
            count_tokens = targets.ne(model.target_ignore_index).sum().item()
            total_loss += loss.item() * count_tokens
            tokens += count_tokens
            hypotheses = model.answers_from_ids(ids)
            references.extend(answers)
            generated.extend(hypotheses)
            if predictions is not None:
                annotation_values = (anno_ids.tolist() if torch.is_tensor(anno_ids)
                                     else anno_ids)
                annotations.extend(annotation_values)
                question_texts.extend(q["prompt"] if isinstance(q, dict) else q for q in questions)
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
        if compute_bertscore and bert_scorer is None:
            encoder, rescale = bertscore_defaults(dataset_name)
            bert_scorer = build_bertscore_scorer(
                model_type=encoder, rescale_with_baseline=rescale)
        scorer = scivqa_score_pairs if dataset_name == "scivqa" else score_pairs
        rows = (scorer(references, generated, bert_scorer, compute_bertscore=compute_bertscore)
                if dataset_name == "scivqa" else scorer(references, generated, bert_scorer))
    scores = mean_scores(rows, metric_names)
    if predictions is not None:
        for annotation, question, reference, hypothesis, row in zip(
                annotations, question_texts, references, generated, rows):
            predictions.append({
                "anno_id": str(annotation), "question": question,
                "reference": reference, "prediction": hypothesis, **row,
            })
    from utils.quality_evidence import guardrail_metrics
    result = {"guardrails": guardrail_metrics(references, generated), "loss": total_loss / max(tokens, 1), "metrics": scores,
              "examples": examples}
    frame = getattr(test_loader.dataset, "data", None)
    if frame is not None and "qa_pair_type" in frame and len(frame) == len(rows):
        groups = {}
        for kind, row in zip(frame["qa_pair_type"], rows):
            groups.setdefault(str(kind), []).append(row)
        result["metrics_by_qa_pair_type"] = {
            kind: {"examples": len(group), "metrics": mean_scores(group, metric_names)}
            for kind, group in groups.items()
        }
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


def _sha256(path):
    from train import _csv_digest
    return _csv_digest(path)


def main():
    args = get_args(evaluation=True)
    device = resolve_device(args.device)
    print(f"Evaluating on device: {device}")
    default = Path(args.model_path) / "best.pt"
    checkpoint = Path(args.checkpoint) if args.checkpoint else default
    if args.pretrained_eval:
        if args.model_family != "qwen3_vl" or args.checkpoint:
            raise ValueError("--pretrained_eval requires --model_family qwen3_vl without --checkpoint")
        from model.factory import build_native
        model = build_native(args, device)
    else:
        model = load_model(checkpoint, device)
    native = getattr(model, "model_family", "legacy") == "qwen3_vl"
    for name in (() if native else ("image_size", "image_preprocessing", "ot_visual_memory", "figure_context")):
        requested = getattr(args, name)
        stored = model.model_config[name]
        if requested is not None and requested != stored:
            raise ValueError(f"--{name} must match the checkpoint ({stored})")
    if args.prompt_version is not None:
        model.prompt_version = args.prompt_version
    if args.decode_version is not None:
        if native:
            raise ValueError("Legacy decode versions do not apply to native generation")
        model.decode_version = args.decode_version
    model.model_config.update(prompt_version=model.prompt_version, decode_version=model.decode_version)
    if native:
        if "--plan_control" in args.explicit_args or args.pretrained_eval:
            model.adapter.plan_control = args.plan_control
    print(f"Fusion method: {model.fusion}")
    csv_path = args.dev_csv_path if args.split == "dev" else args.test_csv_path
    dataset_name = resolve_dataset(args.dataset, csv_path)
    print(f"Dataset: {dataset_name}")
    frame = load_dataframe(csv_path)
    if args.screen_manifest:
        if args.split != "dev":
            raise ValueError("Screening is restricted to dev")
        manifest = json.loads(Path(args.screen_manifest).read_text())
        if manifest["dev_csv_sha256"] != _sha256(csv_path):
            raise ValueError("Screening manifest belongs to a different dev CSV")
        ids = [str(i) for i in manifest["anno_ids"]]
        if len(set(ids)) != len(ids) or not set(ids).issubset(set(frame.anno_id.astype(str))):
            raise ValueError("Invalid screening annotation IDs")
        chosen = frame[frame.anno_id.astype(str).isin(ids)]
        complete = frame[frame.image.isin(chosen.image.unique())]
        if set(complete.anno_id.astype(str)) != set(ids):
            raise ValueError("Screening manifest must contain complete figures")
        frame = chosen.reset_index(drop=True)
    split_image_path = resolve_image_root(
        frame, args.img_path, args.split,
        override=(args.dev_img_path if args.split == "dev" else args.test_img_path),
    )
    dataset = VQADataset(frame, transform=model.image_transform, img_path=split_image_path,
                         dataset_name=dataset_name, figure_context=model.figure_context,
                         prompt_version=model.prompt_version, native=native)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=native_collate if native else None)
    predictions = [] if args.predictions_csv else None
    scorer = build_bertscore_scorer(
        model_type=args.bertscore_model,
        device=str(resolve_device(args.bertscore_device)),
        batch_size=args.bertscore_batch_size,
        rescale_with_baseline=args.bertscore_rescale,
    ) if dataset_name != "gqa" and args.compute_bertscore else None
    result = evaluation(model, loader, nn.CrossEntropyLoss(ignore_index=model.target_ignore_index),
                        device=device, measure_performance=True, predictions=predictions,
                        bert_scorer=scorer, compute_bertscore=args.compute_bertscore)
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
        for column in ("question_type", "qa_pair_type", "figure_type", "image", "figure_id", "paper_id"):
            if column in frame:
                prediction_frame[column] = frame[column].tolist()
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
            "checkpoint_sha256": None if args.pretrained_eval else _sha256(checkpoint),
            "guardrails": result["guardrails"],
            "plan_control": model.adapter.plan_control if native else None,
            "checkpoint": None if args.pretrained_eval else str(checkpoint), "fusion": model.fusion, "split": args.split,
            "dataset": dataset_name,
            "evaluation_csv_sha256": digest.hexdigest(),
            "examples": len(frame), "screen_manifest": args.screen_manifest,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "loss": result["loss"], "generated_metrics": result["metrics"],
            "performance": performance,
            "model_config": model.model_config,
            "answer_postprocessing": ANSWER_POSTPROCESSING,
            "metrics_by_qa_pair_type": result.get("metrics_by_qa_pair_type", {}),
        }
        if scorer is not None:
            report["bertscore"] = {
                "model": args.bertscore_model, "model_hash": scorer.hash,
                "rescale_with_baseline": args.bertscore_rescale,
                "batch_size": args.bertscore_batch_size,
                "device": str(resolve_device(args.bertscore_device)),
            }
        if dataset_name == "scivqa":
            report["rouge"] = {"implementation": "rouge-score", "use_stemmer": True,
                               "aggregation": "mean of per-answer scores"}
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote evaluation report to {output}")


if __name__ == "__main__":
    main()
