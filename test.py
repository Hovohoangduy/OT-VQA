"""Evaluate teacher-forced loss and generated VQA answer quality."""

import json
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
from utils.metrics import compute_em_and_f1
from utils.vqa_dataset import VQADataset, resolve_image_root


def evaluation(model, test_loader, criterion, vocab_swap=None, device=None,
               measure_performance=False, predictions=None):
    model.eval()
    device = device or next(model.parameters()).device
    total_loss = total_em = total_f1 = 0.0
    examples = tokens = 0
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
            em, f1 = compute_em_and_f1(answers, hypotheses)
            if predictions is not None:
                annotation_values = (anno_ids.tolist() if torch.is_tensor(anno_ids)
                                     else anno_ids)
                for annotation, question, reference, hypothesis in zip(
                        annotation_values, questions, answers, hypotheses):
                    row_em, row_f1 = compute_em_and_f1([reference], [hypothesis])
                    predictions.append({
                        "anno_id": str(annotation), "question": question,
                        "reference": reference, "prediction": hypothesis,
                        "em": row_em, "f1": row_f1,
                    })
            count = len(answers)
            examples += count
            total_em += em * count
            total_f1 += f1 * count
    if not examples:
        raise ValueError("Evaluation dataset is empty")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    scores = (total_loss / max(tokens, 1), total_em / examples, total_f1 / examples)
    if not measure_performance:
        return scores
    performance = {
        "examples": examples,
        "elapsed_seconds": elapsed,
        "examples_per_second": examples / elapsed,
        "milliseconds_per_example": 1000 * elapsed / examples,
        "peak_cuda_bytes": (torch.cuda.max_memory_allocated(device)
                            if device.type == "cuda" else None),
    }
    return (*scores, performance)


def main():
    args = get_args()
    device = resolve_device(args.device)
    print(f"Evaluating on device: {device}")
    default = Path(args.model_path) / "best.pt"
    checkpoint = Path(args.checkpoint) if args.checkpoint else default
    model = load_model(checkpoint, device)
    csv_path = args.dev_csv_path if args.split == "dev" else args.test_csv_path
    frame = load_dataframe(csv_path)
    split_image_path = resolve_image_root(
        frame, args.img_path, args.split,
        override=(args.dev_img_path if args.split == "dev" else args.test_img_path),
    )
    dataset = VQADataset(frame, transform=Config.transforms, img_path=split_image_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    predictions = [] if args.predictions_csv else None
    result = evaluation(model, loader, nn.CrossEntropyLoss(ignore_index=model.pad_token_id),
                        device=device, measure_performance=True, predictions=predictions)
    loss, em, f1 = result[:3]
    print(f"{args.split} loss: {loss:.4f}, generated EM: {em:.4f}, F1: {f1:.4f}")
    performance = result[3]
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
        output.write_text(json.dumps({
            "checkpoint": str(checkpoint), "fusion": model.fusion, "split": args.split,
            "loss": loss, "generated_em": em, "generated_f1": f1,
            "performance": performance,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote evaluation report to {output}")


if __name__ == "__main__":
    main()
