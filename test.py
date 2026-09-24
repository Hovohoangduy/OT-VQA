"""Evaluate teacher-forced loss and answer quality from autoregressive generation."""

from pathlib import Path

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


def evaluation(model, test_loader, criterion, vocab_swap=None, device=None):
    model.eval()
    device = device or next(model.parameters()).device
    total_loss = total_em = total_f1 = 0.0
    examples = tokens = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    with torch.no_grad():
        for batch in test_loader:
            anno_ids, images, questions, answers = batch
            images = images.to(device)
            logits, targets = model(images, questions, answers, anno_ids)
            ids = model.generate(images, questions, anno_ids)
            loss = criterion(logits.transpose(1, 2), targets)
            count_tokens = targets.ne(model.pad_token_id).sum().item()
            total_loss += loss.item() * count_tokens
            tokens += count_tokens
            hypotheses = model.answers_from_ids(ids)
            em, f1 = compute_em_and_f1(answers, hypotheses)
            count = len(answers)
            examples += count
            total_em += em * count
            total_f1 += f1 * count
    if not examples:
        raise ValueError("Evaluation dataset is empty")
    scores = (total_loss / max(tokens, 1), total_em / examples, total_f1 / examples)
    return scores


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
    result = evaluation(model, loader, nn.CrossEntropyLoss(ignore_index=model.pad_token_id),
                        device=device)
    loss, em, f1 = result[:3]
    print(f"{args.split} loss: {loss:.4f}, generated EM: {em:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    main()
