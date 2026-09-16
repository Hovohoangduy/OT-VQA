from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from configs.arg_parser import get_args
from configs.config import Config
from utils.data_processing import load_dataframe
from utils.ViTextVQA_dataset import ViTextVQA_Dataset
from utils.metrics import compute_em_and_f1
from utils.checkpoint import load_model


def evaluation(model, test_loader, criterion, vocab_swap=None, device=None):
    model.eval()
    device = device or next(model.parameters()).device
    total_loss = total_em = total_f1 = 0.0
    examples = tokens = 0
    with torch.no_grad():
        for anno_ids, images, questions, answers in test_loader:
            images = images.to(device)
            # Ground-truth answers are used only for loss, never for generation.
            logits, targets = model(images, questions, answers, anno_ids)
            loss = criterion(logits.transpose(1, 2), targets)
            count_tokens = targets.ne(model.pad_token_id).sum().item()
            total_loss += loss.item() * count_tokens
            tokens += count_tokens
            generated = model.generate(images, questions, anno_ids)
            hypotheses = model.answers_from_ids(generated)
            em, f1 = compute_em_and_f1(answers, hypotheses)
            count = len(answers)
            examples += count
            total_em += em * count
            total_f1 += f1 * count
    if not examples:
        raise ValueError("Evaluation dataset is empty")
    return total_loss / tokens, total_em / examples, total_f1 / examples


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, language = load_model(Path(args.model_path) / "vi_text.pt", device)
    csv_path = args.dev_csv_path if args.split == "dev" else args.test_csv_path
    dataframe = load_dataframe(csv_path, language)
    dataset = ViTextVQA_Dataset(dataframe, transform=Config.transforms, img_path=args.img_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    loss, em, f1 = evaluation(model, loader, nn.CrossEntropyLoss(ignore_index=model.pad_token_id), device=device)
    print(f"{args.split} loss: {loss:.4f}, generated EM: {em:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    main()
