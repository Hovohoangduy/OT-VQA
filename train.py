from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from configs.arg_parser import get_args
from configs.config import Config
from utils.data_processing import load_dataframe
from utils.ViTextVQA_dataset import ViTextVQA_Dataset
from utils.metrics import compute_em_and_f1
from model.vqa_model import VQAModel


def train(model, train_loader, num_epochs, optimizer, scheduler, criterion, vocab_swap=None, device=None):
    """
    Executes the training loop with Teacher Forcing.

    Training Flow Pipeline:
        1. DataLoader yields -> (Images, Questions, Answers)
        2. Encoder computes -> Context Memory
        3. Target Shifter (inside model.forward):
             - Decoder Input: ids[:, :-1]  (e.g. [BOS, t1, t2, EOS])
             - Ground Truth:  ids[:, 1:]   (e.g. [t1, t2, EOS, PAD])
        4. Decoder predicts -> Next Token Logits
        5. Loss -> CrossEntropy(Logits, Ground Truth, ignore_index=PAD)
        6. Backward Pass -> Optimizer Step -> Scheduler Step
    """
    device = device or next(model.parameters()).device
    losses, em_scores, f1_scores = [], [], []
    if len(train_loader) == 0:
        raise ValueError("Training dataset is empty")
    for epoch in range(num_epochs):
        model.train()
        total_loss = total_em = total_f1 = 0.0
        examples = tokens = 0
        for batch_idx, (anno_ids, images, questions, answers) in enumerate(train_loader):
            logits, targets = model(images.to(device), questions, answers, anno_ids)
            loss = criterion(logits.transpose(1, 2), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            batch_tokens = targets.ne(model.pad_token_id).sum().item()
            total_loss += loss.item() * batch_tokens
            tokens += batch_tokens
            losses.append(loss.item())
            # Teacher-forced metrics diagnose training only; test.py uses generation.
            hypotheses = model.answers_from_ids(logits.detach().argmax(-1))
            em, f1 = compute_em_and_f1(answers, hypotheses)
            count = len(answers)
            total_em += em * count
            total_f1 += f1 * count
            examples += count
            if (batch_idx + 1) % 2000 == 0:
                print(f"Epoch {epoch + 1}, batch {batch_idx + 1}: loss={loss.item():.4f}")
        em_scores.append(total_em / examples)
        f1_scores.append(total_f1 / examples)
        print(f"Epoch {epoch + 1}/{num_epochs}: loss={total_loss / tokens:.4f}, "
              f"teacher-forced EM={em_scores[-1]:.4f}, F1={f1_scores[-1]:.4f}")
    return losses, em_scores, f1_scores


def main():
    args = get_args()
    if args.batch_size < 1 or args.epochs < 1:
        raise ValueError("batch_size and epochs must be positive")
    torch.manual_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataframe = load_dataframe(args.train_csv_path, args.language)
    dataset = ViTextVQA_Dataset(dataframe, transform=Config.transforms, img_path=args.img_path)
    if len(dataset) == 0:
        raise ValueError("Training dataset is empty")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = VQAModel(text_model=args.text_model, image_model=args.image_model).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_token_id)
    optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=Config.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(loader) * args.epochs)
    _, em_scores, f1_scores = train(model, loader, args.epochs, optimizer, scheduler, criterion, device=device)
    destination = Path(args.model_path)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 2, "model_state_dict": model.state_dict(),
                "text_model": args.text_model, "image_model": args.image_model,
                "language": args.language, "model_config": model.model_config}, destination / "vi_text.pt")
    plt.figure(figsize=(10, 6))
    plt.plot(em_scores, label="Teacher-forced EM")
    plt.plot(f1_scores, label="Teacher-forced F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination / "evaluation_metrics_plot.png")
    plt.close()


if __name__ == "__main__":
    main()
