import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import torch.optim as optim
from utils.metrics import normalize_text, compute_em_and_f1
from configs.arg_parser import get_args
from configs.config import Config
from utils.data_processing import preprocess_data
from utils.ViTextVQA_dataset import ViTextVQA_Dataset
from model.vqa_model import VQAModel

### Train model

args = get_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(model, train_loader, num_epochs, optimizer, scheduler, criterion, vocab_swap, device):
    print_every = 2000
    
    losses = []
    em_scores = []
    f1_scores = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_em = 0.0
        total_f1 = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            anno_id, images, questions, answers = batch
            if len(images) == args.batch_size:
                predicted_tokens, ans_embedds = model(images.to(device), questions, answers, anno_id, mode='train', mask=True)
                predicted_tokens = predicted_tokens.float()
                ans_embedds = ans_embedds.long()
                
                # Prepare references and hypotheses
                references = [normalize_text(answer).split() for answer in answers]
                hypotheses = []
                for i in range(args.batch_size):
                    sentence_predicted = torch.argmax(predicted_tokens[i], axis=1)
                    predicted_sentence = []
                    for idx in sentence_predicted:
                        if idx == 2:  # End of Sentence Token
                            break
                        word = vocab_swap.get(idx.item(), "")  # Handle out-of-vocabulary gracefully
                        if word in {"<pad>", "<s>", "</s>", ""}:
                            continue
                        predicted_sentence.append(word)
                        
                    predicted_sentence = ' '.join(predicted_sentence).strip()
                    hypotheses.append(predicted_sentence.split())
                
                # Compute EM and F1 scores
                em_score, f1_score = compute_em_and_f1(references, hypotheses)
                total_em += em_score
                total_f1 += f1_score
                
                if (batch_idx + 1) % print_every == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                    print(f"Exact Match (EM): {em_score:.4f}")
                    print(f"F1 Score: {f1_score:.4f}")
                    
                    for i in range(args.batch_size):
                        sentence_predicted = torch.argmax(predicted_tokens[i], axis=1)
                        predicted_sentence = []
                        for idx in sentence_predicted:
                            if idx == 2:
                                break
                            word = vocab_swap.get(idx.item(), "")
                            if word in {"<pad>", "<s>", "</s>", ""}:
                                continue
                            predicted_sentence.append(word)
                            
                        predicted_sentence = ' '.join(predicted_sentence).strip()
                        print(f"Question: {questions[i]}")
                        print(f"Answer: {answers[i]}")
                        print(f"Answer Prediction: {predicted_sentence}")
                    print("\n")
                
                # Compute loss and update model
                loss = criterion(predicted_tokens.permute(0, 2, 1), ans_embedds)
                valid_indicies = torch.where(ans_embedds == 1, False, True)
                loss = loss.sum() / valid_indicies.sum()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                losses.append(loss.item())
        
        avg_em = total_em / len(train_loader)
        avg_f1 = total_f1 / len(train_loader)
        
        em_scores.append(avg_em)
        f1_scores.append(avg_f1)
        
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Average Exact Match (EM): {avg_em:.4f}")
        print(f"Average F1 Score: {avg_f1:.4f}")
        print("\n")
    
    return losses, em_scores, f1_scores

if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained(Config.textmodel_dir)
    vocab = tokenizer.get_vocab()
    vocab_swap = {value: key for key, value in vocab.items()}

    df_train, _, _ = preprocess_data(args)
    train_vlsp_dataset = ViTextVQA_Dataset(df_train, transform=Config.transforms)
    train_loader = DataLoader(train_vlsp_dataset, batch_size=args.batch_size, shuffle=True)

    num_epochs = args.epochs
    model = VQAModel().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=1)
    optimizer = optim.AdamW(model.parameters(), Config.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=int(len(train_loader) * args.epochs)
    )
    losses, em_scores, f1_scores = train(model, train_loader, num_epochs, optimizer, scheduler, criterion, vocab_swap, device)

    plt.figure(figsize=(10, 6))
    plt.plot(em_scores, label='EM', marker='o')
    plt.plot(f1_scores, label='F1_SCORE', marker='o')
    plt.title('Evaluation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('evaluation_metrics_plot.png')

    torch.save(model.state_dict(), args.model_path + "/" + 'vi_text.pt')