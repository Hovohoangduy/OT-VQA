import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils.metrics import compute_em_and_f1
from configs.arg_parser import get_args
from configs.config import Config
from utils.data_processing import preprocess_data
from utils.ViTextVQA_dataset import ViTextVQA_Dataset
from model.vqa_model import VQAModel

args = get_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def evaluation(model, test_loader, criterion, vocab_swap, device):
    model.eval()
    total_loss = 0.0
    total_em = 0.0
    total_f1 = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            anno_id, images, questions, answers = batch
            if len(images) == args.batch_size:
                predicted_tokens, ans_embedds = model(images.to(device), questions, answers, anno_id, mode='train', mask=True)
                predicted_tokens = predicted_tokens.float()
                ans_embedds = ans_embedds.long()

                # Prepare references and hypotheses
                references = [answer.split() for answer in answers]
                hypotheses = []
                for i in range(args.batch_size):
                    sentence_predicted = torch.argmax(predicted_tokens[i], axis=1)
                    predicted_sentence = []
                    for idx in sentence_predicted:
                        if idx == 2:  # End of Sentence Token
                            break
                        word = vocab_swap[idx.item()]
                        
                        if word in {"<pad>", "<s>", "</s>", ""}:
                            continue
                        predicted_sentence.append(word)
                        
                
                    predicted_sentence = ' '.join(predicted_sentence).strip()
                    hypotheses.append(predicted_sentence.split())

                em_score, f1_score = compute_em_and_f1(references, hypotheses)
                total_em += em_score
                total_f1 += f1_score

                total_loss += criterion(predicted_tokens.permute(0, 2, 1), ans_embedds).item()

    avg_loss = total_loss / len(test_loader)
    avg_em = total_em / len(test_loader)
    avg_f1 = total_f1 / len(test_loader)

    return avg_loss, avg_em, avg_f1

if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained(Config.textmodel_dir)
    vocab = tokenizer.get_vocab()
    vocab_swap = {value: key for key, value in vocab.items()}

    _, df_dev, _ = preprocess_data(args)
    test_vitextvqa_dataset = ViTextVQA_Dataset(df_dev, transform=Config.transforms)
    test_loader = DataLoader(test_vitextvqa_dataset, batch_size=args.batch_size, shuffle=True)

    model = VQAModel().to(device)
    model.load_state_dict(torch.load(args.model_path + "/" + 'vi_text.pt'))
    criterion = nn.CrossEntropyLoss(ignore_index=1)

    _, test_em, test_f1 = evaluation(model, test_loader, criterion, vocab_swap, device)

    print(f"Test EM: {test_em:.4f}")
    print(f"Test F1_SCORE: {test_f1:.4f}")