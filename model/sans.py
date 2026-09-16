import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.data_processing import preprocess_data
from utils.ViTextVQA_dataset import ViTextVQA_Dataset
from model.features_extraction import ImageEmbedding, QuesEmbedding
from configs.arg_parser import get_args
from configs.config import Config

args = get_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class StackAttention(nn.Module):
    def __init__(self, d=768, k=512, dropout=True):
        super(StackAttention, self).__init__()
        self.ff_image = nn.Linear(d, k)
        self.ff_ques = nn.Linear(d, k)
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Linear(k, 1)

    def forward(self, vi, vq):
        hi = self.ff_image(vi)
        hq = self.ff_ques(vq)
        ha = F.tanh(hi + hq)
        if getattr(self, 'dropout'):
            ha = self.dropout(ha)
        ha = self.ff_attention(ha).squeeze(dim=2)
        pi = F.softmax(ha, dim=1)
        vi_attended = (pi.unsqueeze(dim=2) * vi).sum(dim=1)
        u = vi_attended + vq.squeeze(1)
        return u
    
if __name__=="__main__":
    df_train, _, _ = preprocess_data(args)
    train_vlsp_dataset = ViTextVQA_Dataset(df_train, transform=Config.transforms)
    train_loader = DataLoader(train_vlsp_dataset, batch_size=args.batch_size, shuffle=True)

    image_model = ImageEmbedding().to(device)
    ques_model = QuesEmbedding(output_size=768).to(device)

    for batch in train_loader:
        anno_ids, images, questions, answers = batch
        if torch.cuda.is_available():
            images = images.cuda()
            questions = questions
            anno_ids = anno_ids
        
        with torch.no_grad():
            image_embeddings, att_ids = image_model(images, image_ids=anno_ids)
            ques_embeddings = ques_model(questions)
        break    

    image_embeddings = image_embeddings.reshape(args.batch_size, 768, -1).permute(0, 2, 1)
    ques_embeddings = ques_embeddings.unsqueeze(1)
    san_model = StackAttention(d=768, k=512, dropout=True).to(device)

    img_text_att = san_model(image_embeddings.to(device), ques_embeddings.to(device))
    print("Image Text Attention: ", img_text_att.size())