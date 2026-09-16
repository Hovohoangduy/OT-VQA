import torch
import torch.nn as nn
import torch.nn.functional as F
from model.features_extraction import ImageEmbedding, QuesEmbedding, AnsEmbedding
from model.sans import StackAttention
from model.decoder_model import Decoder
from configs.config import Config
from configs.arg_parser import get_args

args = get_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class VQAModel(nn.Module):

    def __init__(self, vocab_size=64001, output_size=768, d_model=768, 
                 num_heads=4, ffn_hidden=2048, drop_prob=0.1, num_layers=4, 
                 num_att_layers=2, mode='train'):
        super(VQAModel, self).__init__()
        self.mode = mode
        self.image_model = ImageEmbedding().to(device)
        self.ques_model = QuesEmbedding(output_size=output_size).to(device)
        self.ans_model = AnsEmbedding().to(device)
        
        self.san_model = nn.ModuleList(
            [StackAttention(d=d_model, k=512, dropout=True)] * num_att_layers).to(device)
        
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, 
                               drop_prob, num_layers).to(device)

        self.mlp = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, vocab_size))

    def forward(self, images, questions, answers, anno_ids, mask, 
                mode, max_len=Config.MAX_LEN_ANS):
        image_embeddings, att_ids = self.image_model(images.to(device), image_ids=anno_ids)
        if mode == 'train':
            image_embedds = image_embeddings.reshape(args.batch_size, 768, -1).permute(0, 2, 1)
        else:
            image_embedds = image_embeddings.reshape(args.batch_size, 768, -1).permute(0, 2, 1)
        
        ques_embeddings = self.ques_model(questions)
        ques_embedds = ques_embeddings.unsqueeze(1)
        
        for att_layer in self.san_model:
            att_embedds = att_layer(image_embedds.to(device), ques_embedds.to(device))
        
        ans_vocab, ans_embedds = self.ans_model(answers)
        
        x = ans_embedds # 16 * 48 * 768
        y = att_embedds.to(device).unsqueeze(1).expand(-1, max_len, -1).to(device) # 16 * 768 -> 16 * 48 * 768
        if mask == False:
            out = self.decoder(x, y, mask=None).to(device)
        else:
            mask = torch.full([max_len, max_len] , float('-inf'))
            mask = torch.triu(mask, diagonal=1).to(device)
        
            out = self.decoder(x, y, mask).to(device)

        output_logits = self.mlp(out)
        return output_logits, ans_vocab