import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, DeiTModel
from configs.config import Config
from configs.arg_parser import get_args
from utils.data_processing import preprocess_data
from utils.ViTextVQA_dataset import ViTextVQA_Dataset

args = get_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

### Image features extraction model
class ImageEmbedding(nn.Module):
    def __init__(self):
        super(ImageEmbedding, self).__init__()
        self.process = AutoImageProcessor.from_pretrained(Config.image_model)
        self.model = DeiTModel.from_pretrained(Config.image_model)
        #self.model = nn.Sequential(*list(self.model.children())[:3])
        
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image, image_ids):
        inputs = self.process(image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs.to(device))
            
        image_embedding = outputs.last_hidden_state
        return image_embedding, image_ids
    
### Quesion embedding model
class QuesEmbedding(nn.Module):
    def __init__(self, input_size=768, output_size=768):
        super(QuesEmbedding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(Config.textmodel_dir)
        self.phobert = AutoModel.from_pretrained(Config.textmodel_dir)
        self.lstm = nn.LSTM(input_size, output_size, batch_first=True)

    def forward(self, ques):
        tokenized_input = self.tokenizer(ques, return_tensors='pt', padding='max_length', max_length=Config.MAX_LEN_QUES, truncation=True)
        ques = self.phobert(**tokenized_input.to(device)).last_hidden_state
        _, (h, _) = self.lstm(ques)
        return h.squeeze(0)
    
### Answer embedding model
class AnsEmbedding(nn.Module):
    def __init__(self, input_size=768):
        super(AnsEmbedding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(Config.textmodel_dir)
        self.phobert_embed = AutoModel.from_pretrained(Config.textmodel_dir).embeddings.to(device)

    def forward(self, ans):   
        tokenized_input = self.tokenizer(ans, return_tensors='pt', padding='max_length', max_length=Config.MAX_LEN_ANS, truncation=True, return_attention_mask=False)
        ans = self.phobert_embed(**tokenized_input.to(device))
        return tokenized_input['input_ids'], ans
    

if __name__=="__main__":
    df_train, _, _ = preprocess_data(args)
    train_vlsp_dataset = ViTextVQA_Dataset(df_train, transform=Config.transforms)
    train_loader = DataLoader(train_vlsp_dataset, batch_size=args.batch_size, shuffle=True)

    image_model = ImageEmbedding().to(device)
    ques_model = QuesEmbedding(output_size=768).to(device)
    ans_model = AnsEmbedding()

    for batch in train_loader:
        anno_ids, images, questions, answers = batch
        if torch.cuda.is_available():
            images = images.cuda()
            questions = questions
            anno_ids = anno_ids
            answers = answers
        
        with torch.no_grad():
            image_embeddings, att_ids = image_model(images, image_ids=anno_ids)
            ques_embeddings = ques_model(questions)
            ans_vocab, ans_embedds = ans_model(answers)
        break    

    image_embeddings = image_embeddings.reshape(args.batch_size, 768, -1).permute(0, 2, 1)
    ques_embeddings = ques_embeddings.unsqueeze(1)
    print("image embedding size: ", image_embeddings.size())
    print("question embedding size ", ques_embeddings.size())
    print("answer vocab size: ", ans_vocab.size())
    print("answer embedding size: ", ans_embedds.size())