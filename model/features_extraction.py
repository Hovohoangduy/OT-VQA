import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, DeiTModel

from configs.config import Config


class ImageEmbedding(nn.Module):
    def __init__(self, model_name=Config.image_model):
        super().__init__()
        self.process = AutoImageProcessor.from_pretrained(model_name)
        self.model = DeiTModel.from_pretrained(model_name)
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, image, image_ids=None):
        # Dataset tensors are already in [0, 1]. Rescaling again divides by 255.
        inputs = self.process(images=image.detach().cpu(), do_rescale=False, return_tensors="pt")
        device = next(self.model.parameters()).device
        with torch.no_grad():
            outputs = self.model(**inputs.to(device))
        return outputs.last_hidden_state, image_ids


class QuesEmbedding(nn.Module):
    def __init__(self, input_size=None, output_size=768, model_name=Config.textmodel_dir):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.phobert = AutoModel.from_pretrained(model_name)
        self.lstm = nn.LSTM(input_size or self.phobert.config.hidden_size, output_size, batch_first=True)

    def forward(self, questions):
        tokens = self.tokenizer(list(questions), return_tensors='pt', padding=True,
                                max_length=Config.MAX_LEN_QUES, truncation=True)
        tokens = tokens.to(next(self.phobert.parameters()).device)
        embeddings = self.phobert(**tokens).last_hidden_state
        lengths = tokens['attention_mask'].sum(1).cpu()
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return hidden.squeeze(0)


class AnsEmbedding(nn.Module):
    def __init__(self, input_size=768, model_name=Config.textmodel_dir):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.phobert_embed = AutoModel.from_pretrained(model_name).embeddings

    def tokenize(self, answers, max_len=Config.MAX_LEN_ANS):
        if max_len < 2:
            raise ValueError('Answer length must allow BOS and EOS tokens')
        tok = self.tokenizer
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id
        eos = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id
        if bos is None or eos is None or tok.pad_token_id is None:
            raise ValueError('Tokenizer needs BOS/CLS, EOS/SEP and PAD tokens')
        rows = tok(list(answers), add_special_tokens=False, truncation=True,
                   max_length=max_len - 2)['input_ids']
        rows = [[bos] + row + [eos] + [tok.pad_token_id] * (max_len - len(row) - 2) for row in rows]
        return torch.tensor(rows, dtype=torch.long, device=next(self.phobert_embed.parameters()).device)

    def embed_ids(self, input_ids):
        return self.phobert_embed(input_ids=input_ids)

    def forward(self, answers, max_len=Config.MAX_LEN_ANS):
        ids = self.tokenize(answers, max_len)
        return ids, self.embed_ids(ids)
