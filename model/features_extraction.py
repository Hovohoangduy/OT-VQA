import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from configs.config import Config


def validate_english_text_model(model_name):
    name = str(model_name).lower().replace("_", "-")
    if any(marker in name for marker in ("phobert", "vietnam", "vinai/")):
        raise ValueError("Vietnamese text encoders are not supported; use an English encoder")


class ImageEmbedding(nn.Module):
    def __init__(self, model_name=Config.image_model):
        super().__init__()
        self.process = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        model_type = str(getattr(self.model.config, "model_type", "")).lower()
        if model_type not in {"vit", "deit"}:
            raise ValueError(
                "The visual encoder must be a ViT or DeiT token encoder; "
                f"received model_type={model_type!r}"
            )
        # ViT has one CLS token. Distilled DeiT has CLS and distillation tokens.
        self.num_prefix_tokens = 2 if model_type == "deit" else 1
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

    def spatial_tokens(self, hidden_states):
        """Remove ViT/DeiT prefix tokens, retaining only spatial patch tokens."""
        if hidden_states.size(1) <= self.num_prefix_tokens:
            raise ValueError("Visual encoder output does not contain spatial patch tokens")
        return hidden_states[:, self.num_prefix_tokens:]


class QuestionEmbedding(nn.Module):
    def __init__(self, input_size=None, output_size=768, model_name=Config.text_model):
        super().__init__()
        validate_english_text_model(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.lstm = nn.LSTM(input_size or self.text_encoder.config.hidden_size, output_size, batch_first=True)
        self.encoder_frozen = False

    def freeze_encoder(self):
        self.encoder_frozen = True
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.text_encoder.eval()
        return self

    def encode_tokens(self, questions):
        tokens = self.tokenizer(
            list(questions), return_tensors='pt', padding=True,
            max_length=Config.MAX_LEN_QUES, truncation=True,
            return_special_tokens_mask=True,
        )
        special_mask = tokens.pop('special_tokens_mask').bool()
        tokens = tokens.to(next(self.text_encoder.parameters()).device)
        special_mask = special_mask.to(tokens['input_ids'].device)
        if self.encoder_frozen:
            with torch.no_grad():
                embeddings = self.text_encoder(**tokens).last_hidden_state
        else:
            embeddings = self.text_encoder(**tokens).last_hidden_state
        padding_mask = tokens['attention_mask'].eq(0) | special_mask
        return embeddings, padding_mask, tokens['input_ids']

    def forward(self, questions):
        embeddings, _, input_ids = self.encode_tokens(questions)
        # Preserve the SAN baseline contract: summarize all non-padding tokens,
        # including tokenizer boundary tokens, with the LSTM.
        lengths = input_ids.ne(self.tokenizer.pad_token_id).sum(1).cpu()
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return hidden.squeeze(0)


class AnswerEmbedding(nn.Module):
    def __init__(self, input_size=768, model_name=Config.text_model):
        super().__init__()
        validate_english_text_model(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.token_embeddings = AutoModel.from_pretrained(model_name).embeddings
        self.embedding_frozen = False

    def freeze(self):
        self.embedding_frozen = True
        self.token_embeddings.requires_grad_(False)
        self.token_embeddings.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.embedding_frozen:
            self.token_embeddings.eval()
        return self

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
        return torch.tensor(rows, dtype=torch.long, device=next(self.token_embeddings.parameters()).device)

    def embed_ids(self, input_ids):
        return self.token_embeddings(input_ids=input_ids)

    def forward(self, answers, max_len=Config.MAX_LEN_ANS):
        ids = self.tokenize(answers, max_len)
        return ids, self.embed_ids(ids)
