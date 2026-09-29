import torch
from torch import nn
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
        self.has_cls_token = self.model.config.model_type == 'vit'
        if self.model.config.model_type == 'mobilevitv2':
            self.output_dim = self.model.encoder.layer[-1].conv_projection.convolution.out_channels
        else:
            self.output_dim = self.model.config.hidden_size
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, image, image_ids=None):
        device = next(self.model.parameters()).device
        expected_size = self.model.config.image_size
        if isinstance(expected_size, int):
            expected_size = (expected_size, expected_size)
        # The dataset already resizes, crops and converts RGB images to float [0, 1].
        # Apply the remaining processor operations on device for standard batches.
        if (image.ndim == 4 and image.shape[1] == 3 and image.shape[-2:] == tuple(expected_size)
                and image.is_floating_point()):
            image = image.to(device)
            if getattr(self.process, 'do_flip_channel_order', False):
                image = image.flip(1)
            if getattr(self.process, 'do_normalize', False):
                mean = image.new_tensor(self.process.image_mean).view(1, 3, 1, 1)
                std = image.new_tensor(self.process.image_std).view(1, 3, 1, 1)
                image = (image - mean) / std
            pixel_values = image
        else:
            # Preserve the processor path for unusual input sizes or formats.
            inputs = self.process(images=image.detach().cpu(), do_rescale=False,
                                  return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
        features = outputs.last_hidden_state
        if features.ndim == 4:
            features = features.flatten(2).transpose(1, 2)
        return features, image_ids

class QuestionEmbedding(nn.Module):
    def __init__(self, model_name=Config.text_model, max_length=Config.MAX_LEN_QUES):
        super().__init__()
        validate_english_text_model(model_name)
        if max_length < 2:
            raise ValueError('Question length must allow CLS and SEP tokens')
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.use_mean_pooling = str(model_name).startswith('sentence-transformers/')

    def encode_tokens(self, questions):
        tokens = self.tokenizer(
            list(questions), return_tensors='pt', padding=True,
            max_length=self.max_length, truncation=True,
            return_special_tokens_mask=True,
        )
        special_mask = tokens.pop('special_tokens_mask').bool()
        tokens = tokens.to(next(self.text_encoder.parameters()).device)
        special_mask = special_mask.to(tokens['input_ids'].device)
        embeddings = self.text_encoder(**tokens).last_hidden_state
        padding_mask = tokens['attention_mask'].eq(0) | special_mask
        if self.use_mean_pooling:
            valid = tokens['attention_mask'].unsqueeze(-1)
            question_global = (embeddings * valid).sum(1) / valid.sum(1).clamp_min(1)
            question_global = torch.nn.functional.normalize(question_global, p=2, dim=-1)
        else:
            question_global = embeddings[:, 0]
        return embeddings, padding_mask, tokens['input_ids'], question_global

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
