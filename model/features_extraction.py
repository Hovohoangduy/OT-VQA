from pathlib import Path

import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from configs.config import Config


def validate_english_text_model(model_name):
    name = str(model_name).lower().replace("_", "-")
    if any(marker in name for marker in ("phobert", "vietnam", "vinai/")):
        raise ValueError("Vietnamese text encoders are not supported; use an English encoder")


class ImageEmbedding(nn.Module):
    def __init__(self, model_name=Config.image_model, skip_model=False, model=None, process=None):
        super().__init__()
        self.skip_model = skip_model
        if skip_model and model is None:
            self.process = process
            self.model = None
            self.num_prefix_tokens = 1
            try:
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(model_name)
                self.hidden_size = getattr(cfg, "hidden_size", 768)
                model_type = str(getattr(cfg, "model_type", "")).lower()
                self.num_prefix_tokens = 2 if model_type == "deit" else 1
                self.patch_grid_size = self._grid_size_from_config(cfg)
            except Exception:
                self.hidden_size = 768
                self.patch_grid_size = None
        else:
            self.process = process or AutoImageProcessor.from_pretrained(model_name)
            self.model = model or AutoModel.from_pretrained(model_name)
            self.hidden_size = self.model.config.hidden_size
            model_type = str(getattr(self.model.config, "model_type", "")).lower()
            if model_type not in {"vit", "deit"}:
                raise ValueError(
                    "The visual encoder must be a ViT or DeiT token encoder; "
                    f"received model_type={model_type!r}"
                )
            # ViT has one CLS token. Distilled DeiT has CLS and distillation tokens.
            self.num_prefix_tokens = 2 if model_type == "deit" else 1
            self.patch_grid_size = self._grid_size_from_config(self.model.config)
            self.model.requires_grad_(False)
            self.model.eval()

    @staticmethod
    def _grid_size_from_config(config):
        image_size = getattr(config, "image_size", None)
        patch_size = getattr(config, "patch_size", None)
        if image_size is None or patch_size is None:
            return None
        image = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        patch = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
        if len(image) != 2 or len(patch) != 2 or min(*image, *patch) <= 0:
            return None
        if image[0] % patch[0] or image[1] % patch[1]:
            return None
        return (int(image[0] // patch[0]), int(image[1] // patch[1]))

    def train(self, mode=True):
        super().train(mode)
        if self.model is not None:
            self.model.eval()
        return self

    def forward(self, image, image_ids=None):
        if self.model is None or self.process is None:
            raise RuntimeError("Raw image forward pass requires an initialized visual model")
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
    def __init__(
        self, model_name=Config.text_model,
        text_encoder=None, tokenizer=None, skip_model=False,
    ):
        super().__init__()
        validate_english_text_model(model_name)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.skip_model = skip_model
        if skip_model and text_encoder is None:
            self.text_encoder = None
            try:
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(model_name)
                self.hidden_size = getattr(cfg, "hidden_size", 768)
            except Exception:
                self.hidden_size = 768
        else:
            self.text_encoder = text_encoder or AutoModel.from_pretrained(model_name)
            self.hidden_size = self.text_encoder.config.hidden_size
        self.encoder_frozen = False

    def freeze_encoder(self):
        self.encoder_frozen = True
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen and self.text_encoder is not None:
            self.text_encoder.eval()
        return self

    def encode_tokens(self, questions):
        if self.text_encoder is None:
            raise RuntimeError("Raw question encoding requires text_encoder; use precomputed features")
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
        embeddings, padding_mask, _ = self.encode_tokens(questions)
        return embeddings, padding_mask


class AnswerEmbedding(nn.Module):
    def __init__(
        self, input_size=768, model_name=Config.text_model,
        text_encoder=None, tokenizer=None, token_embeddings=None,
        embeddings_path=None,
    ):
        super().__init__()
        validate_english_text_model(model_name)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        if token_embeddings is not None:
            self.token_embeddings = token_embeddings
        elif text_encoder is not None and hasattr(text_encoder, "embeddings"):
            self.token_embeddings = text_encoder.embeddings
        elif embeddings_path is not None and Path(embeddings_path).is_file():
            from transformers import AutoConfig
            from transformers.models.bert.modeling_bert import BertEmbeddings
            config = AutoConfig.from_pretrained(model_name)
            self.token_embeddings = BertEmbeddings(config)
            state = torch.load(embeddings_path, map_location="cpu", weights_only=True)
            self.token_embeddings.load_state_dict(state)
        else:
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
