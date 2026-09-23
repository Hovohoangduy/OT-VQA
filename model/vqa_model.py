"""Cross-Attention VQA student with autoregressive answer generation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from configs.config import Config
from model.decoder_model import Decoder
from model.features_extraction import (
    AnswerEmbedding, ImageEmbedding, QuestionEmbedding,
    validate_english_text_model,
)
from model.fusion_methods import (
    CrossAttentionFusion, CrossAttentionFusionConfig, FusionInput, FusionOutput,
)
from model.ot_routing import OTEvidenceRouter, OTEvidenceRoutingConfig


@dataclass
class EncoderOutput:
    memory: torch.Tensor
    memory_padding_mask: torch.Tensor
    fusion_output: object


@dataclass
class GenerationOutput:
    generated_ids: torch.Tensor
    fusion_output: Optional[object] = None


class VQAModel(nn.Module):
    """Frozen ViT/BERT features, selectable evidence integration, and decoder."""

    def __init__(
        self,
        vocab_size=None,
        d_model=768,
        num_heads=4,
        ffn_hidden=2048,
        drop_prob=0.1,
        num_layers=4,
        text_model=Config.text_model,
        image_model=Config.image_model,
        fusion="cross_attention",
        fusion_config=None,
        routing_config=None,
        freeze_answer_embeddings=False,
        skip_encoders=False,
        shared_text_encoder=None,
        shared_tokenizer=None,
        token_embeddings=None,
        embeddings_path=None,
    ):
        super().__init__()
        supported = {
            "cross_attention", "ot_evidence_routing", "softmax_evidence_routing",
        }
        if fusion not in supported:
            raise ValueError(
                f"Unsupported fusion {fusion!r}; choose one of {sorted(supported)}"
            )
        validate_english_text_model(text_model)
        self.fusion_type = fusion
        self.text_model_name = str(text_model)
        self.image_model_name = str(image_model)
        parsed_fusion = (
            CrossAttentionFusionConfig.from_dict(fusion_config)
            if fusion == "cross_attention" else None
        )
        parsed_routing = (
            OTEvidenceRoutingConfig.from_dict(routing_config)
            if fusion != "cross_attention" else None
        )
        self.model_config = {
            "d_model": d_model,
            "num_heads": num_heads,
            "ffn_hidden": ffn_hidden,
            "drop_prob": drop_prob,
            "num_layers": num_layers,
            "fusion": fusion,
            "fusion_config": parsed_fusion.to_dict() if parsed_fusion else None,
            "routing_config": parsed_routing.to_dict() if parsed_routing else None,
            "freeze_answer_embeddings": freeze_answer_embeddings,
        }
        self.skip_encoders = skip_encoders
        if skip_encoders:
            self.image_model = ImageEmbedding(image_model, skip_model=True)
            self.question_encoder = QuestionEmbedding(
                model_name=text_model, tokenizer=shared_tokenizer, skip_model=True,
            )
        else:
            self.image_model = ImageEmbedding(image_model)
            self.question_encoder = QuestionEmbedding(
                model_name=text_model,
                text_encoder=shared_text_encoder,
                tokenizer=shared_tokenizer,
            )
        self.question_encoder.freeze_encoder()
        image_dim = self.image_model.hidden_size
        question_dim = self.question_encoder.hidden_size

        text_encoder_ref = (
            getattr(self.question_encoder, "text_encoder", None) or shared_text_encoder
        )
        self.answer_embedding = AnswerEmbedding(
            model_name=text_model,
            text_encoder=text_encoder_ref,
            tokenizer=self.question_encoder.tokenizer,
            token_embeddings=token_embeddings,
            embeddings_path=embeddings_path,
        )
        if freeze_answer_embeddings:
            self.answer_embedding.freeze()
        else:
            # Keep trainable answer embeddings independent from frozen BERT.
            if (self.question_encoder.text_encoder is not None and
                    self.answer_embedding.token_embeddings is
                    self.question_encoder.text_encoder.embeddings):
                self.answer_embedding.token_embeddings = copy.deepcopy(
                    self.answer_embedding.token_embeddings
                )
            self.answer_embedding.token_embeddings.requires_grad_(True)
        self.tokenizer = self.answer_embedding.tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else self.tokenizer.cls_token_id
        )
        self.eos_token_id = (
            self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id is not None
            else self.tokenizer.sep_token_id
        )
        if any(
            value is None
            for value in (self.pad_token_id, self.bos_token_id, self.eos_token_id)
        ):
            raise ValueError("Tokenizer needs PAD, BOS/CLS, and EOS/SEP tokens")

        answer_dim = self.answer_embedding.token_embeddings.word_embeddings.embedding_dim
        self.answer_projection = (
            nn.Identity() if answer_dim == d_model else nn.Linear(answer_dim, d_model)
        )
        if fusion == "cross_attention":
            self.fusion_module = CrossAttentionFusion(
                image_dim, question_dim, d_model, parsed_fusion
            )
        else:
            self.fusion_module = OTEvidenceRouter(
                image_dim,
                question_dim,
                d_model,
                parsed_routing,
                routing_mode="ot" if fusion == "ot_evidence_routing" else "softmax",
            )
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
        actual_vocab = self.answer_embedding.token_embeddings.word_embeddings.num_embeddings
        if vocab_size is not None and vocab_size != actual_vocab:
            raise ValueError("vocab_size must match the text-model embedding vocabulary")
        self.mlp = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, actual_vocab),
        )

    def _fusion_input(
        self, visual_tokens, question_tokens, visual_padding_mask,
        question_padding_mask,
    ) -> FusionInput:
        reference = next(self.fusion_module.parameters())
        return FusionInput(
            visual_tokens=visual_tokens.to(reference.device, reference.dtype),
            question_tokens=question_tokens.to(reference.device, reference.dtype),
            visual_padding_mask=visual_padding_mask.to(reference.device, torch.bool),
            question_padding_mask=question_padding_mask.to(reference.device, torch.bool),
        )

    def _fuse(
        self, visual_tokens, question_tokens, visual_padding_mask,
        question_padding_mask, return_diagnostics=False,
    ) -> EncoderOutput:
        inputs = self._fusion_input(
            visual_tokens, question_tokens, visual_padding_mask,
            question_padding_mask,
        )
        if self.fusion_type == "cross_attention":
            output = self.fusion_module(
                inputs, return_diagnostics=return_diagnostics,
            )
        else:
            output = self.fusion_module(
                inputs.visual_tokens,
                inputs.question_tokens,
                inputs.visual_padding_mask,
                inputs.question_padding_mask,
                grid_size=self.image_model.patch_grid_size,
                return_diagnostics=return_diagnostics,
            )
        if return_diagnostics:
            valid = ~output.memory_padding_mask
            normalizer = valid.sum(1).clamp_min(1)
            diagnostics = dict(output.diagnostics or {})
            diagnostics.update({
                "memory_length": valid.sum(1).float(),
                "memory_norm": (
                    output.memory.masked_fill(~valid.unsqueeze(-1), 0.0)
                    .float().norm(dim=-1).sum(1) / normalizer
                ),
            })
            output.diagnostics = diagnostics
        return EncoderOutput(output.memory, output.memory_padding_mask, output)

    def encode_from_features(
        self, image_embeddings, question_embeddings, question_padding_mask,
        return_diagnostics=False,
    ) -> EncoderOutput:
        reference = next(self.fusion_module.parameters())
        image_embeddings = image_embeddings.to(reference.device, reference.dtype)
        visual_tokens = self.image_model.spatial_tokens(image_embeddings)
        visual_mask = torch.zeros(
            visual_tokens.shape[:2], dtype=torch.bool, device=visual_tokens.device
        )
        return self._fuse(
            visual_tokens, question_embeddings, visual_mask,
            question_padding_mask, return_diagnostics,
        )

    def encode(self, images, questions, anno_ids=None, return_diagnostics=False):
        image_embeddings, _ = self.image_model(images, image_ids=anno_ids)
        question_embeddings, question_mask, _ = self.question_encoder.encode_tokens(
            questions
        )
        return self.encode_from_features(
            image_embeddings, question_embeddings, question_mask, return_diagnostics
        )

    def decode(self, input_ids, memory, causal=True, memory_padding_mask=None):
        target = self.answer_projection(self.answer_embedding.embed_ids(input_ids))
        target_length = input_ids.size(1)
        blocked = input_ids.eq(self.pad_token_id)[:, None, None, :].expand(
            -1, 1, target_length, -1
        )
        if causal:
            blocked = blocked | torch.triu(
                torch.ones(
                    target_length, target_length, dtype=torch.bool,
                    device=input_ids.device,
                ),
                diagonal=1,
            )
        cross_mask = None
        if memory_padding_mask is not None:
            if memory_padding_mask.shape != memory.shape[:2]:
                raise ValueError("memory_padding_mask must match memory")
            cross_mask = memory_padding_mask[:, None, None, :].expand(
                -1, 1, target_length, -1
            )
        return self.mlp(self.decoder(memory, target, blocked, cross_mask))

    def forward(
        self, images=None, questions=None, answers=None, anno_ids=None, mask=True,
        mode="train", max_len=Config.MAX_LEN_ANS, return_diagnostics=False,
        image_features=None, question_features=None, question_padding_mask=None,
        visual_features=None, visual_padding_mask=None,
        return_attention_weights=False,
    ):
        diagnostics_requested = return_diagnostics or return_attention_weights
        if visual_features is not None:
            if question_features is None or question_padding_mask is None:
                raise ValueError("Spatial visual features require question features and mask")
            if visual_padding_mask is None:
                visual_padding_mask = torch.zeros(
                    visual_features.shape[:2], dtype=torch.bool,
                    device=visual_features.device,
                )
            encoded = self._fuse(
                visual_features, question_features, visual_padding_mask,
                question_padding_mask, diagnostics_requested,
            )
        elif image_features is not None:
            if question_features is None or question_padding_mask is None:
                raise ValueError("Cached image features require question features and mask")
            encoded = self.encode_from_features(
                image_features, question_features, question_padding_mask,
                diagnostics_requested,
            )
        else:
            if mode not in {"train", "eval", "test", "infer", "generate"}:
                raise ValueError(f"Unknown mode: {mode}")
            if mode in {"test", "infer", "generate"} or answers is None:
                return self.generate(
                    images, questions, anno_ids, max_len,
                    return_diagnostics=return_diagnostics,
                )
            encoded = self.encode(
                images, questions, anno_ids, diagnostics_requested
            )

        if answers is None:
            return self._generate_from_memory(
                encoded.memory, encoded.memory_padding_mask, max_len
            )
        ids = self.answer_embedding.tokenize(answers, max_len)
        logits = self.decode(
            ids[:, :-1], encoded.memory, causal=mask,
            memory_padding_mask=encoded.memory_padding_mask,
        )
        if return_attention_weights:
            if encoded.fusion_output.attention_weights is None:
                raise RuntimeError("Requested routing/attention weights were not produced")
            return logits, ids[:, 1:], encoded.fusion_output.attention_weights
        if return_diagnostics:
            return logits, ids[:, 1:], encoded.fusion_output
        return logits, ids[:, 1:]

    def forward_from_features(
        self, image_embeddings, question_embeddings, question_padding_mask,
        answers, max_len=Config.MAX_LEN_ANS, mask=True, return_diagnostics=False,
    ):
        return self.forward(
            image_features=image_embeddings,
            question_features=question_embeddings,
            question_padding_mask=question_padding_mask,
            answers=answers,
            max_len=max_len,
            mask=mask,
            return_diagnostics=return_diagnostics,
        )

    @torch.no_grad()
    def generate(
        self, images, questions, anno_ids=None, max_len=Config.MAX_LEN_ANS,
        return_diagnostics=False,
    ):
        if not 2 <= max_len <= Config.MAX_LEN_ANS:
            raise ValueError(f"max_len must be between 2 and {Config.MAX_LEN_ANS}")
        was_training = self.training
        self.eval()
        try:
            encoded = self.encode(images, questions, anno_ids, return_diagnostics)
            generated = self._generate_from_memory(
                encoded.memory, encoded.memory_padding_mask, max_len
            )
            if return_diagnostics:
                return GenerationOutput(generated, encoded.fusion_output)
            return generated
        finally:
            self.train(was_training)

    def _generate_from_memory(self, memory, memory_padding_mask, max_len):
        ids = torch.full(
            (memory.size(0), 1), self.bos_token_id, dtype=torch.long,
            device=memory.device,
        )
        finished = torch.zeros(memory.size(0), dtype=torch.bool, device=memory.device)
        for _ in range(max_len - 1):
            logits = self.decode(
                ids, memory, memory_padding_mask=memory_padding_mask
            )[:, -1]
            logits[:, [self.pad_token_id, self.bos_token_id]] = float("-inf")
            next_ids = logits.argmax(-1).masked_fill(finished, self.pad_token_id)
            ids = torch.cat([ids, next_ids.unsqueeze(1)], dim=1)
            finished |= next_ids.eq(self.eos_token_id)
            if bool(finished.all()):
                break
        return ids[:, 1:]

    @torch.no_grad()
    def generate_from_features(
        self, image_embeddings, question_embeddings, question_padding_mask,
        max_len=Config.MAX_LEN_ANS, return_diagnostics=False,
    ):
        if not 2 <= max_len <= Config.MAX_LEN_ANS:
            raise ValueError(f"max_len must be between 2 and {Config.MAX_LEN_ANS}")
        was_training = self.training
        self.eval()
        try:
            encoded = self.encode_from_features(
                image_embeddings, question_embeddings, question_padding_mask,
                return_diagnostics,
            )
            generated = self._generate_from_memory(
                encoded.memory, encoded.memory_padding_mask, max_len
            )
            if return_diagnostics:
                return GenerationOutput(generated, encoded.fusion_output)
            return generated
        finally:
            self.train(was_training)

    def answers_from_ids(self, ids):
        if isinstance(ids, GenerationOutput):
            ids = ids.generated_ids
        rows = []
        for row in ids.detach().cpu().tolist():
            if self.eos_token_id in row:
                row = row[:row.index(self.eos_token_id)]
            rows.append(row)
        return self.tokenizer.batch_decode(rows, skip_special_tokens=True)
