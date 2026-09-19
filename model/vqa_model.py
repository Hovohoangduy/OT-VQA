"""Selectable multimodal-fusion VQA models with autoregressive generation."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    FusionInput, FusionOutput, build_fusion_module, config_for_method,
    parse_fusion_spec,
)
from model.optimal_transport import OTConfig, OptimalTransportFusion, TransportOutput
from model.ot_san import OTSAN, OTSANConfig, OTSANOutput
from model.sans import StackAttention


@dataclass
class EncoderOutput:
    memory: torch.Tensor
    memory_padding_mask: torch.Tensor
    transport: Optional[TransportOutput] = None
    ot_san: Optional[OTSANOutput] = None
    fusion_output: Optional[FusionOutput] = None


@dataclass
class GenerationOutput:
    generated_ids: torch.Tensor
    transport: Optional[TransportOutput] = None
    ot_san: Optional[OTSANOutput] = None
    fusion_output: Optional[FusionOutput] = None


class VQAModel(nn.Module):
    """VQA generator with native, balanced-OT, or UOT-augmented fusion."""

    def __init__(
        self, vocab_size=None, output_size=768, d_model=768, num_heads=4,
        ffn_hidden=2048, drop_prob=0.1, num_layers=4, num_att_layers=2,
        mode='train', text_model=Config.text_model, image_model=Config.image_model,
        fusion='san', ot_config=None, ot_san_config=None, fusion_config=None,
        fusion_spec=None,
        freeze_answer_embeddings=False,
    ):
        super().__init__()
        if output_size != d_model or num_att_layers < 1:
            raise ValueError('output_size must equal d_model and at least one attention layer is needed')
        methods = {'san', 'ban', 'mutan', 'cross_attention', 'qformer'}
        valid_fusions = {'san', 'balanced_ot', 'uot'} | methods | {
            f'balanced_ot_{method}' for method in methods
        } | {f'uot_{method}' for method in methods}
        if fusion not in valid_fusions:
            raise ValueError(
                f"Unknown fusion '{fusion}'; expected one of {sorted(valid_fusions)}"
            )
        validate_english_text_model(text_model)
        self.mode = mode
        self.fusion_type = fusion
        self.text_model_name = str(text_model)
        self.image_model_name = str(image_model)
        self.fusion_spec = parse_fusion_spec(fusion)
        normalized_spec = {
            'method': self.fusion_spec.method,
            'transport': self.fusion_spec.transport,
        }
        if fusion_spec is not None and fusion_spec != normalized_spec:
            raise ValueError(
                f"fusion_spec {fusion_spec} does not match fusion name '{fusion}'"
            )
        parsed_ot = ot_config if isinstance(ot_config, OTConfig) else OTConfig.from_dict(ot_config)
        if self.fusion_spec.transport == 'balanced':
            parsed_ot = replace(parsed_ot, transport_type='balanced')
        elif self.fusion_spec.transport == 'unbalanced':
            parsed_ot = replace(parsed_ot, transport_type='unbalanced')
        parsed_ot_san = (
            ot_san_config if isinstance(ot_san_config, OTSANConfig)
            else OTSANConfig.from_dict(ot_san_config)
        )
        uses_ot_san = fusion in {'balanced_ot_san', 'uot_san'}
        parsed_fusion_config = config_for_method(
            self.fusion_spec.method, fusion_config
        )
        self.ot_config = parsed_ot
        self.model_config = dict(
            output_size=output_size, d_model=d_model, num_heads=num_heads,
            ffn_hidden=ffn_hidden, drop_prob=drop_prob, num_layers=num_layers,
            num_att_layers=num_att_layers, fusion=fusion,
            fusion_spec=normalized_spec,
            ot_config=parsed_ot.to_dict() if self.fusion_spec.uses_ot else None,
            ot_san_config=parsed_ot_san.to_dict() if uses_ot_san else None,
            fusion_config=(parsed_fusion_config.to_dict()
                           if parsed_fusion_config is not None else None),
            freeze_answer_embeddings=freeze_answer_embeddings,
        )
        self.image_model = ImageEmbedding(image_model)
        self.question_encoder = QuestionEmbedding(output_size=output_size, model_name=text_model)
        self.answer_embedding = AnswerEmbedding(model_name=text_model)
        if freeze_answer_embeddings:
            self.answer_embedding.freeze()
        self.tokenizer = self.answer_embedding.tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = (self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None
                             else self.tokenizer.cls_token_id)
        self.eos_token_id = (self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None
                             else self.tokenizer.sep_token_id)
        if any(value is None for value in (self.pad_token_id, self.bos_token_id, self.eos_token_id)):
            raise ValueError('Tokenizer needs PAD, BOS/CLS and EOS/SEP tokens')

        image_dim = self.image_model.model.config.hidden_size
        question_dim = self.question_encoder.text_encoder.config.hidden_size
        answer_dim = self.answer_embedding.token_embeddings.word_embeddings.embedding_dim
        # Preserve original SAN names so version-2 checkpoints load strictly.
        self.image_projection = nn.Identity() if image_dim == d_model else nn.Linear(image_dim, d_model)
        self.answer_projection = nn.Identity() if answer_dim == d_model else nn.Linear(answer_dim, d_model)
        self.san_model = nn.ModuleList(
            [StackAttention(d=d_model, k=512) for _ in range(num_att_layers)]
        )
        self.ot_fusion = (
            OptimalTransportFusion(
                visual_dim=image_dim, question_dim=question_dim, model_dim=d_model,
                config=parsed_ot,
            ) if self.fusion_spec.uses_ot else None
        )
        self.ot_san = OTSAN(d_model, parsed_ot_san) if uses_ot_san else None
        self.fusion_module = build_fusion_module(
            self.fusion_spec.method, image_dim, question_dim, d_model,
            parsed_fusion_config, uses_ot=self.fusion_spec.uses_ot,
        )
        if fusion != 'san':
            self.question_encoder.freeze_encoder()
            # These legacy SAN-only parameters stay in the state dict so current
            # version-3 OT checkpoints still load strictly, but token fusion must
            # not count or optimize modules that never participate in its graph.
            self.question_encoder.lstm.requires_grad_(False)
            self.image_projection.requires_grad_(False)
            self.san_model.requires_grad_(False)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
        actual_vocab = self.answer_embedding.token_embeddings.word_embeddings.num_embeddings
        if vocab_size is not None and vocab_size != actual_vocab:
            raise ValueError('vocab_size must match the text model embedding vocabulary')
        self.mlp = nn.Sequential(
            nn.Dropout(p=0.3), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, actual_vocab),
        )

    def encode_from_features(
        self, image_embeddings, question_embeddings, question_padding_mask,
        return_diagnostics=False,
    ):
        if self.fusion_type == 'san':
            raise ValueError('Precomputed token features require a token-level fusion')
        reference_module = self.ot_fusion or self.fusion_module
        if reference_module is None:
            raise ValueError('Fusion does not support precomputed token features')
        reference = next(reference_module.parameters())
        image_embeddings = image_embeddings.to(device=reference.device, dtype=reference.dtype)
        question_embeddings = question_embeddings.to(device=reference.device, dtype=reference.dtype)
        question_padding_mask = question_padding_mask.to(reference.device, dtype=torch.bool)
        visual_tokens = self.image_model.spatial_tokens(image_embeddings)
        visual_padding_mask = torch.zeros(
            visual_tokens.shape[:2], dtype=torch.bool, device=visual_tokens.device
        )
        transport = None
        if self.ot_fusion is not None:
            transport = self.ot_fusion(
                visual_tokens, question_embeddings, visual_padding_mask,
                question_padding_mask, return_diagnostics=return_diagnostics,
            )
        memory = transport.fused_tokens if transport is not None else None
        memory_padding_mask = (
            transport.memory_padding_mask if transport is not None else None
        )
        ot_san = None
        if self.ot_san is not None:
            ot_san = self.ot_san(
                memory, memory_padding_mask,
                return_diagnostics=return_diagnostics,
            )
            memory = ot_san.memory
            memory_padding_mask = ot_san.memory_padding_mask
            transport.ot_san = ot_san
        fusion_output = None
        if self.fusion_module is not None:
            fusion_output = self.fusion_module(
                FusionInput(
                    visual_tokens=visual_tokens,
                    question_tokens=question_embeddings,
                    visual_padding_mask=visual_padding_mask,
                    question_padding_mask=question_padding_mask,
                    transport=transport,
                ),
                return_diagnostics=return_diagnostics,
            )
            if return_diagnostics:
                valid = ~fusion_output.memory_padding_mask
                valid_memory = fusion_output.memory.masked_fill(
                    ~valid.unsqueeze(-1), 0.0
                )
                normalizer = valid.sum(1).clamp_min(1)
                diagnostics = dict(fusion_output.diagnostics or {})
                diagnostics.update({
                    "memory_length": valid.sum(1).float(),
                    "memory_norm": (
                        valid_memory.float().norm(dim=-1).sum(1) / normalizer
                    ),
                })
                fusion_output.diagnostics = diagnostics
            memory = fusion_output.memory
            memory_padding_mask = fusion_output.memory_padding_mask
        if memory is None or memory_padding_mask is None:
            raise RuntimeError(f"Fusion '{self.fusion_type}' produced no decoder memory")
        return EncoderOutput(
            memory=memory,
            memory_padding_mask=memory_padding_mask,
            transport=transport,
            ot_san=ot_san,
            fusion_output=fusion_output,
        )

    def encode(self, images, questions, anno_ids=None, return_diagnostics=False):
        image_embeddings, _ = self.image_model(images, image_ids=anno_ids)
        if self.fusion_type == 'san':
            projected_images = self.image_projection(image_embeddings)
            context = self.question_encoder(questions)
            for layer in self.san_model:
                context = layer(projected_images, context.unsqueeze(1))
            memory = context.unsqueeze(1)
            return EncoderOutput(
                memory=memory,
                memory_padding_mask=torch.zeros(
                    memory.shape[:2], dtype=torch.bool, device=memory.device
                ),
            )
        question_embeddings, question_mask, _ = self.question_encoder.encode_tokens(questions)
        return self.encode_from_features(
            image_embeddings, question_embeddings, question_mask, return_diagnostics
        )

    @staticmethod
    def _unpack_encoder_output(encoded):
        # Accept tensors for older callers and lightweight test mocks.
        if isinstance(encoded, torch.Tensor):
            return encoded, None, None
        return encoded.memory, encoded.memory_padding_mask, encoded.transport

    def decode(self, input_ids, memory, causal=True, memory_padding_mask=None):
        target = self.answer_projection(self.answer_embedding.embed_ids(input_ids))
        target_length = input_ids.size(1)
        blocked = input_ids.eq(self.pad_token_id)[:, None, None, :].expand(
            -1, 1, target_length, -1
        )
        if causal:
            blocked = blocked | torch.triu(
                torch.ones(target_length, target_length, dtype=torch.bool,
                            device=input_ids.device), diagonal=1
            )
        cross_mask = None
        if memory_padding_mask is not None:
            if memory_padding_mask.shape != memory.shape[:2]:
                raise ValueError('memory_padding_mask must match memory batch and length')
            cross_mask = memory_padding_mask[:, None, None, :].expand(
                -1, 1, target_length, -1
            )
        return self.mlp(self.decoder(memory, target, blocked, cross_mask))

    def forward(
        self, images, questions, answers=None, anno_ids=None, mask=True, mode='train',
        max_len=Config.MAX_LEN_ANS, return_diagnostics=False,
    ):
        if mode not in {'train', 'eval', 'test', 'infer', 'generate'}:
            raise ValueError(f'Unknown mode: {mode}')
        if mode in {'test', 'infer', 'generate'} or answers is None:
            return self.generate(
                images, questions, anno_ids, max_len,
                return_diagnostics=return_diagnostics,
            )
        encoded = self.encode(images, questions, anno_ids, return_diagnostics)
        memory, memory_mask, transport = self._unpack_encoder_output(encoded)
        ids = self.answer_embedding.tokenize(answers, max_len)
        logits = self.decode(ids[:, :-1], memory, causal=mask,
                             memory_padding_mask=memory_mask)
        if return_diagnostics:
            return logits, ids[:, 1:], transport
        return logits, ids[:, 1:]

    def forward_from_features(
        self, image_embeddings, question_embeddings, question_padding_mask,
        answers, max_len=Config.MAX_LEN_ANS, mask=True, return_diagnostics=False,
    ):
        """Train from cached frozen-encoder outputs using the normal OT decoder path."""
        encoded = self.encode_from_features(
            image_embeddings, question_embeddings, question_padding_mask,
            return_diagnostics,
        )
        memory, memory_mask, transport = self._unpack_encoder_output(encoded)
        ids = self.answer_embedding.tokenize(answers, max_len)
        logits = self.decode(ids[:, :-1], memory, causal=mask,
                             memory_padding_mask=memory_mask)
        if return_diagnostics:
            return logits, ids[:, 1:], transport
        return logits, ids[:, 1:]

    @torch.no_grad()
    def generate(
        self, images, questions, anno_ids=None, max_len=Config.MAX_LEN_ANS,
        return_diagnostics=False,
    ):
        if not 2 <= max_len <= Config.MAX_LEN_ANS:
            raise ValueError(f'max_len must be between 2 and {Config.MAX_LEN_ANS}')
        was_training = self.training
        self.eval()
        try:
            encoded = self.encode(images, questions, anno_ids, return_diagnostics)
            memory, memory_mask, transport = self._unpack_encoder_output(encoded)
            generated = self._generate_from_memory(memory, memory_mask, max_len)
            if return_diagnostics:
                return GenerationOutput(
                    generated_ids=generated,
                    transport=transport,
                    ot_san=getattr(encoded, 'ot_san', None),
                    fusion_output=getattr(encoded, 'fusion_output', None),
                )
            return generated
        finally:
            self.train(was_training)

    def _generate_from_memory(self, memory, memory_padding_mask, max_len):
        batch_size = memory.size(0)
        ids = torch.full(
            (batch_size, 1), self.bos_token_id, dtype=torch.long,
            device=memory.device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=memory.device)
        for _ in range(max_len - 1):
            logits = self.decode(
                ids, memory, memory_padding_mask=memory_padding_mask
            )[:, -1, :]
            logits[:, [self.pad_token_id, self.bos_token_id]] = float('-inf')
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
            raise ValueError(f'max_len must be between 2 and {Config.MAX_LEN_ANS}')
        was_training = self.training
        self.eval()
        try:
            encoded = self.encode_from_features(
                image_embeddings, question_embeddings, question_padding_mask,
                return_diagnostics,
            )
            memory, memory_mask, transport = self._unpack_encoder_output(encoded)
            generated = self._generate_from_memory(memory, memory_mask, max_len)
            if return_diagnostics:
                return GenerationOutput(
                    generated_ids=generated,
                    transport=transport,
                    ot_san=getattr(encoded, 'ot_san', None),
                    fusion_output=getattr(encoded, 'fusion_output', None),
                )
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
