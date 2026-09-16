import argparse
import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder


@dataclass
class QuMLAGConfig:
    vocab_size: int
    image_feature_dim: int = 2048
    fasttext_dim: int = 300
    hidden_dim: int = 512
    num_heads: int = 8
    num_sa_layers: int = 4
    num_ga_layers: int = 4
    num_decoder_layers: int = 3
    ff_dim: int = 2048
    dropout: float = 0.1
    max_question_len: int = 64
    max_answer_len: int = 48
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2


class FastTextEmbedding(nn.Module):
    """
    Token embedding layer intended for FastText vectors.
    Pass `pretrained_weights` with shape [vocab_size, fasttext_dim] when available.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        pad_token_id: int,
        pretrained_weights: Optional[torch.Tensor] = None,
        freeze: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_token_id,
        )
        if pretrained_weights is not None:
            if pretrained_weights.shape != (vocab_size, embedding_dim):
                raise ValueError(
                    "pretrained_weights must have shape "
                    f"[{vocab_size}, {embedding_dim}]"
                )
            with torch.no_grad():
                self.embedding.weight.copy_(pretrained_weights)
        self.embedding.weight.requires_grad = not freeze
        self.proj = nn.Linear(embedding_dim, hidden_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embedding(token_ids))


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, hidden_dim: int):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return x + self.pos_embedding(pos_ids)


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            query=x_norm, key=x_norm, value=x_norm, key_padding_mask=key_padding_mask
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class GuidedAttentionBlock(nn.Module):
    """
    Question-guided attention: image tokens query question tokens.
    """

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        image_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        question_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        q = self.norm_q(image_tokens)
        kv = question_tokens
        guided_out, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=question_padding_mask,
        )
        image_tokens = image_tokens + guided_out
        image_tokens = image_tokens + self.ffn(self.norm_out(image_tokens))
        return image_tokens


class DecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        target_tokens: torch.Tensor,
        memory_tokens: torch.Tensor,
        causal_mask: torch.Tensor,
        target_padding_mask: Optional[torch.Tensor],
        memory_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        t = self.norm1(target_tokens)
        self_out, _ = self.self_attn(
            query=t,
            key=t,
            value=t,
            attn_mask=causal_mask,
            key_padding_mask=target_padding_mask,
        )
        target_tokens = target_tokens + self_out

        t = self.norm2(target_tokens)
        cross_out, _ = self.cross_attn(
            query=t,
            key=memory_tokens,
            value=memory_tokens,
            key_padding_mask=memory_padding_mask,
        )
        target_tokens = target_tokens + cross_out
        target_tokens = target_tokens + self.ffn(self.norm3(target_tokens))
        return target_tokens


class CrossAttentionBlock(nn.Module):
    """
    Generic cross-attention block:
    query tokens attend to context tokens, followed by FFN.
    """

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        context_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        q = self.norm1(query_tokens)
        attn_out, _ = self.cross_attn(
            query=q,
            key=context_tokens,
            value=context_tokens,
            key_padding_mask=context_padding_mask,
        )
        query_tokens = query_tokens + attn_out
        query_tokens = query_tokens + self.ffn(self.norm2(query_tokens))
        return query_tokens


@dataclass
class MLPAGConfig:
    vocab_size: int
    image_feature_dim: int = 2048
    fasttext_dim: int = 300
    hidden_dim: int = 512
    num_heads: int = 8
    ff_dim: int = 2048
    num_decoder_layers: int = 3
    dropout: float = 0.1
    max_question_len: int = 64
    max_answer_len: int = 48
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    unk_token_id: int = 3


class MLPAG(nn.Module):
    """
    MultiModal Learning and Pointer-augmented Answer Generator.
    - Embedding: image regions + question tokens + scene-text tokens.
    - Fusion: question self-attn, then context/spatial cross-attn.
    - Generator: Transformer decoder + dynamic pointer over scene text.
    """

    def __init__(self, config: MLPAGConfig, pretrained_fasttext: Optional[torch.Tensor] = None):
        super().__init__()
        self.config = config

        self.text_embedding = FastTextEmbedding(
            vocab_size=config.vocab_size,
            embedding_dim=config.fasttext_dim,
            hidden_dim=config.hidden_dim,
            pad_token_id=config.pad_token_id,
            pretrained_weights=pretrained_fasttext,
            freeze=False,
        )
        self.question_pos = LearnedPositionalEncoding(config.max_question_len, config.hidden_dim)
        self.answer_pos = LearnedPositionalEncoding(config.max_answer_len, config.hidden_dim)

        self.image_proj = nn.Linear(config.image_feature_dim, config.hidden_dim)
        self.image_norm = nn.LayerNorm(config.hidden_dim)
        self.image_dropout = nn.Dropout(config.dropout)

        self.question_self_attention = SelfAttentionBlock(
            config.hidden_dim, config.num_heads, config.ff_dim, config.dropout
        )
        self.context_attention = CrossAttentionBlock(
            config.hidden_dim, config.num_heads, config.ff_dim, config.dropout
        )
        self.spatial_attention = CrossAttentionBlock(
            config.hidden_dim, config.num_heads, config.ff_dim, config.dropout
        )

        self.decoder_layers = nn.ModuleList(
            [
                DecoderBlock(config.hidden_dim, config.num_heads, config.ff_dim, config.dropout)
                for _ in range(config.num_decoder_layers)
            ]
        )

        # Dynamic pointer network parameters from Eq. (11).
        self.ptr_h = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.ptr_s = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.vocab_proj = nn.Linear(config.hidden_dim, config.vocab_size)

    def build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
            diagonal=1,
        )

    def masked_mean(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if pad_mask is None:
            return x.mean(dim=1)
        valid = (~pad_mask).to(x.dtype).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (x * valid).sum(dim=1) / denom

    def encode(
        self,
        question_token_ids: torch.Tensor,
        scene_token_ids: torch.Tensor,
        image_region_features: torch.Tensor,
        question_padding_mask: Optional[torch.Tensor],
        scene_padding_mask: Optional[torch.Tensor],
        image_padding_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if question_padding_mask is None:
            question_padding_mask = question_token_ids.eq(self.config.pad_token_id)
        if scene_padding_mask is None:
            scene_padding_mask = scene_token_ids.eq(self.config.pad_token_id)
        if image_padding_mask is None:
            image_padding_mask = image_region_features.abs().sum(dim=-1).eq(0)

        x_q = self.text_embedding(question_token_ids)
        x_q = self.question_pos(x_q)
        a_q = self.question_self_attention(x_q, question_padding_mask)

        x_s = self.text_embedding(scene_token_ids)
        x_s = self.context_attention(
            query_tokens=x_s,
            context_tokens=a_q,
            context_padding_mask=question_padding_mask,
        )

        x_i = self.image_dropout(self.image_norm(self.image_proj(image_region_features)))
        x_i = self.spatial_attention(
            query_tokens=x_i,
            context_tokens=a_q,
            context_padding_mask=question_padding_mask,
        )

        # Eq. (9): fuse scene-text and image information with element-wise sum.
        # Here image stream is pooled to a global vector then broadcast to scene tokens.
        x_i_global = self.masked_mean(x_i, image_padding_mask).unsqueeze(1)  # [B, 1, D]
        x_f = x_s + x_i_global  # [B, N_scene, D]
        return x_f, x_s, scene_padding_mask

    def decode_hidden(
        self,
        decoder_input_ids: torch.Tensor,
        memory_tokens: torch.Tensor,
        memory_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        target = self.text_embedding(decoder_input_ids)
        target = self.answer_pos(target)
        causal_mask = self.build_causal_mask(target.shape[1], target.device)
        target_padding_mask = decoder_input_ids.eq(self.config.pad_token_id)
        for layer in self.decoder_layers:
            target = layer(
                target_tokens=target,
                memory_tokens=memory_tokens,
                causal_mask=causal_mask,
                target_padding_mask=target_padding_mask,
                memory_padding_mask=memory_padding_mask,
            )
        return target

    def compute_scores(
        self,
        decoder_hidden: torch.Tensor,
        scene_tokens: torch.Tensor,
        scene_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        vocab_scores = self.vocab_proj(decoder_hidden)  # [B, T, V]
        ptr_q = self.ptr_h(decoder_hidden)  # [B, T, D]
        ptr_k = self.ptr_s(scene_tokens)  # [B, N, D]
        ptr_scores = torch.matmul(ptr_q, ptr_k.transpose(1, 2))  # [B, T, N]
        if scene_padding_mask is not None:
            ptr_scores = ptr_scores.masked_fill(scene_padding_mask.unsqueeze(1), float("-inf"))
        return torch.cat([vocab_scores, ptr_scores], dim=-1)  # [B, T, V + N]

    def map_extended_to_vocab_ids(
        self,
        extended_ids: torch.Tensor,
        scene_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert generated ids in extended space [vocab | scene pointers] back to vocab ids
        so they can be re-embedded at the next decoding step.
        """
        vocab_ids = extended_ids.clone()
        is_ptr = extended_ids.ge(self.config.vocab_size)
        if is_ptr.any():
            ptr_idx = (extended_ids - self.config.vocab_size).clamp_min(0)
            ptr_idx = ptr_idx.clamp_max(scene_token_ids.size(1) - 1)
            copied_vocab = scene_token_ids.gather(1, ptr_idx)
            vocab_ids = torch.where(is_ptr, copied_vocab, vocab_ids)
        vocab_ids = vocab_ids.clamp_max(self.config.vocab_size - 1)
        vocab_ids = torch.where(
            vocab_ids.eq(self.config.pad_token_id),
            torch.full_like(vocab_ids, self.config.unk_token_id),
            vocab_ids,
        )
        return vocab_ids

    def forward(
        self,
        question_token_ids: torch.Tensor,
        scene_token_ids: torch.Tensor,
        image_region_features: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        question_padding_mask: Optional[torch.Tensor] = None,
        scene_padding_mask: Optional[torch.Tensor] = None,
        image_padding_mask: Optional[torch.Tensor] = None,
        max_decode_len: Optional[int] = None,
    ):
        """
        Training:
            provide `decoder_input_ids` (answer tokens shifted-right with BOS).
            returns {"scores": [B, T, vocab_size + num_scene_tokens]}.

        Inference:
            leave `decoder_input_ids=None`.
            returns generated tokens in extended space and step scores.
        """
        memory_tokens, scene_tokens, scene_pad = self.encode(
            question_token_ids=question_token_ids,
            scene_token_ids=scene_token_ids,
            image_region_features=image_region_features,
            question_padding_mask=question_padding_mask,
            scene_padding_mask=scene_padding_mask,
            image_padding_mask=image_padding_mask,
        )

        if decoder_input_ids is not None:
            hidden = self.decode_hidden(
                decoder_input_ids=decoder_input_ids,
                memory_tokens=memory_tokens,
                memory_padding_mask=scene_pad,
            )
            scores = self.compute_scores(hidden, scene_tokens, scene_pad)
            return {"scores": scores}

        max_len = max_decode_len or self.config.max_answer_len
        batch_size = question_token_ids.size(0)
        generated_ext = torch.full(
            (batch_size, 1),
            fill_value=self.config.bos_token_id,
            dtype=torch.long,
            device=question_token_ids.device,
        )
        generated_vocab = generated_ext.clone()
        step_scores = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=question_token_ids.device)

        for _ in range(max_len):
            hidden = self.decode_hidden(
                decoder_input_ids=generated_vocab,
                memory_tokens=memory_tokens,
                memory_padding_mask=scene_pad,
            )
            scores = self.compute_scores(hidden, scene_tokens, scene_pad)
            last_scores = scores[:, -1, :]
            next_ext = last_scores.argmax(dim=-1, keepdim=True)
            next_vocab = self.map_extended_to_vocab_ids(next_ext, scene_token_ids)

            step_scores.append(last_scores.unsqueeze(1))
            generated_ext = torch.cat([generated_ext, next_ext], dim=1)
            generated_vocab = torch.cat([generated_vocab, next_vocab], dim=1)

            finished = finished | next_vocab.squeeze(1).eq(self.config.eos_token_id)
            if finished.all():
                break

        all_scores = torch.cat(step_scores, dim=1) if step_scores else torch.empty(
            batch_size,
            0,
            self.config.vocab_size + scene_token_ids.size(1),
            device=question_token_ids.device,
        )
        return {"scores": all_scores, "generated_ids": generated_ext[:, 1:]}


class QuMLAG(nn.Module):
    """
    QuMLAG/GMCAN re-implementation:
    - Text Embedding: FastText-based token embedding.
    - Information Fusion: 4 SA layers + 4 GA layers (default).
    - Answer Generator: 3 decoder layers, each with self-attention + cross-attention.
    """

    def __init__(self, config: QuMLAGConfig, pretrained_fasttext: Optional[torch.Tensor] = None):
        super().__init__()
        self.config = config

        self.text_embedding = FastTextEmbedding(
            vocab_size=config.vocab_size,
            embedding_dim=config.fasttext_dim,
            hidden_dim=config.hidden_dim,
            pad_token_id=config.pad_token_id,
            pretrained_weights=pretrained_fasttext,
            freeze=False,
        )
        self.question_pos = LearnedPositionalEncoding(config.max_question_len, config.hidden_dim)
        self.answer_pos = LearnedPositionalEncoding(config.max_answer_len, config.hidden_dim)

        # FasterRCNN/ResNeXt152++ region features are expected to be pre-extracted.
        self.image_proj = nn.Linear(config.image_feature_dim, config.hidden_dim)
        self.image_norm = nn.LayerNorm(config.hidden_dim)
        self.image_dropout = nn.Dropout(config.dropout)

        self.question_sa_layers = nn.ModuleList(
            [
                SelfAttentionBlock(config.hidden_dim, config.num_heads, config.ff_dim, config.dropout)
                for _ in range(config.num_sa_layers)
            ]
        )
        self.image_sa_layers = nn.ModuleList(
            [
                SelfAttentionBlock(config.hidden_dim, config.num_heads, config.ff_dim, config.dropout)
                for _ in range(config.num_sa_layers)
            ]
        )
        self.guided_attention_layers = nn.ModuleList(
            [
                GuidedAttentionBlock(config.hidden_dim, config.num_heads, config.ff_dim, config.dropout)
                for _ in range(config.num_ga_layers)
            ]
        )

        self.decoder_layers = nn.ModuleList(
            [
                DecoderBlock(config.hidden_dim, config.num_heads, config.ff_dim, config.dropout)
                for _ in range(config.num_decoder_layers)
            ]
        )
        self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size)

    def build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def encode(
        self,
        question_token_ids: torch.Tensor,
        image_region_features: torch.Tensor,
        question_padding_mask: Optional[torch.Tensor],
        image_padding_mask: Optional[torch.Tensor],
    ):
        question_tokens = self.text_embedding(question_token_ids)
        question_tokens = self.question_pos(question_tokens)

        image_tokens = self.image_dropout(self.image_norm(self.image_proj(image_region_features)))

        for q_sa, i_sa in zip(self.question_sa_layers, self.image_sa_layers):
            question_tokens = q_sa(question_tokens, question_padding_mask)
            image_tokens = i_sa(image_tokens, image_padding_mask)

        for ga in self.guided_attention_layers:
            image_tokens = ga(image_tokens, question_tokens, question_padding_mask)

        memory_tokens = torch.cat([question_tokens, image_tokens], dim=1)
        if question_padding_mask is None and image_padding_mask is None:
            memory_padding_mask = None
        elif question_padding_mask is None:
            memory_padding_mask = image_padding_mask
        elif image_padding_mask is None:
            memory_padding_mask = question_padding_mask
        else:
            memory_padding_mask = torch.cat([question_padding_mask, image_padding_mask], dim=1)

        return memory_tokens, memory_padding_mask

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        memory_tokens: torch.Tensor,
        memory_padding_mask: Optional[torch.Tensor],
        target_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = self.text_embedding(decoder_input_ids)
        target = self.answer_pos(target)
        causal_mask = self.build_causal_mask(target.shape[1], target.device)
        for layer in self.decoder_layers:
            target = layer(
                target_tokens=target,
                memory_tokens=memory_tokens,
                causal_mask=causal_mask,
                target_padding_mask=target_padding_mask,
                memory_padding_mask=memory_padding_mask,
            )
        return self.output_proj(target)

    def forward(
        self,
        question_token_ids: torch.Tensor,
        image_region_features: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        question_padding_mask: Optional[torch.Tensor] = None,
        image_padding_mask: Optional[torch.Tensor] = None,
        max_decode_len: Optional[int] = None,
    ):
        """
        Training:
            provide `decoder_input_ids` (usually answer shifted-right with BOS).
            returns {"logits": [B, T, vocab_size]}.

        Inference:
            leave `decoder_input_ids=None`.
            returns {"generated_ids": [B, T], "logits": [B, T, vocab_size]}.
        """
        memory_tokens, memory_padding_mask = self.encode(
            question_token_ids=question_token_ids,
            image_region_features=image_region_features,
            question_padding_mask=question_padding_mask,
            image_padding_mask=image_padding_mask,
        )

        if decoder_input_ids is not None:
            target_padding_mask = decoder_input_ids.eq(self.config.pad_token_id)
            logits = self.decode(
                decoder_input_ids=decoder_input_ids,
                memory_tokens=memory_tokens,
                memory_padding_mask=memory_padding_mask,
                target_padding_mask=target_padding_mask,
            )
            return {"logits": logits}

        max_len = max_decode_len or self.config.max_answer_len
        batch_size = question_token_ids.size(0)
        generated = torch.full(
            (batch_size, 1),
            fill_value=self.config.bos_token_id,
            dtype=torch.long,
            device=question_token_ids.device,
        )
        step_logits = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=question_token_ids.device)

        for _ in range(max_len):
            logits = self.decode(
                decoder_input_ids=generated,
                memory_tokens=memory_tokens,
                memory_padding_mask=memory_padding_mask,
                target_padding_mask=generated.eq(self.config.pad_token_id),
            )
            last_step = logits[:, -1, :]
            next_token = last_step.argmax(dim=-1, keepdim=True)
            step_logits.append(last_step.unsqueeze(1))
            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | next_token.squeeze(1).eq(self.config.eos_token_id)
            if finished.all():
                break

        if step_logits:
            all_logits = torch.cat(step_logits, dim=1)
        else:
            all_logits = torch.empty(
                batch_size, 0, self.config.vocab_size, device=question_token_ids.device
            )

        return {"generated_ids": generated[:, 1:], "logits": all_logits}


@dataclass
class M4CConfig:
    vocab_size: int
    d_model: int = 768
    object_feat_dim: int = 2048
    ocr_det_feat_dim: int = 2048
    ocr_rec_feat_dim: int = 256
    ocr_fasttext_dim: int = 300
    num_heads: int = 8
    num_mmt_layers: int = 4
    num_question_layers: int = 4
    max_answer_len: int = 48
    dropout: float = 0.1
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    bert_model_name: str = "bert-base-uncased"
    pretrained_bert: bool = True


class DynamicPointerNetwork(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

    def forward(
        self,
        decoder_tokens: torch.Tensor,
        ocr_tokens: torch.Tensor,
        ocr_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.query(decoder_tokens)  # [B, T, D]
        k = self.key(ocr_tokens)  # [B, O, D]
        scores = torch.matmul(q, k.transpose(1, 2)) / self.scale  # [B, T, O]
        if ocr_padding_mask is not None:
            scores = scores.masked_fill(ocr_padding_mask.unsqueeze(1), float("-inf"))
        return scores


class M4C(nn.Module):
    """
    M4C re-implementation:
    - Question embedding by BERT (bert-base-uncased default).
    - Multimodal encoder uses BERT encoder blocks.
    - Dynamic Pointer Network for vocab + OCR token generation.
    """

    def __init__(self, config: M4CConfig):
        super().__init__()
        self.config = config

        # BERT question embedding and question encoder
        q_bert_cfg = BertConfig(
            hidden_size=config.d_model,
            num_hidden_layers=config.num_question_layers,
            num_attention_heads=config.num_heads,
            intermediate_size=config.d_model * 4,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout,
            vocab_size=30522,
        )
        q_bert_cfg._attn_implementation = "eager"
        self.question_embeddings = BertEmbeddings(q_bert_cfg)
        self.question_encoder = BertEncoder(q_bert_cfg)

        # MMT encoder (BERT encoder blocks)
        mmt_cfg = BertConfig(
            hidden_size=config.d_model,
            num_hidden_layers=config.num_mmt_layers,
            num_attention_heads=config.num_heads,
            intermediate_size=config.d_model * 4,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout,
            vocab_size=30522,
        )
        mmt_cfg._attn_implementation = "eager"
        self.mmt_encoder = BertEncoder(mmt_cfg)

        # Object embedding
        self.obj_feat_proj = nn.Linear(config.object_feat_dim, config.d_model)
        self.obj_bbox_proj = nn.Linear(4, config.d_model)
        self.obj_norm_feat = nn.LayerNorm(config.d_model)
        self.obj_norm_bbox = nn.LayerNorm(config.d_model)
        self.obj_drop = nn.Dropout(config.dropout)

        # OCR embedding
        ocr_in_dim = config.ocr_det_feat_dim + config.ocr_rec_feat_dim + config.ocr_fasttext_dim
        self.ocr_feat_proj = nn.Linear(ocr_in_dim, config.d_model)
        self.ocr_bbox_proj = nn.Linear(4, config.d_model)
        self.ocr_norm_feat = nn.LayerNorm(config.d_model)
        self.ocr_norm_bbox = nn.LayerNorm(config.d_model)
        self.ocr_drop = nn.Dropout(config.dropout)

        # Answer embedding (previous tokens) as in M4C decoding-by-encoder style
        self.answer_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.answer_pos = nn.Embedding(config.max_answer_len, config.d_model)

        # Output heads
        self.vocab_proj = nn.Linear(config.d_model, config.vocab_size)
        self.dynamic_pointer = DynamicPointerNetwork(config.d_model)
        self.load_pretrained_bert_if_needed()

    def load_pretrained_bert_if_needed(self) -> None:
        if not self.config.pretrained_bert:
            return
        pretrained = BertModel.from_pretrained(self.config.bert_model_name)
        if pretrained.config.hidden_size != self.config.d_model:
            raise ValueError(
                "d_model must match pretrained BERT hidden size. "
                f"Got d_model={self.config.d_model}, "
                f"bert_hidden={pretrained.config.hidden_size}."
            )

        # Question embedding module initialized from pretrained BERT embeddings.
        self.question_embeddings.load_state_dict(pretrained.embeddings.state_dict(), strict=True)

        # Initialize question encoder layers from pretrained BERT encoder.
        for i, layer in enumerate(self.question_encoder.layer):
            layer.load_state_dict(pretrained.encoder.layer[i].state_dict(), strict=True)

        # Initialize MMT encoder with pretrained BERT encoder layers too.
        for i, layer in enumerate(self.mmt_encoder.layer):
            layer.load_state_dict(pretrained.encoder.layer[i].state_dict(), strict=True)

    def make_padding_mask_from_features(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, D] -> [B, L], True means pad
        return x.abs().sum(dim=-1).eq(0)

    def build_autoregressive_joint_mask(
        self,
        obj_len: int,
        ocr_len: int,
        q_len: int,
        ans_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_len = obj_len + ocr_len + q_len + ans_len
        # True means masked for BertEncoder attention mask
        mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
        ans_start = obj_len + ocr_len + q_len
        if ans_len > 0:
            causal = torch.triu(torch.ones(ans_len, ans_len, dtype=torch.bool, device=device), diagonal=1)
            mask[ans_start:, ans_start:] = causal
        return mask

    def encode_question(
        self, question_token_ids: torch.Tensor, question_padding_mask: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if question_padding_mask is None:
            question_padding_mask = question_token_ids.eq(self.config.pad_token_id)
        q_emb = self.question_embeddings(input_ids=question_token_ids)
        q_attn_mask = question_padding_mask[:, None, None, :].to(dtype=q_emb.dtype) * -10000.0
        q_feats = self.question_encoder(q_emb, attention_mask=q_attn_mask).last_hidden_state
        return q_feats, question_padding_mask

    def encode_objects(
        self, obj_features: torch.Tensor, obj_boxes: torch.Tensor, obj_padding_mask: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if obj_padding_mask is None:
            obj_padding_mask = self.make_padding_mask_from_features(obj_features)
        x = self.obj_norm_feat(self.obj_feat_proj(obj_features)) + self.obj_norm_bbox(
            self.obj_bbox_proj(obj_boxes)
        )
        return self.obj_drop(x), obj_padding_mask

    def encode_ocr(
        self,
        ocr_det_features: torch.Tensor,
        ocr_rec_features: torch.Tensor,
        ocr_fasttext_features: torch.Tensor,
        ocr_boxes: torch.Tensor,
        ocr_padding_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ocr_padding_mask is None:
            ocr_padding_mask = self.make_padding_mask_from_features(ocr_det_features)
        ocr_det_features = F.normalize(ocr_det_features, dim=-1)
        ocr_rec_features = F.normalize(ocr_rec_features, dim=-1)
        ocr_fasttext_features = F.normalize(ocr_fasttext_features, dim=-1)
        ocr_joint = torch.cat([ocr_det_features, ocr_rec_features, ocr_fasttext_features], dim=-1)
        x = self.ocr_norm_feat(self.ocr_feat_proj(ocr_joint)) + self.ocr_norm_bbox(
            self.ocr_bbox_proj(ocr_boxes)
        )
        return self.ocr_drop(x), ocr_padding_mask

    def encode_answer_tokens(self, answer_prev_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ans_pad = answer_prev_ids.eq(self.config.pad_token_id)
        ans_emb = self.answer_embedding(answer_prev_ids)
        pos = torch.arange(answer_prev_ids.size(1), device=answer_prev_ids.device).unsqueeze(0)
        ans_emb = ans_emb + self.answer_pos(pos)
        return ans_emb, ans_pad

    def mmt_forward(
        self,
        obj_tokens: torch.Tensor,
        obj_pad: torch.Tensor,
        ocr_tokens: torch.Tensor,
        ocr_pad: torch.Tensor,
        q_tokens: torch.Tensor,
        q_pad: torch.Tensor,
        ans_tokens: torch.Tensor,
        ans_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        joint = torch.cat([obj_tokens, ocr_tokens, q_tokens, ans_tokens], dim=1)
        joint_pad = torch.cat([obj_pad, ocr_pad, q_pad, ans_pad], dim=1)
        obj_len, ocr_len, q_len, ans_len = (
            obj_tokens.size(1),
            ocr_tokens.size(1),
            q_tokens.size(1),
            ans_tokens.size(1),
        )
        total_len = joint.size(1)

        # base padding mask over keys
        key_pad = joint_pad[:, None, None, :].to(dtype=joint.dtype) * -10000.0
        # causal only for answer block
        causal_bool = self.build_autoregressive_joint_mask(
            obj_len=obj_len,
            ocr_len=ocr_len,
            q_len=q_len,
            ans_len=ans_len,
            device=joint.device,
        )
        causal = causal_bool[None, None, :, :].to(dtype=joint.dtype) * -10000.0
        attn_mask = key_pad.expand(-1, -1, total_len, -1) + causal

        encoded = self.mmt_encoder(joint, attention_mask=attn_mask).last_hidden_state

        ocr_begin = obj_len
        ocr_end = ocr_begin + ocr_len
        ans_begin = obj_len + ocr_len + q_len
        ans_end = ans_begin + ans_len

        ocr_encoded = encoded[:, ocr_begin:ocr_end]
        ans_encoded = encoded[:, ans_begin:ans_end]
        return ans_encoded, ocr_encoded, ocr_pad

    def compute_scores(
        self, ans_encoded: torch.Tensor, ocr_encoded: torch.Tensor, ocr_pad: torch.Tensor
    ) -> torch.Tensor:
        vocab_scores = self.vocab_proj(ans_encoded)  # [B, T, V]
        ptr_scores = self.dynamic_pointer(ans_encoded, ocr_encoded, ocr_padding_mask=ocr_pad)  # [B, T, O]
        return torch.cat([vocab_scores, ptr_scores], dim=-1)  # [B, T, V+O]

    def forward(
        self,
        question_token_ids: torch.Tensor,
        obj_features: torch.Tensor,
        obj_boxes: torch.Tensor,
        ocr_det_features: torch.Tensor,
        ocr_rec_features: torch.Tensor,
        ocr_fasttext_features: torch.Tensor,
        ocr_boxes: torch.Tensor,
        answer_prev_ids: Optional[torch.Tensor] = None,
        question_padding_mask: Optional[torch.Tensor] = None,
        obj_padding_mask: Optional[torch.Tensor] = None,
        ocr_padding_mask: Optional[torch.Tensor] = None,
        max_decode_len: Optional[int] = None,
    ):
        """
        Training:
            `answer_prev_ids` is teacher-forcing input ids (shift-right target).
            Return {"scores": [B, T, vocab_size + num_ocr_tokens]}.

        Inference:
            leave `answer_prev_ids=None`, run greedy decoding.
        """
        q_tokens, q_pad = self.encode_question(question_token_ids, question_padding_mask)
        obj_tokens, obj_pad = self.encode_objects(obj_features, obj_boxes, obj_padding_mask)
        ocr_tokens, ocr_pad = self.encode_ocr(
            ocr_det_features,
            ocr_rec_features,
            ocr_fasttext_features,
            ocr_boxes,
            ocr_padding_mask,
        )

        if answer_prev_ids is not None:
            ans_tokens, ans_pad = self.encode_answer_tokens(answer_prev_ids)
            ans_encoded, ocr_encoded, ocr_pad = self.mmt_forward(
                obj_tokens, obj_pad, ocr_tokens, ocr_pad, q_tokens, q_pad, ans_tokens, ans_pad
            )
            scores = self.compute_scores(ans_encoded, ocr_encoded, ocr_pad)
            return {"scores": scores}

        max_len = max_decode_len or self.config.max_answer_len
        batch_size = question_token_ids.size(0)
        generated = torch.full(
            (batch_size, max_len),
            fill_value=self.config.pad_token_id,
            dtype=torch.long,
            device=question_token_ids.device,
        )
        generated[:, 0] = self.config.bos_token_id

        latest_scores = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=question_token_ids.device)
        for step in range(max_len):
            cur_ids = generated[:, : step + 1]
            ans_tokens, ans_pad = self.encode_answer_tokens(cur_ids)
            ans_encoded, ocr_encoded, ocr_pad = self.mmt_forward(
                obj_tokens, obj_pad, ocr_tokens, ocr_pad, q_tokens, q_pad, ans_tokens, ans_pad
            )
            scores = self.compute_scores(ans_encoded, ocr_encoded, ocr_pad)
            latest_scores = scores
            next_ids = scores[:, -1, : self.config.vocab_size].argmax(dim=-1)
            if step + 1 < max_len:
                generated[:, step + 1] = next_ids
            finished = finished | next_ids.eq(self.config.eos_token_id)
            if finished.all():
                break
        return {"scores": latest_scores, "generated_ids": generated}


def tokensFromIds(
    ids: torch.Tensor,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
) -> list[list[int]]:
    ids = ids.detach().cpu()
    batch_tokens: list[list[int]] = []
    for row in ids:
        tokens: list[int] = []
        for token in row.tolist():
            if token == bos_token_id:
                continue
            if token == eos_token_id:
                break
            if token == pad_token_id:
                continue
            tokens.append(token)
        batch_tokens.append(tokens)
    return batch_tokens


def lcsLength(a: list[int], b: list[int]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len(a)][len(b)]


def countNgrams(tokens: list[int], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpusBleuN(
    references: list[list[int]],
    hypotheses: list[list[int]],
    max_n: int,
) -> float:
    if max_n < 1:
        return 0.0

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        clipped_total = 0
        cand_total = 0
        for ref, hyp in zip(references, hypotheses):
            ref_counts = countNgrams(ref, n)
            hyp_counts = countNgrams(hyp, n)
            cand_total += sum(hyp_counts.values())
            for gram, c in hyp_counts.items():
                clipped_total += min(c, ref_counts.get(gram, 0))

        if cand_total == 0 or clipped_total == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped_total / cand_total)

    if any(p == 0.0 for p in precisions):
        return 0.0

    ref_len = sum(len(r) for r in references)
    hyp_len = sum(len(h) for h in hypotheses)
    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - (ref_len / hyp_len))
    geo_mean = math.exp(sum((1.0 / max_n) * math.log(p) for p in precisions))
    return bp * geo_mean


def rougeL(
    references: list[list[int]],
    hypotheses: list[list[int]],
) -> float:
    if not references:
        return 0.0

    total = 0.0
    for ref, hyp in zip(references, hypotheses):
        if len(ref) == 0 and len(hyp) == 0:
            total += 1.0
            continue
        if len(ref) == 0 or len(hyp) == 0:
            total += 0.0
            continue

        lcs = lcsLength(ref, hyp)
        prec = lcs / len(hyp)
        rec = lcs / len(ref)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        total += f1
    return total / len(references)


def computeBleuRouge(
    reference_ids: torch.Tensor,
    predicted_ids: torch.Tensor,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
) -> dict[str, float]:
    refs = tokensFromIds(reference_ids, pad_token_id, bos_token_id, eos_token_id)
    hyps = tokensFromIds(predicted_ids, pad_token_id, bos_token_id, eos_token_id)
    return {
        "BLEU-1": corpusBleuN(refs, hyps, max_n=1),
        "BLEU-2": corpusBleuN(refs, hyps, max_n=2),
        "BLEU-3": corpusBleuN(refs, hyps, max_n=3),
        "BLEU-4": corpusBleuN(refs, hyps, max_n=4),
        "ROUGE-L": rougeL(refs, hyps),
    }


def printMetrics(tag: str, metrics: dict[str, float]) -> None:
    print(
        f"[{tag}][metrics] "
        f"BLEU-1: {metrics['BLEU-1']:.4f}, "
        f"BLEU-2: {metrics['BLEU-2']:.4f}, "
        f"BLEU-3: {metrics['BLEU-3']:.4f}, "
        f"BLEU-4: {metrics['BLEU-4']:.4f}, "
        f"ROUGE-L: {metrics['ROUGE-L']:.4f}"
    )


def build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quick runnable entrypoint for all re-implemented VQA models."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["mlpag", "qumlag", "m4c", "all"],
        help="Select which model runner to execute.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=64001)
    parser.add_argument("--num-regions", type=int, default=36)
    parser.add_argument("--question-len", type=int, default=20)
    parser.add_argument("--scene-len", type=int, default=16)
    parser.add_argument("--answer-len", type=int, default=12)
    parser.add_argument("--image-feature-dim", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    return parser


def run_mlpag(args: argparse.Namespace, device: torch.device) -> None:
    cfg = MLPAGConfig(
        vocab_size=args.vocab_size,
        image_feature_dim=args.image_feature_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.heads,
        num_decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        max_question_len=args.question_len,
        max_answer_len=args.answer_len,
    )
    model = MLPAG(cfg).to(device)
    model.train()

    # Fake tensors to verify the forward path end-to-end.
    question_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.question_len), device=device)
    scene_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.scene_len), device=device)
    image_feats = torch.randn(
        args.batch_size, args.num_regions, args.image_feature_dim, device=device
    )
    decoder_input_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.answer_len), device=device)
    train_out = model(
        question_token_ids=question_ids,
        scene_token_ids=scene_ids,
        image_region_features=image_feats,
        decoder_input_ids=decoder_input_ids,
    )

    # Minimal training-like loss: predict each step token over extended vocab.
    train_scores = train_out["scores"]  # [B, T, V+N]
    train_target = torch.randint(
        0, cfg.vocab_size + args.scene_len, (args.batch_size, args.answer_len), device=device
    )
    loss = F.cross_entropy(
        train_scores.reshape(-1, train_scores.size(-1)),
        train_target.reshape(-1),
    )
    print(f"[MLPAG][train] scores shape: {tuple(train_scores.shape)}, loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        infer_out = model(
            question_token_ids=question_ids,
            scene_token_ids=scene_ids,
            image_region_features=image_feats,
            max_decode_len=args.answer_len,
        )
    print(
        f"[MLPAG][infer] generated_ids shape: {tuple(infer_out['generated_ids'].shape)}, "
        f"scores shape: {tuple(infer_out['scores'].shape)}"
    )
    reference_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.answer_len), device=device)
    predicted_vocab_ids = model.map_extended_to_vocab_ids(infer_out["generated_ids"], scene_ids)
    metrics = computeBleuRouge(
        reference_ids=reference_ids,
        predicted_ids=predicted_vocab_ids,
        pad_token_id=cfg.pad_token_id,
        bos_token_id=cfg.bos_token_id,
        eos_token_id=cfg.eos_token_id,
    )
    printMetrics("MLPAG", metrics)


def run_qumlag(args: argparse.Namespace, device: torch.device) -> None:
    cfg = QuMLAGConfig(
        vocab_size=args.vocab_size,
        image_feature_dim=args.image_feature_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.heads,
        num_decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        max_question_len=args.question_len,
        max_answer_len=args.answer_len,
    )
    model = QuMLAG(cfg).to(device)
    model.train()

    question_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.question_len), device=device)
    image_feats = torch.randn(
        args.batch_size, args.num_regions, args.image_feature_dim, device=device
    )
    decoder_input_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.answer_len), device=device)
    train_out = model(
        question_token_ids=question_ids,
        image_region_features=image_feats,
        decoder_input_ids=decoder_input_ids,
    )
    train_logits = train_out["logits"]  # [B, T, V]
    train_target = torch.randint(0, cfg.vocab_size, (args.batch_size, args.answer_len), device=device)
    loss = F.cross_entropy(
        train_logits.reshape(-1, train_logits.size(-1)),
        train_target.reshape(-1),
    )
    print(f"[QuMLAG][train] logits shape: {tuple(train_logits.shape)}, loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        infer_out = model(
            question_token_ids=question_ids,
            image_region_features=image_feats,
            max_decode_len=args.answer_len,
        )
    print(
        f"[QuMLAG][infer] generated_ids shape: {tuple(infer_out['generated_ids'].shape)}, "
        f"logits shape: {tuple(infer_out['logits'].shape)}"
    )
    reference_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.answer_len), device=device)
    metrics = computeBleuRouge(
        reference_ids=reference_ids,
        predicted_ids=infer_out["generated_ids"],
        pad_token_id=cfg.pad_token_id,
        bos_token_id=cfg.bos_token_id,
        eos_token_id=cfg.eos_token_id,
    )
    printMetrics("QuMLAG", metrics)


def run_m4c(args: argparse.Namespace, device: torch.device) -> None:
    cfg = M4CConfig(
        vocab_size=args.vocab_size,
        d_model=768,
        object_feat_dim=args.image_feature_dim,
        num_heads=args.heads,
        num_mmt_layers=4,
        num_question_layers=4,
        max_answer_len=args.answer_len,
        dropout=args.dropout,
        pretrained_bert=False,
    )
    model = M4C(cfg).to(device)
    model.train()

    batch_size = args.batch_size
    num_obj = args.num_regions
    num_ocr = args.scene_len
    question_ids = torch.randint(0, 30522, (batch_size, args.question_len), device=device)
    obj_features = torch.randn(batch_size, num_obj, cfg.object_feat_dim, device=device)
    obj_boxes = torch.rand(batch_size, num_obj, 4, device=device)
    ocr_det = torch.randn(batch_size, num_ocr, cfg.ocr_det_feat_dim, device=device)
    ocr_rec = torch.randn(batch_size, num_ocr, cfg.ocr_rec_feat_dim, device=device)
    ocr_fasttext = torch.randn(batch_size, num_ocr, cfg.ocr_fasttext_dim, device=device)
    ocr_boxes = torch.rand(batch_size, num_ocr, 4, device=device)
    answer_prev_ids = torch.randint(0, cfg.vocab_size, (batch_size, args.answer_len), device=device)

    train_out = model(
        question_token_ids=question_ids,
        obj_features=obj_features,
        obj_boxes=obj_boxes,
        ocr_det_features=ocr_det,
        ocr_rec_features=ocr_rec,
        ocr_fasttext_features=ocr_fasttext,
        ocr_boxes=ocr_boxes,
        answer_prev_ids=answer_prev_ids,
    )
    train_scores = train_out["scores"]  # [B, T, V+O]
    train_target = torch.randint(
        0, cfg.vocab_size + num_ocr, (batch_size, args.answer_len), device=device
    )
    loss = F.cross_entropy(
        train_scores.reshape(-1, train_scores.size(-1)),
        train_target.reshape(-1),
    )
    print(f"[M4C][train] scores shape: {tuple(train_scores.shape)}, loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        infer_out = model(
            question_token_ids=question_ids,
            obj_features=obj_features,
            obj_boxes=obj_boxes,
            ocr_det_features=ocr_det,
            ocr_rec_features=ocr_rec,
            ocr_fasttext_features=ocr_fasttext,
            ocr_boxes=ocr_boxes,
            max_decode_len=args.answer_len,
        )
    print(
        f"[M4C][infer] generated_ids shape: {tuple(infer_out['generated_ids'].shape)}, "
        f"scores shape: {tuple(infer_out['scores'].shape)}"
    )
    reference_ids = torch.randint(0, cfg.vocab_size, (batch_size, args.answer_len), device=device)
    metrics = computeBleuRouge(
        reference_ids=reference_ids,
        predicted_ids=infer_out["generated_ids"],
        pad_token_id=cfg.pad_token_id,
        bos_token_id=cfg.bos_token_id,
        eos_token_id=cfg.eos_token_id,
    )
    printMetrics("M4C", metrics)


def main() -> None:
    parser = build_main_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    if args.model in ("mlpag", "all"):
        run_mlpag(args, device)
    if args.model in ("qumlag", "all"):
        run_qumlag(args, device)
    if args.model in ("m4c", "all"):
        run_m4c(args, device)


if __name__ == "__main__":
    main()
