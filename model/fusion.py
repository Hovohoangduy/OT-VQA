"""Question-conditioned visual fusion baselines with a shared decoder interface.

All modules take spatial image tokens, question tokens and a valid-content-token mask.
They return decoder memory and a boolean mask where True means blocked.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


FUSION_METHODS = ("ot", "san", "ban", "cross_attention", "qformer")


def _check_inputs(images, questions, valid):
    if images.ndim != 3 or questions.ndim != 3 or valid.shape != questions.shape[:2]:
        raise ValueError("Expected image [B,K,D], question [B,L,T], mask [B,L]")
    if images.size(0) != questions.size(0) or images.size(1) == 0:
        raise ValueError("Batch sizes must match and images need patches")
    if not valid.bool().any(-1).all():
        raise ValueError("Every question needs a content token")


class _TextFusion(nn.Module):
    def __init__(self, text_dim, d_model, drop_prob):
        super().__init__()
        self.question_projection = nn.Linear(text_dim, d_model)
        self.text_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_prob)

    def _text(self, questions, valid):
        return self.text_norm(self.question_projection(questions)).masked_fill(
            ~valid.unsqueeze(-1), 0)

    @staticmethod
    def _memory(global_value, tokens, valid):
        memory = torch.cat((global_value.unsqueeze(1), tokens.masked_fill(
            ~valid.unsqueeze(-1), 0)), dim=1)
        blocked = torch.cat((torch.zeros_like(valid[:, :1]), ~valid), dim=1)
        return memory, blocked


class SANFusion(_TextFusion):
    """Stacked attention network: recurrent question-guided patch glimpses."""

    def __init__(self, text_dim, d_model, drop_prob=0.1, glimpses=2):
        super().__init__(text_dim, d_model, drop_prob)
        if glimpses < 1:
            raise ValueError("SAN glimpses must be positive")
        self.image_keys = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(glimpses))
        self.query_layers = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(glimpses))
        self.updates = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(glimpses))
        self.output = nn.LayerNorm(d_model)

    def forward(self, image_tokens, question_tokens, question_mask, question_global=None):
        _check_inputs(image_tokens, question_tokens, question_mask)
        valid = question_mask.bool()
        text = self._text(question_tokens, valid)
        query = (self.question_projection(question_global) if question_global is not None
                 else text.sum(1) / valid.sum(1, keepdim=True))
        for keys, query_layer, update in zip(self.image_keys, self.query_layers, self.updates):
            logits = torch.bmm(keys(image_tokens), query_layer(query).unsqueeze(-1)).squeeze(-1)
            weights = F.softmax(logits / image_tokens.size(-1) ** 0.5, dim=-1)
            glimpse = torch.bmm(weights.unsqueeze(1), image_tokens).squeeze(1)
            query = update(query + self.dropout(glimpse))
        grounded = self.output(text + self.dropout(query.unsqueeze(1)))
        return self._memory(query, grounded, valid)


class BANFusion(_TextFusion):
    """Low-rank bilinear question-patch attention with multiple glimpses."""

    def __init__(self, text_dim, d_model, drop_prob=0.1, glimpses=2):
        super().__init__(text_dim, d_model, drop_prob)
        if glimpses < 1:
            raise ValueError("BAN glimpses must be positive")
        self.image_factor = nn.Linear(d_model, d_model * glimpses)
        self.text_factor = nn.Linear(d_model, d_model * glimpses)
        self.glimpses = glimpses
        self.update = nn.Linear(d_model * glimpses, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, image_tokens, question_tokens, question_mask, question_global=None):
        _check_inputs(image_tokens, question_tokens, question_mask)
        valid = question_mask.bool()
        text = self._text(question_tokens, valid)
        batch, patches, width = image_tokens.shape
        length = text.size(1)
        image_factors = self.image_factor(image_tokens).reshape(batch, patches, self.glimpses, width)
        text_factors = self.text_factor(text).reshape(batch, length, self.glimpses, width)
        # Each glimpse scores every (question token, image patch) pair.
        scores = torch.einsum("bkgd,blgd->bglk", image_factors, text_factors) / width ** 0.5
        weights = F.softmax(scores, dim=-1)
        attended = torch.einsum("bglk,bkd->blgd", weights, image_tokens)
        grounded = self.norm(text + self.dropout(self.update(attended.reshape(batch, length, -1))))
        global_value = (grounded * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        return self._memory(global_value, grounded, valid)


class CrossAttentionFusion(_TextFusion):
    """Multihead question-to-image cross attention with residual text tokens."""

    def __init__(self, text_dim, d_model, num_heads=4, drop_prob=0.1):
        super().__init__(text_dim, d_model, drop_prob)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=drop_prob, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, image_tokens, question_tokens, question_mask, question_global=None):
        _check_inputs(image_tokens, question_tokens, question_mask)
        valid = question_mask.bool()
        text = self._text(question_tokens, valid)
        attended, _ = self.attention(text, image_tokens, image_tokens, need_weights=False)
        grounded = self.norm(text + self.dropout(attended))
        global_value = (grounded * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        return self._memory(global_value, grounded, valid)


class _QueryBlock(nn.Module):
    def __init__(self, d_model, num_heads, drop_prob):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=drop_prob, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=drop_prob, batch_first=True)
        self.self_norm = nn.LayerNorm(d_model)
        self.cross_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Dropout(drop_prob), nn.Linear(4 * d_model, d_model))
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, queries, text, image, text_blocked):
        combined = torch.cat((queries, text), dim=1)
        blocked = torch.cat((torch.zeros(queries.shape[:2], dtype=torch.bool,
                                        device=queries.device), text_blocked), dim=1)
        updated, _ = self.self_attention(combined, combined, combined,
                                          key_padding_mask=blocked, need_weights=False)
        combined = self.self_norm(combined + self.dropout(updated))
        queries = combined[:, :queries.size(1)]
        text = combined[:, queries.size(1):]
        attended, _ = self.cross_attention(queries, image, image, need_weights=False)
        queries = self.cross_norm(queries + self.dropout(attended))
        queries = self.ffn_norm(queries + self.dropout(self.ffn(queries)))
        return queries, text


class QFormerFusion(_TextFusion):
    """Learned queries that mix with question tokens and read image patches."""

    def __init__(self, text_dim, d_model, num_heads=4, drop_prob=0.1,
                 num_queries=8, num_layers=2):
        super().__init__(text_dim, d_model, drop_prob)
        if num_queries < 1 or num_layers < 1:
            raise ValueError("Q-Former queries and layers must be positive")
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        self.layers = nn.ModuleList(_QueryBlock(d_model, num_heads, drop_prob)
                                    for _ in range(num_layers))

    def forward(self, image_tokens, question_tokens, question_mask, question_global=None):
        _check_inputs(image_tokens, question_tokens, question_mask)
        valid = question_mask.bool()
        text = self._text(question_tokens, valid)
        queries = self.query_tokens.expand(image_tokens.size(0), -1, -1)
        for layer in self.layers:
            queries, text = layer(queries, text, image_tokens, ~valid)
        return queries, torch.zeros(queries.shape[:2], dtype=torch.bool, device=queries.device)
