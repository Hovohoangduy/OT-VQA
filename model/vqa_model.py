"""Optimal-transport VQA with autoregressive answer generation."""

from __future__ import annotations

import torch
from torch import nn

from configs.config import Config
from model.decoder_model import Decoder
from model.features_extraction import AnswerEmbedding, ImageEmbedding, QuestionEmbedding, validate_english_text_model
from model.optimal_transport import PartialTransportFusion
from utils.answer_text import postprocess_answer, restore_numeric_spacing
from utils.image_processing import build_image_transform


class TiedOutputProjection(nn.Module):
    """Output projection head tied to answer token embeddings to regularize vocabulary prediction."""

    def __init__(self, d_model, word_embeddings, drop_prob=0.3):
        super().__init__()
        self.word_embeddings = word_embeddings
        emb_dim = word_embeddings.embedding_dim
        self.proj = nn.Sequential(
            nn.Dropout(p=drop_prob),
            nn.Linear(d_model, emb_dim),
            nn.GELU(),
        )
        self.bias = nn.Parameter(torch.zeros(word_embeddings.num_embeddings))

    def forward(self, x):
        hidden = self.proj(x)
        return torch.nn.functional.linear(hidden, self.word_embeddings.weight, self.bias)


class VQAModel(nn.Module):
    """Generate answers from question-conditioned image features."""

    def __init__(self, vocab_size=None, output_size=None, d_model=768, num_heads=4,
                 ffn_hidden=2048, drop_prob=0.1, num_layers=4,
                 mode='train', text_model=Config.text_model, image_model=Config.image_model,
                 freeze_answer_embeddings=False, freeze_question_encoder=False, ot_epsilon=0.05,
                 ot_iterations=20, ot_dustbin_mass=0.2, ot_dustbin_cost=1.0,
                 max_answer_tokens=Config.MAX_LEN_ANS,
                  max_question_tokens=Config.MAX_LEN_QUES, fusion='ot',
                  image_size=256, image_preprocessing='crop', ot_visual_memory=False,
                  figure_context=False, tie_weights=False, ot_num_heads=1,
                  fgw_alpha=0.0, unbalanced_tau=None):
        super().__init__()
        output_size = d_model if output_size is None else output_size
        if output_size != d_model:
            raise ValueError('output_size must equal d_model')
        if max_answer_tokens < 2:
            raise ValueError('max_answer_tokens must be at least 2')
        if max_question_tokens < 2:
            raise ValueError('max_question_tokens must be at least 2')
        if fusion != 'ot':
            raise ValueError('Only OT fusion is supported')
        if d_model % num_heads:
            raise ValueError('d_model must be divisible by num_heads')
        validate_english_text_model(text_model)
        self.mode = mode
        self.fusion = fusion
        self.max_answer_tokens = max_answer_tokens
        self.max_question_tokens = max_question_tokens
        self.text_model_name = str(text_model)
        self.image_model_name = str(image_model)
        self.image_size = image_size
        self.image_preprocessing = image_preprocessing
        self.figure_context = figure_context
        self.tie_weights = tie_weights
        self.ot_num_heads = ot_num_heads
        self.fgw_alpha = float(fgw_alpha)
        self.unbalanced_tau = float(unbalanced_tau) if unbalanced_tau is not None else None
        self.image_transform = build_image_transform(image_size, image_preprocessing)
        self.model_config = dict(
            output_size=output_size, d_model=d_model, num_heads=num_heads,
            ffn_hidden=ffn_hidden, drop_prob=drop_prob, num_layers=num_layers,
            freeze_answer_embeddings=freeze_answer_embeddings,
            freeze_question_encoder=freeze_question_encoder,
            fusion=fusion,
            ot_epsilon=ot_epsilon, ot_iterations=ot_iterations,
            ot_dustbin_mass=ot_dustbin_mass, ot_dustbin_cost=ot_dustbin_cost,
            max_answer_tokens=max_answer_tokens,
            max_question_tokens=max_question_tokens,
            image_size=image_size, image_preprocessing=image_preprocessing,
            ot_visual_memory=ot_visual_memory,
            figure_context=figure_context,
            tie_weights=tie_weights,
            ot_num_heads=ot_num_heads,
            fgw_alpha=fgw_alpha,
            unbalanced_tau=unbalanced_tau,
        )
        self.image_model = ImageEmbedding(image_model, image_size=image_size)
        self.question_encoder = QuestionEmbedding(model_name=text_model,
                                                  max_length=max_question_tokens)
        if freeze_question_encoder:
            self.question_encoder.freeze()
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
        self._setup_constrained_tokens()

        image_dim = self.image_model.output_dim
        answer_dim = self.answer_embedding.token_embeddings.word_embeddings.embedding_dim
        self.image_projection = nn.Identity() if image_dim == d_model else nn.Linear(image_dim, d_model)
        self.answer_projection = nn.Identity() if answer_dim == d_model else nn.Linear(answer_dim, d_model)
        text_dim = self.question_encoder.text_encoder.config.hidden_size
        self.ot_fusion = PartialTransportFusion(
            text_dim=text_dim, d_model=d_model, epsilon=ot_epsilon,
            iterations=ot_iterations, dustbin_mass=ot_dustbin_mass,
            dustbin_cost=ot_dustbin_cost,
            visual_memory=ot_visual_memory,
            num_heads=ot_num_heads,
            fgw_alpha=fgw_alpha,
            unbalanced_tau=unbalanced_tau,
        )
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
        actual_vocab = self.answer_embedding.token_embeddings.word_embeddings.num_embeddings
        if vocab_size is not None and vocab_size != actual_vocab:
            raise ValueError('vocab_size must match the text model embedding vocabulary')
        if tie_weights:
            self.mlp = TiedOutputProjection(
                d_model, self.answer_embedding.token_embeddings.word_embeddings,
                drop_prob=max(drop_prob, 0.25),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Dropout(p=max(drop_prob, 0.2)), nn.Linear(d_model, d_model), nn.GELU(),
                nn.Linear(d_model, actual_vocab),
            )

    def encode(self, images, questions, anno_ids=None, return_transport=False):
        image_embeddings, _ = self.image_model(images, image_ids=anno_ids)
        projected_images = self.image_projection(image_embeddings)
        question_tokens, blocked, _, question_global = self.question_encoder.encode_tokens(questions)
        kwargs = {'return_transport': True} if return_transport else {}
        image_tokens = (projected_images[:, 1:] if self.image_model.has_cls_token
                        else projected_images)
        return self.ot_fusion(image_tokens, question_tokens, ~blocked,
                              question_global=question_global, **kwargs)

    def decode(self, input_ids, memory, causal=True, memory_blocked=None, last_token_only=False):
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
        cross_mask = (memory_blocked[:, None, None, :] if memory_blocked is not None
                      else None)
        hidden = self.decoder(memory, target, blocked, cross_mask)
        # Autoregressive generation consumes only the newest position. Avoid
        # projecting all previous positions into the 30k-token vocabulary again.
        return self.mlp(hidden[:, -1:] if last_token_only else hidden)

    def forward(self, images, questions, answers=None, anno_ids=None, mask=True,
                mode='train', max_len=None, return_ot_loss=False):
        if mode not in {'train', 'eval', 'test', 'infer', 'generate'}:
            raise ValueError(f'Unknown mode: {mode}')
        max_len = self.max_answer_tokens if max_len is None else max_len
        if mode in {'test', 'infer', 'generate'} or answers is None:
            return self.generate(images, questions, anno_ids, max_len)
        if return_ot_loss:
            memory, memory_blocked, diagnostics = self.encode(
                images, questions, anno_ids, return_transport=True
            )
            ot_loss = diagnostics.get("ot_loss")
        else:
            memory, memory_blocked = self.encode(images, questions, anno_ids)
            ot_loss = None
        ids = self.answer_embedding.tokenize(answers, max_len)
        logits = self.decode(ids[:, :-1], memory, causal=mask,
                             memory_blocked=memory_blocked)
        if return_ot_loss:
            return logits, ids[:, 1:], ot_loss
        return logits, ids[:, 1:]

    def _setup_constrained_tokens(self):
        """Pre-cache token IDs for task-adaptive constrained decoding."""
        tok = self.tokenizer
        yes_ids = tok.encode("yes", add_special_tokens=False)
        no_ids = tok.encode("no", add_special_tokens=False)
        self.yes_token_id = yes_ids[0] if yes_ids else 2748
        self.no_token_id = no_ids[0] if no_ids else 2053

        self.option_token_ids = {}
        for letter in ["A", "B", "C", "D", "E", "F"]:
            enc = tok.encode(letter, add_special_tokens=False)
            if enc:
                self.option_token_ids[letter] = enc[0]

        comma_ids = tok.encode(",", add_special_tokens=False)
        self.comma_token_id = comma_ids[0] if comma_ids else 1010

        self.numeric_token_ids = set()
        for ch in "0123456789.-+":
            enc = tok.encode(ch, add_special_tokens=False)
            for tid in enc:
                self.numeric_token_ids.add(tid)

    def _apply_step_constraints(self, logits, ids, step, questions):
        """Apply task-adaptive token masking based on question structure."""
        if questions is None or len(questions) != logits.size(0):
            return logits

        for b, q in enumerate(questions):
            q_str = str(q).lower()

            # 1. Binary question: "Yes" or "No"
            is_binary = "answer with 'yes' or 'no'" in q_str or "yes or no" in q_str
            if is_binary:
                if step == 0:
                    allowed = [self.yes_token_id, self.no_token_id]
                    mask = torch.ones(logits.size(1), dtype=torch.bool, device=logits.device)
                    mask[allowed] = False
                    logits[b, mask] = float("-inf")
                else:
                    mask = torch.ones(logits.size(1), dtype=torch.bool, device=logits.device)
                    mask[self.eos_token_id] = False
                    logits[b, mask] = float("-inf")
                continue

            # 2. Multiple choice question: options present or explicitly instructed
            is_mc = "return only the letter(s)" in q_str or "answer options:" in q_str
            if is_mc:
                candidate_ids = []
                for letter, tid in self.option_token_ids.items():
                    if f"{letter.lower()}:" in q_str or f"({letter.lower()})" in q_str or f"option {letter.lower()}" in q_str:
                        candidate_ids.append(tid)
                if not candidate_ids:
                    candidate_ids = [self.option_token_ids[l] for l in ["A", "B", "C", "D"] if l in self.option_token_ids]

                if step == 0:
                    mask = torch.ones(logits.size(1), dtype=torch.bool, device=logits.device)
                    mask[candidate_ids] = False
                    logits[b, mask] = float("-inf")
                else:
                    last_token = ids[b, -1].item()
                    mask = torch.ones(logits.size(1), dtype=torch.bool, device=logits.device)
                    already_selected = set(ids[b, 1:].tolist())
                    remaining_candidates = [tid for tid in candidate_ids if tid not in already_selected]
                    multi_choice_cue = "all that apply" in q_str or "letter(s)" in q_str or "options" in q_str
                    if last_token == self.comma_token_id:
                        if remaining_candidates:
                            mask[remaining_candidates] = False
                        else:
                            mask[self.eos_token_id] = False
                    elif last_token in candidate_ids:
                        allowed = [self.eos_token_id]
                        if multi_choice_cue and remaining_candidates and step < 6:
                            allowed.append(self.comma_token_id)
                        mask[allowed] = False
                    else:
                        mask[self.eos_token_id] = False
                    logits[b, mask] = float("-inf")
                continue

            # 3. Numeric question: raw numerical value instructed
            is_numeric = "answer with only the raw numerical value" in q_str or "raw numerical value" in q_str
            if is_numeric and self.numeric_token_ids:
                allowed = list(self.numeric_token_ids)
                if step > 0:
                    allowed.append(self.eos_token_id)
                mask = torch.ones(logits.size(1), dtype=torch.bool, device=logits.device)
                mask[allowed] = False
                logits[b, mask] = float("-inf")
                continue

        return logits

    @torch.no_grad()
    def evaluate_batch(self, images, questions, answers, anno_ids=None,
                       max_len=None, constrained=True):
        """Compute teacher-forced logits and generated answers from one encoding."""
        max_len = self.max_answer_tokens if max_len is None else max_len
        memory, memory_blocked = self.encode(images, questions, anno_ids)
        ids = self.answer_embedding.tokenize(answers, max_len)
        logits = self.decode(ids[:, :-1], memory, memory_blocked=memory_blocked)
        generated = self._generate_from_memory(memory, memory_blocked, max_len,
                                               questions=questions, constrained=constrained)
        return logits, ids[:, 1:], generated

    def _generate_from_memory(self, memory, memory_blocked, max_len, questions=None, constrained=True):
        ids = torch.full((memory.size(0), 1), self.bos_token_id, dtype=torch.long,
                         device=memory.device)
        finished = torch.zeros(memory.size(0), dtype=torch.bool, device=memory.device)
        for step in range(max_len - 1):
            logits = self.decode(ids, memory, memory_blocked=memory_blocked,
                                 last_token_only=True)[:, -1, :]
            logits[:, [self.pad_token_id, self.bos_token_id]] = float('-inf')
            if constrained and questions is not None:
                logits = self._apply_step_constraints(logits, ids, step, questions)
            next_ids = logits.argmax(-1).masked_fill(finished, self.pad_token_id)
            ids = torch.cat([ids, next_ids.unsqueeze(1)], dim=1)
            finished |= next_ids.eq(self.eos_token_id)
            if bool(finished.all()):
                break
        return ids[:, 1:]

    @torch.no_grad()
    def generate(self, images, questions, anno_ids=None, max_len=None, constrained=True):
        max_len = self.max_answer_tokens if max_len is None else max_len
        if not 2 <= max_len <= self.max_answer_tokens:
            raise ValueError(f'max_len must be between 2 and {self.max_answer_tokens}')
        was_training = self.training
        self.eval()
        try:
            memory, memory_blocked = self.encode(images, questions, anno_ids)
            return self._generate_from_memory(memory, memory_blocked, max_len,
                                               questions=questions, constrained=constrained)
        finally:
            self.train(was_training)

    def answers_from_ids(self, ids):
        rows = []
        for row in ids.detach().cpu().tolist():
            if self.eos_token_id in row:
                row = row[:row.index(self.eos_token_id)]
            rows.append(row)
        decoded = self.tokenizer.batch_decode(rows, skip_special_tokens=True)
        return [postprocess_answer(answer) for answer in decoded]
