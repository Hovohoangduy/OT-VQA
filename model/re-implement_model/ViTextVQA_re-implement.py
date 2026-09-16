import argparse
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TextVQAConfig:
    vocab_size: int = 64001
    visual_feature_dim: int = 2048
    token_embed_dim: int = 300
    hidden_dim: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    ff_dim: int = 2048
    dropout: float = 0.1
    max_question_len: int = 32
    max_answer_len: int = 16
    max_ocr_tokens: int = 24
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


class TokenProjector(nn.Module):
    def __init__(self, vocab_size: int, token_embed_dim: int, hidden_dim: int, pad_token_id: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, token_embed_dim, padding_idx=pad_token_id)
        self.proj = nn.Linear(token_embed_dim, hidden_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embedding(token_ids))


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, hidden_dim: int):
        super().__init__()
        self.position = nn.Embedding(max_len, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        pos_ids = torch.arange(t, device=x.device).unsqueeze(0).expand(b, -1)
        return x + self.position(pos_ids)


class TransformerTextVQABase(nn.Module):
    def __init__(self, config: TextVQAConfig):
        super().__init__()
        self.config = config

        self.question_projector = TokenProjector(
            config.vocab_size, config.token_embed_dim, config.hidden_dim, config.pad_token_id
        )
        self.ocr_projector = TokenProjector(
            config.vocab_size, config.token_embed_dim, config.hidden_dim, config.pad_token_id
        )
        self.answer_projector = TokenProjector(
            config.vocab_size, config.token_embed_dim, config.hidden_dim, config.pad_token_id
        )

        self.visual_proj = nn.Linear(config.visual_feature_dim, config.hidden_dim)
        self.ocr_box_proj = nn.Linear(4, config.hidden_dim)

        self.question_pos = LearnedPositionalEncoding(config.max_question_len, config.hidden_dim)
        self.ocr_pos = LearnedPositionalEncoding(config.max_ocr_tokens, config.hidden_dim)
        self.answer_pos = LearnedPositionalEncoding(config.max_answer_len, config.hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=config.num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=config.num_decoder_layers)
        self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size)

    def buildCausalMask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def decodeAnswer(self, answer_input_ids: torch.Tensor, memory: torch.Tensor, memory_pad: torch.Tensor):
        answer_tokens = self.answer_projector(answer_input_ids)
        answer_tokens = self.answer_pos(answer_tokens)
        tgt_pad = answer_input_ids.eq(self.config.pad_token_id)
        causal = self.buildCausalMask(answer_tokens.size(1), answer_tokens.device)
        decoded = self.decoder(
            tgt=answer_tokens,
            memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=memory_pad,
        )
        return self.output_proj(decoded)

    def greedyGenerate(self, memory: torch.Tensor, memory_pad: torch.Tensor, max_decode_len: int) -> torch.Tensor:
        batch_size = memory.size(0)
        generated = torch.full(
            (batch_size, 1),
            fill_value=self.config.bos_token_id,
            dtype=torch.long,
            device=memory.device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=memory.device)
        for _ in range(max_decode_len):
            logits = self.decodeAnswer(generated, memory, memory_pad)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | next_token.squeeze(1).eq(self.config.eos_token_id)
            if finished.all():
                break
        return generated[:, 1:]


class PreSTUModel(TransformerTextVQABase):
    """
    PreSTU-inspired implementation:
    - OCR tokens are reordered by their 2D layout before multimodal encoding.
    - Uses Transformer encoder/decoder pipeline in the spirit of a pre-trained text-centric LM.
    """

    def reorderOCRByLayout(
        self,
        ocr_token_ids: torch.Tensor,
        ocr_boxes: torch.Tensor,
        ocr_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centers = 0.5 * (ocr_boxes[..., :2] + ocr_boxes[..., 2:4])  # [B, O, 2]
        # Top-to-bottom, then left-to-right sorting key.
        sort_key = centers[..., 1] * 10.0 + centers[..., 0]
        sort_key = sort_key.masked_fill(ocr_pad, 1e6)
        sort_index = torch.argsort(sort_key, dim=1)

        sorted_ids = ocr_token_ids.gather(1, sort_index)
        sorted_pad = ocr_pad.gather(1, sort_index)
        sorted_boxes = ocr_boxes.gather(1, sort_index.unsqueeze(-1).expand(-1, -1, 4))
        return sorted_ids, sorted_boxes, sorted_pad

    def encodeMultimodal(
        self,
        visual_features: torch.Tensor,
        question_ids: torch.Tensor,
        ocr_token_ids: torch.Tensor,
        ocr_boxes: torch.Tensor,
        visual_pad: Optional[torch.Tensor] = None,
        question_pad: Optional[torch.Tensor] = None,
        ocr_pad: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if visual_pad is None:
            visual_pad = visual_features.abs().sum(dim=-1).eq(0)
        if question_pad is None:
            question_pad = question_ids.eq(self.config.pad_token_id)
        if ocr_pad is None:
            ocr_pad = ocr_token_ids.eq(self.config.pad_token_id)

        ocr_ids, ocr_boxes, ocr_pad = self.reorderOCRByLayout(ocr_token_ids, ocr_boxes, ocr_pad)
        visual_tokens = self.visual_proj(visual_features)
        question_tokens = self.question_pos(self.question_projector(question_ids))
        ocr_tokens = self.ocr_pos(self.ocr_projector(ocr_ids) + self.ocr_box_proj(ocr_boxes))

        memory = torch.cat([visual_tokens, question_tokens, ocr_tokens], dim=1)
        memory_pad = torch.cat([visual_pad, question_pad, ocr_pad], dim=1)
        encoded = self.encoder(memory, src_key_padding_mask=memory_pad)
        return encoded, memory_pad

    def forward(
        self,
        visual_features: torch.Tensor,
        question_ids: torch.Tensor,
        ocr_token_ids: torch.Tensor,
        ocr_boxes: torch.Tensor,
        answer_input_ids: Optional[torch.Tensor] = None,
        max_decode_len: Optional[int] = None,
    ):
        memory, memory_pad = self.encodeMultimodal(
            visual_features=visual_features,
            question_ids=question_ids,
            ocr_token_ids=ocr_token_ids,
            ocr_boxes=ocr_boxes,
        )
        if answer_input_ids is not None:
            logits = self.decodeAnswer(answer_input_ids, memory, memory_pad)
            return {"logits": logits}

        decode_len = max_decode_len or self.config.max_answer_len
        generated_ids = self.greedyGenerate(memory, memory_pad, decode_len)
        return {"generated_ids": generated_ids}


class TextSemanticSeparation(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.question_gate = nn.Linear(hidden_dim, hidden_dim)
        self.ocr_norm = nn.LayerNorm(hidden_dim)

    def forward(self, ocr_tokens: torch.Tensor, question_context: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.question_gate(question_context)).unsqueeze(1)
        semantic = self.ocr_norm(ocr_tokens * gate + ocr_tokens)
        return semantic


class SpatialCirclePosition(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(3, hidden_dim)

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        centers = 0.5 * (boxes[..., :2] + boxes[..., 2:4])  # normalized x,y in [0,1]
        shifted = centers - 0.5
        radius = torch.sqrt((shifted[..., 0] ** 2 + shifted[..., 1] ** 2).clamp_min(1e-8))
        angle = torch.atan2(shifted[..., 1], shifted[..., 0]) / 3.14159265  # scale to [-1,1]
        circle = torch.stack([shifted[..., 0], shifted[..., 1], radius + angle], dim=-1)
        return self.proj(circle)


class SaLModel(TransformerTextVQABase):
    """
    SaL-inspired implementation:
    - Text Semantic Separation module conditions OCR semantics on question context.
    - Spatial Circle Position module injects circular positional cues from OCR boxes.
    """

    def __init__(self, config: TextVQAConfig):
        super().__init__(config)
        self.textSemanticSeparation = TextSemanticSeparation(config.hidden_dim)
        self.spatialCirclePosition = SpatialCirclePosition(config.hidden_dim)

    def maskedMean(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        valid = (~pad_mask).to(x.dtype).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (x * valid).sum(dim=1) / denom

    def encodeMultimodal(
        self,
        visual_features: torch.Tensor,
        question_ids: torch.Tensor,
        ocr_token_ids: torch.Tensor,
        ocr_boxes: torch.Tensor,
        visual_pad: Optional[torch.Tensor] = None,
        question_pad: Optional[torch.Tensor] = None,
        ocr_pad: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if visual_pad is None:
            visual_pad = visual_features.abs().sum(dim=-1).eq(0)
        if question_pad is None:
            question_pad = question_ids.eq(self.config.pad_token_id)
        if ocr_pad is None:
            ocr_pad = ocr_token_ids.eq(self.config.pad_token_id)

        visual_tokens = self.visual_proj(visual_features)
        question_tokens = self.question_pos(self.question_projector(question_ids))
        ocr_tokens = self.ocr_projector(ocr_token_ids) + self.ocr_box_proj(ocr_boxes)

        question_context = self.maskedMean(question_tokens, question_pad)
        semantic_tokens = self.textSemanticSeparation(ocr_tokens, question_context)
        spatial_tokens = self.spatialCirclePosition(ocr_boxes)
        ocr_tokens = self.ocr_pos(semantic_tokens + spatial_tokens)

        memory = torch.cat([visual_tokens, question_tokens, ocr_tokens], dim=1)
        memory_pad = torch.cat([visual_pad, question_pad, ocr_pad], dim=1)
        encoded = self.encoder(memory, src_key_padding_mask=memory_pad)
        return encoded, memory_pad

    def forward(
        self,
        visual_features: torch.Tensor,
        question_ids: torch.Tensor,
        ocr_token_ids: torch.Tensor,
        ocr_boxes: torch.Tensor,
        answer_input_ids: Optional[torch.Tensor] = None,
        max_decode_len: Optional[int] = None,
    ):
        memory, memory_pad = self.encodeMultimodal(
            visual_features=visual_features,
            question_ids=question_ids,
            ocr_token_ids=ocr_token_ids,
            ocr_boxes=ocr_boxes,
        )
        if answer_input_ids is not None:
            logits = self.decodeAnswer(answer_input_ids, memory, memory_pad)
            return {"logits": logits}

        decode_len = max_decode_len or self.config.max_answer_len
        generated_ids = self.greedyGenerate(memory, memory_pad, decode_len)
        return {"generated_ids": generated_ids}


def normalizeText(text: str) -> str:
    return " ".join(text.lower().strip().split())


def idsToText(
    ids: torch.Tensor,
    id_to_token: dict[int, str],
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
) -> list[str]:
    lines = []
    for row in ids.detach().cpu().tolist():
        tokens = []
        for token_id in row:
            if token_id == bos_token_id:
                continue
            if token_id == eos_token_id:
                break
            if token_id == pad_token_id:
                continue
            tokens.append(id_to_token.get(token_id, "<unk>"))
        lines.append(normalizeText(" ".join(tokens)))
    return lines


def computeEmAndF1(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    em_total = 0.0
    f1_total = 0.0
    for ref, hyp in zip(references, hypotheses):
        if ref == hyp:
            em_total += 1.0

        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        if not ref_tokens and not hyp_tokens:
            f1_total += 1.0
            continue
        if not ref_tokens or not hyp_tokens:
            continue

        ref_counter = {}
        hyp_counter = {}
        for t in ref_tokens:
            ref_counter[t] = ref_counter.get(t, 0) + 1
        for t in hyp_tokens:
            hyp_counter[t] = hyp_counter.get(t, 0) + 1

        common = 0
        for t, c in hyp_counter.items():
            common += min(c, ref_counter.get(t, 0))

        precision = common / max(len(hyp_tokens), 1)
        recall = common / max(len(ref_tokens), 1)
        if precision + recall > 0:
            f1_total += 2 * precision * recall / (precision + recall)

    n = max(len(references), 1)
    return {"EM": em_total / n, "F1": f1_total / n}


def evaluateModel(
    model_name: str,
    model: nn.Module,
    visual_features: torch.Tensor,
    question_ids: torch.Tensor,
    ocr_token_ids: torch.Tensor,
    ocr_boxes: torch.Tensor,
    answer_input_ids: torch.Tensor,
    reference_answer_ids: torch.Tensor,
    id_to_token: dict[int, str],
    config: TextVQAConfig,
) -> None:
    model.train()
    train_out = model(
        visual_features=visual_features,
        question_ids=question_ids,
        ocr_token_ids=ocr_token_ids,
        ocr_boxes=ocr_boxes,
        answer_input_ids=answer_input_ids,
    )
    logits = train_out["logits"]
    train_target = torch.randint(
        0, config.vocab_size, (logits.size(0), logits.size(1)), device=logits.device
    )
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), train_target.reshape(-1))

    model.eval()
    with torch.no_grad():
        infer_out = model(
            visual_features=visual_features,
            question_ids=question_ids,
            ocr_token_ids=ocr_token_ids,
            ocr_boxes=ocr_boxes,
            max_decode_len=config.max_answer_len,
        )
    pred_ids = infer_out["generated_ids"]
    refs = idsToText(
        reference_answer_ids,
        id_to_token,
        config.pad_token_id,
        config.bos_token_id,
        config.eos_token_id,
    )
    hyps = idsToText(
        pred_ids,
        id_to_token,
        config.pad_token_id,
        config.bos_token_id,
        config.eos_token_id,
    )
    metrics = computeEmAndF1(refs, hyps)

    print(
        f"[{model_name}] train_logits={tuple(logits.shape)}, loss={loss.item():.4f}, "
        f"EM={metrics['EM']:.4f}, F1={metrics['F1']:.4f}"
    )


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SaL and PreSTU-inspired TextVQA implementation.")
    parser.add_argument("--model", type=str, default="all", choices=["sal", "prestu", "all"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-regions", type=int, default=16)
    parser.add_argument("--question-len", type=int, default=14)
    parser.add_argument("--answer-len", type=int, default=10)
    parser.add_argument("--ocr-len", type=int, default=24)
    parser.add_argument("--vocab-size", type=int, default=64001)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    config = TextVQAConfig(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        max_question_len=args.question_len,
        max_answer_len=args.answer_len,
        max_ocr_tokens=args.ocr_len,
    )

    visual_features = torch.randn(
        args.batch_size, args.num_regions, config.visual_feature_dim, device=device
    )
    question_ids = torch.randint(0, config.vocab_size, (args.batch_size, args.question_len), device=device)
    ocr_token_ids = torch.randint(0, config.vocab_size, (args.batch_size, args.ocr_len), device=device)
    ocr_boxes = torch.rand(args.batch_size, args.ocr_len, 4, device=device)
    # Ensure x1<=x2 and y1<=y2
    x1y1 = torch.minimum(ocr_boxes[..., :2], ocr_boxes[..., 2:4])
    x2y2 = torch.maximum(ocr_boxes[..., :2], ocr_boxes[..., 2:4])
    ocr_boxes = torch.cat([x1y1, x2y2], dim=-1)

    reference_answer_ids = torch.randint(
        3, config.vocab_size, (args.batch_size, args.answer_len), device=device
    )
    answer_input_ids = torch.cat(
        [
            torch.full((args.batch_size, 1), config.bos_token_id, dtype=torch.long, device=device),
            reference_answer_ids[:, :-1],
        ],
        dim=1,
    )

    id_to_token = {i: f"tok{i}" for i in range(config.vocab_size)}
    id_to_token[config.pad_token_id] = "<pad>"
    id_to_token[config.bos_token_id] = "<bos>"
    id_to_token[config.eos_token_id] = "<eos>"

    if args.model in ("prestu", "all"):
        prestu = PreSTUModel(config).to(device)
        evaluateModel(
            model_name="PreSTU",
            model=prestu,
            visual_features=visual_features,
            question_ids=question_ids,
            ocr_token_ids=ocr_token_ids,
            ocr_boxes=ocr_boxes,
            answer_input_ids=answer_input_ids,
            reference_answer_ids=reference_answer_ids,
            id_to_token=id_to_token,
            config=config,
        )

    if args.model in ("sal", "all"):
        sal = SaLModel(config).to(device)
        evaluateModel(
            model_name="SaL",
            model=sal,
            visual_features=visual_features,
            question_ids=question_ids,
            ocr_token_ids=ocr_token_ids,
            ocr_boxes=ocr_boxes,
            answer_input_ids=answer_input_ids,
            reference_answer_ids=reference_answer_ids,
            id_to_token=id_to_token,
            config=config,
        )


def main() -> None:
    parser = buildParser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

