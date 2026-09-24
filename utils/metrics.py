"""Generated-answer metrics from the PlantExpertVQA evaluation section.

Scores are fractions (0 to 1 for lexical metrics), not percentages. BERTScore
can be negative when baseline rescaling is enabled. The paper gives formulas
but not its tokenizer or BERTScore encoder; those choices are explicit here.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from statistics import fmean


PAPER_METRICS = ("em", "token_f1", "bleu_1", "bleu_2", "rouge_l", "bertscore_f1")
_ARTICLE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WORD_PUNCT = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def normalize_text(text):
    """Light normalization retained for dataset diagnostics."""
    return " ".join(str(text).casefold().strip().split())


def _as_text(value):
    return value if isinstance(value, str) else " ".join(value)


def _squad_tokens(text):
    text = _as_text(text).lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = _ARTICLE.sub(" ", text)
    return text.split()


def _word_tokens(text):
    return _WORD_PUNCT.findall(_as_text(text).casefold())


def _token_f1(reference, hypothesis):
    ref, hyp = _squad_tokens(reference), _squad_tokens(hypothesis)
    if not ref or not hyp:
        return float(ref == hyp)
    common = sum((Counter(ref) & Counter(hyp)).values())
    return 2 * common / (len(ref) + len(hyp))


def _bleu(reference, hypothesis, order):
    ref, hyp = _word_tokens(reference), _word_tokens(hypothesis)
    if not ref or not hyp:
        return float(ref == hyp)
    precisions = []
    for n in range(1, order + 1):
        ref_ngrams = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
        hyp_ngrams = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
        if not hyp_ngrams:
            return 0.0
        matches = sum((ref_ngrams & hyp_ngrams).values())
        if matches == 0:
            return 0.0
        precisions.append(matches / sum(hyp_ngrams.values()))
    brevity_penalty = min(1.0, math.exp(1 - len(ref) / len(hyp)))
    return brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / order)


def _rouge_l(reference, hypothesis):
    ref, hyp = _word_tokens(reference), _word_tokens(hypothesis)
    if not ref or not hyp:
        return float(ref == hyp)
    previous = [0] * (len(hyp) + 1)
    for ref_token in ref:
        current = [0]
        for column, hyp_token in enumerate(hyp, 1):
            current.append(previous[column - 1] + 1 if ref_token == hyp_token
                           else max(previous[column], current[-1]))
        previous = current
    return 2 * previous[-1] / (len(ref) + len(hyp))


def lexical_scores(reference, hypothesis):
    """One example's EM, SQuAD-style Token-F1, BLEU-1/2 and ROUGE-L F1."""
    ref, hyp = _as_text(reference), _as_text(hypothesis)
    return {
        "em": float(ref.casefold() == hyp.casefold()),
        "token_f1": _token_f1(ref, hyp),
        "bleu_1": _bleu(ref, hyp, 1),
        "bleu_2": _bleu(ref, hyp, 2),
        "rouge_l": _rouge_l(ref, hyp),
    }


def compute_em_and_f1(references, hypotheses):
    """Compatibility wrapper for lightweight training and diagnostic metrics."""
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have equal lengths")
    if not references:
        return 0.0, 0.0
    rows = [lexical_scores(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    return fmean(row["em"] for row in rows), fmean(row["token_f1"] for row in rows)


def build_bertscore_scorer(model_type="bert-base-uncased", device="cpu", batch_size=16,
                           rescale_with_baseline=True):
    """Create a reusable frozen semantic scorer with explicit settings."""
    try:
        from bert_score import BERTScorer
    except ImportError as exc:
        raise RuntimeError("BERTScore requires `pip install bert-score` or `pip install -r requirements.txt`") from exc
    return BERTScorer(
        model_type=model_type, lang="en", device=device, batch_size=batch_size,
        rescale_with_baseline=rescale_with_baseline,
    )


def score_pairs(references, hypotheses, bert_scorer):
    """Score aligned generated answers once; return per-example metric rows."""
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have equal lengths")
    if bert_scorer is None:
        raise ValueError("A BERTScore scorer is required for the six paper metrics")
    rows = [lexical_scores(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    nonempty = [i for i, (ref, hyp) in enumerate(zip(references, hypotheses))
                if _as_text(ref).strip() and _as_text(hyp).strip()]
    nonempty_indices = set(nonempty)
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        if i not in nonempty_indices:
            rows[i]["bertscore_f1"] = float(not _as_text(ref).strip() and
                                           not _as_text(hyp).strip())
    if nonempty:
        _, _, f1 = bert_scorer.score([_as_text(hypotheses[i]) for i in nonempty],
                                     [_as_text(references[i]) for i in nonempty])
        values = f1.detach().cpu().tolist() if hasattr(f1, "detach") else list(f1)
        if len(values) != len(nonempty):
            raise RuntimeError("BERTScore returned the wrong number of examples")
        for i, value in zip(nonempty, values):
            rows[i]["bertscore_f1"] = float(value)
    return rows


def mean_scores(rows):
    if not rows:
        raise ValueError("Cannot aggregate an empty evaluation set")
    return {metric: fmean(row[metric] for row in rows) for metric in PAPER_METRICS}
