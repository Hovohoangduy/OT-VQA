"""Dataset-specific generated-answer metrics for GQA, PlantExpertVQA and SciVQA.

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
GQA_METRICS = ("accuracy",)
SCIVQA_METRICS = (
    "rouge_1_f1", "rouge_1_precision", "rouge_1_recall",
    "rouge_l_f1", "rouge_l_precision", "rouge_l_recall",
    "bertscore_f1", "bertscore_precision", "bertscore_recall",
)
ALL_METRICS = tuple(dict.fromkeys((*GQA_METRICS, *PAPER_METRICS, *SCIVQA_METRICS)))
_ARTICLE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WORD_PUNCT = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def resolve_dataset(dataset="auto", csv_path=None):
    """Use an explicit dataset or detect GQA/SciVQA in a CSV path."""
    dataset = dataset.lower()
    if dataset not in {"auto", "gqa", "plantexpert", "scivqa"}:
        raise ValueError(f"Unknown dataset: {dataset}")
    if dataset != "auto":
        return dataset
    for name in ("scivqa", "gqa"):
        if csv_path is not None and re.search(
                rf"(?:^|[^a-z0-9]){name}(?:$|[^a-z0-9])", str(csv_path).lower()):
            return name
    return "plantexpert"


def metrics_for_dataset(dataset):
    return {"gqa": GQA_METRICS, "plantexpert": PAPER_METRICS,
            "scivqa": SCIVQA_METRICS}[resolve_dataset(dataset)]


def bertscore_defaults(dataset):
    """Explicit project defaults; the SciVQA paper does not specify its encoder."""
    return ("roberta-large", False) if resolve_dataset(dataset) == "scivqa" else (
        "bert-base-uncased", True)


def normalize_text(text):
    """Light normalization retained for dataset diagnostics."""
    return " ".join(str(text).casefold().strip().split())


def normalize_llm_answer(text: str) -> str:
    """Remove LLM whitespace/Markdown formatting while retaining answer content.

    Handle actual and literal escaped newlines/tabs without decoding arbitrary
    escapes or removing meaningful punctuation, numbers, Unicode, or case.
    """
    text = re.sub(r"\\(?:r\\n|[nrt])", "\n", text)
    text = re.sub(r"(?m)^[ \t]*(?:`{3,}|~{3,})[ \t]*(?:[\w+-]+)?[ \t]*$", "", text)
    text = re.sub(r"(?m)^[ \t]{0,3}(?:#{1,6}[ \t]+|>[ \t]?|[-+*][ \t]+)", "", text)
    for marker in ("**", "__", "~~", "```", "``", "`", "*", "_"):
        escaped = re.escape(marker)
        boundary = r"(?<!\w)" if marker in {"*", "_", "__"} else ""
        ending = r"(?!\w)" if marker in {"*", "_", "__"} else ""
        text = re.sub(boundary + escaped + r"(.+?)" + escaped + ending,
                      r"\1", text, flags=re.DOTALL)
    return " ".join(text.split())


def gqa_accuracy(reference, hypothesis):
    """Binary GQA exact match against one reference, with surrounding whitespace trimmed.

    A matching generated answer receives 1 and a wrong answer receives 0.
    Case, punctuation, and internal whitespace remain significant; answers are
    not extracted from prose, mapped to synonyms, or scored by annotator votes.
    """
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise TypeError("GQA accuracy requires one reference string and one prediction string")
    return float(reference.strip() == hypothesis.strip())


def gqa_score_pairs(references, hypotheses):
    """Per-example binary accuracy for aligned GQA references and predictions."""
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have equal lengths")
    return [{"accuracy": gqa_accuracy(ref, hyp)}
            for ref, hyp in zip(references, hypotheses)]


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
    _add_bertscores(rows, references, hypotheses, bert_scorer, ("f1",))
    return rows


def _add_bertscores(rows, references, hypotheses, bert_scorer, components):
    """Reuse one semantic scoring call, with explicit empty-answer conventions."""
    nonempty = [i for i, (ref, hyp) in enumerate(zip(references, hypotheses))
                if _as_text(ref).strip() and _as_text(hyp).strip()]
    nonempty_indices = set(nonempty)
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        if i not in nonempty_indices:
            for component in components:
                rows[i][f"bertscore_{component}"] = float(
                    not _as_text(ref).strip() and not _as_text(hyp).strip())
    if nonempty:
        precision, recall, f1 = bert_scorer.score(
            [_as_text(hypotheses[i]) for i in nonempty],
            [_as_text(references[i]) for i in nonempty])
        for component, tensor in (("precision", precision), ("recall", recall), ("f1", f1)):
            if component not in components:
                continue
            values = tensor.detach().cpu().tolist() if hasattr(tensor, "detach") else list(tensor)
            if len(values) != len(nonempty):
                raise RuntimeError("BERTScore returned the wrong number of examples")
            for i, value in zip(nonempty, values):
                rows[i][f"bertscore_{component}"] = float(value)


def scivqa_score_pairs(references, hypotheses, bert_scorer):
    """SciVQA Tables 1/2: ROUGE-1/L and BERTScore precision, recall and F1.

    ROUGE uses rouge-score's lowercase alphanumeric tokenizer without stemming.
    Scores are macro means of per-answer values when aggregated by mean_scores.
    Answers retain source choice labels and numerical formats; no gold-dependent
    abstention rewrite, synonym mapping, or choice reordering is applied.
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have equal lengths")
    if bert_scorer is None:
        raise ValueError("A BERTScore scorer is required for the nine SciVQA metrics")
    from rouge_score.rouge_scorer import RougeScorer

    rouge = RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    rows = []
    for reference, hypothesis in zip(references, hypotheses):
        scores = rouge.score(_as_text(reference), _as_text(hypothesis))
        row = {}
        for source, target in (("rouge1", "rouge_1"), ("rougeL", "rouge_l")):
            score = scores[source]
            row.update({f"{target}_f1": score.fmeasure,
                        f"{target}_precision": score.precision,
                        f"{target}_recall": score.recall})
        rows.append(row)
    _add_bertscores(rows, references, hypotheses, bert_scorer,
                   ("precision", "recall", "f1"))
    return rows


def mean_scores(rows, metrics=PAPER_METRICS):
    if not rows:
        raise ValueError("Cannot aggregate an empty evaluation set")
    return {metric: fmean(row[metric] for row in rows) for metric in metrics}
