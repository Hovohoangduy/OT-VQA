"""Restore numerical punctuation split by WordPiece answer decoding."""

import re


ANSWER_POSTPROCESSING = "wordpiece_numeric_spacing_v2"


def restore_numeric_spacing(text):
    """Repair decimals without changing digits, choice lists, or answer content.

    WordPiece decodes ["0", ".", "5"] as "0. 5". Correct this even in
    a phrase such as "approximately 0. 5". Sign/thousands/exponent cleanup is
    restricted to an entire numerical answer so prose and choice lists survive.
    This function uses the prediction alone, never reference answers or types.
    """
    text = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", text)
    numeric = r"[+-]?\s*\d+(?:,\s*\d{3})*(?:\.\d+)?(?:\s*[eE]\s*[+-]?\s*\d+)?\s*%?"
    if re.fullmatch(numeric, text.strip()):
        return re.sub(r"\s+", "", text.strip())
    return text


def postprocess_answer(text):
    """Clean and standardize predictions following SciVQA benchmark standards."""
    text = restore_numeric_spacing(text)
    text = re.sub(r"\|end\|", "", text).strip()
    unanswerable_patterns = (
        "not possible to answer",
        "cannot be answered",
        "insufficient information",
        "impossible to answer",
        "cannot answer this question",
        "not enough information",
    )
    lower = text.lower()
    if any(p in lower for p in unanswerable_patterns):
        return "It is not possible to answer this question based only on the provided data."
    if re.fullmatch(r"[A-D](?:\s*,\s*[A-D])+", text):
        return re.sub(r"\s*,\s*", ",", text)
    opt_match = re.fullmatch(r"(?:option|choice)?\s*\(?([A-D])\)?\.?", text, re.IGNORECASE)
    if opt_match and len(text.strip()) <= 10:
        return opt_match.group(1).upper()
    return text
