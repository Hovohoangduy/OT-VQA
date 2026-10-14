"""Restore numerical punctuation split by WordPiece answer decoding."""

import re


ANSWER_POSTPROCESSING = "wordpiece_numeric_spacing_v1"


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
