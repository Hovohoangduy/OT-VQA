"""Official-metric comparisons and supplementary numeric/abstention checks."""
from decimal import Decimal, InvalidOperation
import re

import numpy as np

NUMBER = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([^\d\s]*)\s*$")


def parse_number(text):
    match = NUMBER.fullmatch(str(text).replace("−", "-"))
    if match is None:
        return None
    try:
        return Decimal(match[1]), match[2].casefold()
    except InvalidOperation:
        return None


def guardrail_metrics(references, predictions):
    if len(references) != len(predictions) or not references:
        raise ValueError("Guardrails require equally sized nonempty predictions/references")
    exact = sum(str(a).strip().casefold() == str(b).strip().casefold() for a, b in zip(references, predictions))
    numeric = [(parse_number(a), parse_number(b)) for a, b in zip(references, predictions) if parse_number(a) is not None]
    correct = sum(b is not None and a == b for a, b in numeric)
    relaxed = sum(b is not None and a[1] == b[1] and abs(a[0]-b[0]) <= abs(a[0])*Decimal(".05") for a, b in numeric)
    abstain = lambda s: "not possible to answer" in str(s).casefold()
    answerable = [(a, b) for a, b in zip(references, predictions) if not abstain(a)]
    unanswerable = [(a, b) for a, b in zip(references, predictions) if abstain(a)]
    return {"exact_match": exact/len(references), "numeric_examples": len(numeric),
            "numeric_exact": correct/len(numeric) if numeric else None,
            "numeric_relaxed_5pct": relaxed/len(numeric) if numeric else None,
            "false_abstention_rate": sum(abstain(b) for _, b in answerable)/len(answerable) if answerable else None,
            "false_answer_rate": sum(not abstain(b) for _, b in unanswerable)/len(unanswerable) if unanswerable else None}


def paired_cluster_interval(deltas, groups, seed=1105, resamples=2000):
    """Paired macro-answer mean delta, resampling complete figures or papers."""
    deltas = np.asarray(deltas, dtype=float)
    groups = np.asarray(groups, dtype=str)
    if deltas.shape != groups.shape or not len(deltas) or not np.isfinite(deltas).all():
        raise ValueError("Cluster deltas/groups must be finite and nonempty")
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=deltas)
    sizes = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(resamples):
        selected = rng.integers(0, len(unique), len(unique))
        means.append(sums[selected].sum()/sizes[selected].sum())
    lo, hi = np.quantile(means, [.025, .975])
    return {"delta": float(deltas.mean()), "ci95": [float(lo), float(hi)],
            "clusters": len(unique), "resamples": resamples, "seed": seed}
