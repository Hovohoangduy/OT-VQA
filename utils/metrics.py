def normalize_text(text):
        # Lowercase, strip, and normalize spaces
        text = text.lower().strip()
        text = ' '.join(text.split())  # Remove redundant spaces
        return text

def compute_em_and_f1(references, hypotheses):
    """
    Computes the Exact Match (EM) and F1 score for a batch of predictions.
    :param references: List of reference answers (ground truth).
    :param hypotheses: List of predicted answers.
    :return: Tuple (em_score, f1_score)
    """

    total_em = 0
    total_f1 = 0

    for ref, hyp in zip(references, hypotheses):
        # Normalize and join tokens for comparison
        ref_normalized = normalize_text(' '.join(ref))
        hyp_normalized = normalize_text(' '.join(hyp))

        # Exact Match
        if ref_normalized == hyp_normalized:
            total_em += 1

        # Token-level F1 Score Calculation
        ref_tokens = set(ref_normalized.split())
        hyp_tokens = set(hyp_normalized.split())

        common_tokens = ref_tokens & hyp_tokens
        precision = len(common_tokens) / len(hyp_tokens) if hyp_tokens else 0
        recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
        total_f1 += f1

    em_score = total_em / len(references)
    avg_f1_score = total_f1 / len(references)
    return em_score, avg_f1_score
