from collections import Counter


def normalize_text(text):
    return " ".join(text.lower().strip().split())


def compute_em_and_f1(references, hypotheses):
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have equal lengths")
    if not references:
        return 0.0, 0.0
    total_em = total_f1 = 0.0
    for ref, hyp in zip(references, hypotheses):
        ref = normalize_text(ref if isinstance(ref, str) else " ".join(ref))
        hyp = normalize_text(hyp if isinstance(hyp, str) else " ".join(hyp))
        total_em += float(ref == hyp)
        ref_tokens, hyp_tokens = ref.split(), hyp.split()
        if not ref_tokens and not hyp_tokens:
            total_f1 += 1.0
            continue
        common = sum((Counter(ref_tokens) & Counter(hyp_tokens)).values())
        if common:
            total_f1 += 2 * common / (len(ref_tokens) + len(hyp_tokens))
    return total_em / len(references), total_f1 / len(references)
