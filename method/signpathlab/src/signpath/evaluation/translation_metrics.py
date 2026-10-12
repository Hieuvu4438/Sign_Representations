"""Pinned metric signatures for actual reference/hypothesis text, without gloss heuristics."""
import importlib.metadata
from .metrics import corpus_wer


def _lcs(first, second):
    previous = [0] * (len(second) + 1)
    for token in first:
        current = [0]
        for i, other in enumerate(second, 1):
            current.append(previous[i - 1] + 1 if token == other else max(previous[i], current[-1]))
        previous = current
    return previous[-1]


def translation_metrics(references, hypotheses, *, tokenizer="13a", lowercase=False):
    from sacrebleu.metrics import BLEU
    if len(references) != len(hypotheses) or not references or any(not isinstance(r, str) or not r.strip() for r in references) or any(not isinstance(h, str) for h in hypotheses):
        raise ValueError("Translation metrics require aligned nonempty references and text hypotheses")
    metric = BLEU(tokenize=tokenizer, lowercase=lowercase, effective_order=False, smooth_method="exp")
    bleu = metric.corpus_score(hypotheses, [references])
    refs = [(r.lower() if lowercase else r).split() for r in references]
    hyps = [(h.lower() if lowercase else h).split() for h in hypotheses]
    rouge = []
    for reference, hypothesis in zip(refs, hyps):
        length = _lcs(reference, hypothesis)
        rouge.append(2 * length / (len(reference) + len(hypothesis)))
    return {"bleu": float(bleu.score), "bleu_signature": str(metric.get_signature()),
        "sacrebleu_version": importlib.metadata.version("sacrebleu"), "rouge_l_f1": sum(rouge) / len(rouge),
        "rouge_recipe": "mean sentence LCS F1; whitespace word tokens; declared case",
        "wer": corpus_wer(refs, hyps), "word_preprocessing": "lowercase_whitespace" if lowercase else "case_preserving_whitespace"}
