"""Explicit micro/macro class-ranking metrics for adapted dictionary lookup.

Inherits the verified binary-DCG/MRR class-ranking scorer, adding complete-class
aggregation; this is not the official MS-ASL classifier/paper pipeline.
"""
from collections import Counter
import math

from metrics import evaluate, score_ranking


def evaluate_dictionary(predictions, references, class_ids):
    if (not class_ids or any(type(c) is not int or c < 0 for c in class_ids)
            or len(set(class_ids)) != len(class_ids)):
        raise ValueError('Invalid class vocabulary')
    if any(type(r['label']) is not int or not isinstance(r['sample_id'],str) or not r['sample_id']
           for r in references):
        raise ValueError('Reference label/ID types invalid')
    micro = evaluate(predictions, references, class_ids)
    counts = Counter(x['label'] for x in references)
    if set(counts) != set(class_ids):
        raise ValueError('Every declared class requires query support for macro scoring')
    observed = {p['sample_id']:p['ranking'] for p in predictions}
    grouped = {c:[] for c in class_ids}
    for ref in references:
        grouped[ref['label']].append(score_ranking(observed[ref['sample_id']], ref['label'], class_ids))
    by_class = {str(c):{metric:math.fsum(x[metric] for x in grouped[c])/counts[c] for metric in micro}
                for c in class_ids}
    macro = {metric:math.fsum(by_class[str(c)][metric] for c in class_ids)/len(class_ids) for metric in micro}
    return dict(micro=micro, macro=macro, per_class=by_class,
                class_support={str(c):counts[c] for c in class_ids}, query_count=len(references),
                class_count=len(class_ids),
                specification='Binary DCG1/log2(rank+1), MRR1/rank; complete class ranking; TopK uses rank<=K (K>=C yields1). Micro weights queries; macro gives each declared supported class equal weight. Ascending-class tie rule inherited locally. Adapted protocol only.')
