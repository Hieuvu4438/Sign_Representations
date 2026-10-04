"""ASL Citizen class-ranking metrics with explicit prediction coverage."""
import math


def ranked_classes(scores, class_ids):
    """Descending scores; ties resolved by ascending class ID (local rule)."""
    if len(scores) != len(class_ids) or not class_ids or len(set(class_ids)) != len(class_ids):
        raise ValueError('Scores must cover a unique nonempty class vocabulary')
    if not all(math.isfinite(float(s)) for s in scores):
        raise ValueError('Nonfinite prediction score')
    return [c for s, c in sorted(zip(scores, class_ids), key=lambda x: (-float(x[0]), x[1]))]


def score_ranking(ranking, label, class_ids):
    if not class_ids or len(set(class_ids)) != len(class_ids):
        raise ValueError('Vocabulary must be unique and nonempty')
    if len(ranking) != len(class_ids) or set(ranking) != set(class_ids):
        raise ValueError('Ranking must be a complete permutation of vocabulary')
    if label not in class_ids:
        raise ValueError('Unknown ground-truth label')
    rank = ranking.index(label) + 1
    return {'dcg': 1 / math.log2(rank + 1),
            'top1': float(rank <= 1), 'top5': float(rank <= 5),
            'top10': float(rank <= 10), 'top20': float(rank <= 20),
            'mrr': 1 / rank}


def evaluate(predictions, references, class_ids):
    """Order independent join by sample ID; all queries have exactly one label."""
    expected = {r['sample_id']: r['label'] for r in references}
    if len(expected) != len(references) or not expected:
        raise ValueError('Reference IDs must be unique and nonempty')
    observed = {p['sample_id']: p['ranking'] for p in predictions}
    if len(observed) != len(predictions) or set(observed) != set(expected):
        raise ValueError('Prediction IDs must exactly match reference IDs without duplicates')
    rows = [score_ranking(observed[k], v, class_ids) for k, v in expected.items()]
    return {name: math.fsum(row[name] for row in rows)/len(rows) for name in rows[0]}
