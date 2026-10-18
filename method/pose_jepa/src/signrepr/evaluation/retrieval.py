"""Exact cosine reference with fixed failure denominators and deterministic ties."""
import numpy as np


def normalize_embeddings(embeddings):
    e = np.asarray(embeddings, dtype=np.float32)
    if e.ndim != 2:
        raise ValueError('Expected embedding matrix')
    norm = np.linalg.norm(e, axis=-1)
    valid = np.isfinite(e).all(-1) & np.isfinite(norm) & (norm > 1e-12)
    out = np.zeros_like(e)
    out[valid] = e[valid]/norm[valid, None]
    return out, valid


def metrics_from_gloss_ranks(ranks, recall_k=(1, 5, 10)):
    ranks = np.asarray(ranks, dtype=np.float64)
    if ranks.ndim != 1 or not len(ranks) or np.any(np.isnan(ranks)) or np.any(ranks < 1):
        raise ValueError('Ranks must be nonempty, 1-based, with infinity for failure')
    valid = np.isfinite(ranks)
    rr, dcg = np.zeros_like(ranks), np.zeros_like(ranks)
    rr[valid] = 1/ranks[valid]
    dcg[valid] = 1/np.log2(ranks[valid]+1)
    return {**{f'R@{k}': float(np.mean(ranks <= k)) for k in recall_k}, 'MRR': float(rr.mean()),
            'DCG_single_relevant': float(dcg.mean()), 'num_queries': len(ranks), 'num_ranked_queries': int(valid.sum())}


def rank_dictionary(gallery, queries, gallery_ids, query_ids, gallery_glosses, query_glosses,
                    protocol='gloss_max_single_relevant', recall_k=(1, 5, 10), chunk_size=128):
    g, gv = normalize_embeddings(gallery)
    q, qv = normalize_embeddings(queries)
    if g.shape[1] != q.shape[1] or len(g) != len(gallery_ids) or len(g) != len(gallery_glosses) or len(q) != len(query_ids) or len(q) != len(query_glosses):
        raise ValueError('Embedding and metadata dimensions mismatch')
    if len(set(gallery_ids)) != len(gallery_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError('Duplicate sample IDs')
    if set(gallery_ids) & set(query_ids):
        raise ValueError('Query and gallery must contain distinct clips')
    if not len(q):
        raise ValueError('No queries')
    if protocol not in ('gloss_max_single_relevant', 'video_rank_multiple_relevant'):
        raise ValueError('Unknown retrieval protocol')
    indices = np.flatnonzero(gv)
    keys = sorted({gallery_glosses[i] for i in indices})
    groups = {key: [i for i in indices if gallery_glosses[i] == key] for key in keys}
    ranks, records, aps, ndcgs = [], [], [], []
    for start in range(0, len(q), chunk_size):
        scores = q[start:start+chunk_size] @ g.T
        for local, score in enumerate(scores):
            i = start+local
            relevant = query_glosses[i]
            failure = 'invalid_query_embedding' if not qv[i] else 'missing_gallery_gloss' if relevant not in groups else None
            ranking = []
            rank = float('inf')
            ap = ndcg = 0.
            if failure is None:
                if protocol == 'gloss_max_single_relevant':
                    ranking = sorted(keys, key=lambda key: (-float(score[groups[key]].max()), key))
                    rank = ranking.index(relevant)+1
                else:
                    order = sorted(indices, key=lambda j: (-float(score[j]), gallery_ids[j]))
                    hits = np.array([gallery_glosses[j] == relevant for j in order])
                    positions = np.flatnonzero(hits)+1
                    rank = int(positions[0])
                    ap = float(np.mean(np.arange(1, len(positions)+1)/positions))
                    dcg = np.sum(1/np.log2(positions+1))
                    ideal = np.sum(1/np.log2(np.arange(1, len(positions)+1)+1))
                    ndcg = float(dcg/ideal)
                    ranking = [gallery_ids[j] for j in order]
            ranks.append(rank); aps.append(ap); ndcgs.append(ndcg)
            records.append({'sample_id': query_ids[i], 'rank': None if not np.isfinite(rank) else int(rank),
                            'failure': failure, 'ranking': ranking})
    aggregate = metrics_from_gloss_ranks(ranks, recall_k)
    if protocol == 'video_rank_multiple_relevant':
        aggregate.pop('DCG_single_relevant')
        aggregate.update(mAP=float(np.mean(aps)), nDCG=float(np.mean(ndcgs)))
    covered = [ranks[i] for i in range(len(q)) if query_glosses[i] in groups]
    return {'protocol': protocol, 'unit': 'fraction', 'metrics': aggregate, 'queries': records,
            'gallery_coverage': {'total': len(g), 'valid': int(gv.sum()), 'valid_glosses': len(keys)},
            'covered_gallery_subset': metrics_from_gloss_ranks(covered, recall_k) if covered and protocol == 'gloss_max_single_relevant' else {'num_queries': len(covered)}}
