"""Bounded-memory, full-corpus dictionary retrieval and held-out validation."""
import numpy as np
import torch
from .extract import extract_full_sequence
from .retrieval import normalize_embeddings, metrics_from_gloss_ranks
from ..data.pose_dataset import PoseDataset
from ..data.samplers import gloss_key


def dictionary_metrics(gallery, queries, gallery_labels, query_labels, gallery_ids, query_ids,
                       device='cpu', query_chunk=64, gallery_chunk=8192, should_stop=None):
    g, gv = normalize_embeddings(gallery); q, qv = normalize_embeddings(queries)
    if g.shape[1] != q.shape[1] or len(g) != len(gallery_labels) or len(q) != len(query_labels):
        raise ValueError('Dictionary embedding/label dimensions mismatch')
    if len(gallery_ids) != len(g) or len(query_ids) != len(q):
        raise ValueError('Dictionary ID dimensions mismatch')
    if len(set(gallery_ids)) != len(g) or len(set(query_ids)) != len(q) or set(gallery_ids) & set(query_ids):
        raise ValueError('Dictionary needs unique, disjoint source IDs')
    if not len(q): raise ValueError('Dictionary has no queries')
    classes = sorted(set(gallery_labels) | set(query_labels)); lookup = {v:i for i,v in enumerate(classes)}
    gy = np.array([lookup[x] for x in gallery_labels]); qy = np.array([lookup[x] for x in query_labels])
    good_g = np.flatnonzero(gv); present = np.zeros(len(classes), bool); present[gy[good_g]] = True
    ranks = np.full(len(q), np.inf)
    # Explicit pessimistic ties match the historical dictionary protocol.
    with torch.inference_mode():
        for start in range(0, len(q), query_chunk):
            if should_stop and should_stop(): raise InterruptedError('Dictionary evaluation paused')
            indices = np.arange(start, min(start+query_chunk, len(q)))
            features = torch.from_numpy(q[indices]).to(device)
            scores = torch.full((len(indices), len(classes)), -torch.inf, device=device)
            for offset in range(0, len(good_g), gallery_chunk):
                gi = good_g[offset:offset+gallery_chunk]
                sim = features @ torch.from_numpy(g[gi]).to(device).T
                labels = torch.from_numpy(gy[gi]).to(device).expand(len(indices), -1)
                scores.scatter_reduce_(1, labels, sim, reduce='amax', include_self=True)
            scores = scores.cpu().numpy(); truth = scores[np.arange(len(indices)), qy[indices]]
            good = qv[indices] & present[qy[indices]]
            ranks[indices[good]] = (scores[good] >= truth[good, None]).sum(1)
    metrics = metrics_from_gloss_ranks(ranks)
    metrics['DCG'] = metrics['DCG_single_relevant']
    metrics['R@1_classmacro'] = float(np.mean([(ranks[qy == c] <= 1).mean() for c in np.unique(qy)]))
    records = [{'sample_id':sid, 'rank':int(ranks[i]) if np.isfinite(ranks[i]) else None,
                'failure':'invalid_query_embedding' if not qv[i] else 'missing_gallery_gloss' if not present[qy[i]] else None}
               for i,sid in enumerate(query_ids)]
    return {'protocol':'dictionary_gloss_max_single_relevant', 'tie_rule':'pessimistic_ge', 'unit':'fraction',
            'metrics':metrics, 'queries':records,
            'gallery_coverage':{'total':len(g), 'valid':int(gv.sum()), 'valid_glosses':int(present.sum())}}


@torch.inference_mode()
def extract_manifest(encoder, manifest, config, progress_callback=None, should_stop=None):
    dataset = PoseDataset(manifest, config['data'], stage='evaluation')
    vectors = []
    for index in range(len(dataset)):
        if should_stop and should_stop(): raise InterruptedError('Feature extraction paused')
        result = extract_full_sequence(encoder, dataset[index], config['evaluation']['window_frames'],
                                       config['evaluation']['window_stride_frames'])
        vectors.append(result['raw_embedding'])
        if progress_callback and (index % 100 == 0 or index+1 == len(dataset)):
            progress_callback(index+1, len(dataset))
    return torch.stack(vectors).numpy(), dataset.rows


def validate_dictionary(encoder, gallery_manifest, query_manifest, config,
                        progress_callback=None, should_stop=None):
    def notify(split):
        return lambda completed,total: progress_callback(split,completed,total) if progress_callback else None
    gallery, gr = extract_manifest(encoder, gallery_manifest, config, notify('gallery'), should_stop)
    queries, qr = extract_manifest(encoder, query_manifest, config, notify('validation'), should_stop)
    return dictionary_metrics(gallery, queries, [gloss_key(r) for r in gr], [gloss_key(r) for r in qr],
                              [r['sample_id'] for r in gr], [r['sample_id'] for r in qr],
                              device=next(encoder.parameters()).device, should_stop=should_stop)
