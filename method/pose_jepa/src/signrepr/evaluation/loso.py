"""Raw frozen LOSO validation: exclude same signer, aggregate video cosine by gloss."""
import numpy as np
import torch
from .retrieval import normalize_embeddings, metrics_from_gloss_ranks
from .representation import representation_diagnostics
from .extract import extract_full_sequence
from ..data.pose_dataset import PoseDataset
from ..data.samplers import gloss_key


def loso_metrics(embeddings, labels, signers, sample_ids, chunk_size=64):
    features, valid = normalize_embeddings(embeddings)
    if len(features) != len(labels) or len(features) != len(signers) or len(features) != len(sample_ids):
        raise ValueError('LOSO metadata count mismatch')
    if len(set(sample_ids)) != len(sample_ids) or any(s is None for s in signers):
        raise ValueError('LOSO needs unique clips and known signer IDs')
    classes = sorted(set(labels)); mapping = {label: i for i, label in enumerate(classes)}
    y = np.array([mapping[label] for label in labels]); signer = np.array(signers)
    ranks, records = [], []
    for start in range(0, len(features), chunk_size):
        scores = features[start:start+chunk_size] @ features.T
        for row, score in enumerate(scores):
            i = start+row
            allowed = valid & (signer != signer[i])
            failure = 'invalid_query_embedding' if not valid[i] else 'no_cross_signer_positive' if not (allowed & (y == y[i])).any() else None
            rank = float('inf')
            if failure is None:
                sc = np.full(len(classes), -np.inf, np.float32)
                np.maximum.at(sc, y[allowed], score[allowed])
                order = np.lexsort((np.arange(len(classes)), -sc))
                rank = int(np.flatnonzero(order == y[i])[0])+1
            ranks.append(rank)
            records.append({'sample_id': sample_ids[i], 'rank': int(rank) if np.isfinite(rank) else None, 'failure': failure})
    return {'protocol': 'selection_raw_loso_gloss_max', 'metrics': metrics_from_gloss_ranks(ranks), 'queries': records,
            'representation_diagnostics': representation_diagnostics(embeddings)}


@torch.inference_mode()
def validate_loso(encoder, manifest, config):
    dataset = PoseDataset(manifest, config['data'], stage='evaluation')
    vectors = []
    for index in range(len(dataset)):
        result = extract_full_sequence(encoder, dataset[index], config['evaluation']['window_frames'], config['evaluation']['window_stride_frames'])
        vectors.append(result['raw_embedding'])
    return loso_metrics(torch.stack(vectors).numpy(), [gloss_key(r) for r in dataset.rows],
                        [r['signer_id'] for r in dataset.rows], [r['sample_id'] for r in dataset.rows])
