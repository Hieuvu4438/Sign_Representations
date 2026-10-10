"""Measured CPU inference/ranking on frozen dev only; no selection or tuning."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from signrel.utils import read_jsonl, sha256, atomic_json, code_hash
from signrel.config import ROOT
from signrel.eval.embed import restore_encoder, embed_arrays
from signrel.eval.retrieval import score_retrieval
from signrel.data.cache import read_cache

torch.set_num_threads(4)
checkpoint = 'runs/pilot_a4_seed17/checkpoints/last.pt'
start = time.perf_counter()
model, cfg, state = restore_encoder(checkpoint)
restore_seconds = time.perf_counter() - start
index = read_jsonl('results/pilot_a4_seed17_dev/index.jsonl')
vectors = np.load('results/pilot_a4_seed17_dev/embeddings.npy', allow_pickle=False)
caches = {r['video_id']: r for r in read_jsonl('data/cache/eval_cache_manifest.jsonl')}
costs = []
maximum_error = 0.
for item in index:
    start = time.perf_counter()
    arrays, metadata = read_cache(caches[item['video_id']]['cache_path'])
    result = embed_arrays(arrays, model, 64, 32)
    costs.append(time.perf_counter() - start)
    assert result['status'] == item['status'] == 'ok'
    maximum_error = max(maximum_error, float(np.max(np.abs(result['embedding'] - vectors[item['row_index']]))))
assert maximum_error <= 2e-6
by_id = {r['video_id']: r['row_index'] for r in index}
labels = {r['video_id']: r for r in read_jsonl('data/manifests/eval_labels.jsonl')}
queries = json.loads(Path('data/splits/dev_query_ids.json').read_text())
gallery = json.loads(Path('data/splits/frozen_gallery_ids.json').read_text())
q = np.stack([vectors[by_id[i]] for i in queries])
g = np.stack([vectors[by_id[i]] for i in gallery])
ranking = []
for _ in range(5):
    start = time.perf_counter()
    result = score_retrieval(q, [labels[i] for i in queries], g, [labels[i] for i in gallery], True)
    ranking.append(time.perf_counter() - start)
    assert result['macro_gloss_r1'] == .0875
atomic_json('artifacts/inference_profile.json', {
    'status': 'measured', 'source_sha256': code_hash(ROOT), 'device': 'cpu', 'threads': 4,
    'checkpoint_sha256': sha256(checkpoint), 'cached_export_videos': len(costs),
    'restore_model_seconds': restore_seconds, 'total_cached_export_seconds': sum(costs),
    'cached_export_mean_seconds_per_video': float(np.mean(costs)),
    'cached_export_median_seconds_per_video': float(np.median(costs)),
    'cached_export_max_seconds_per_video': max(costs),
    'maximum_absolute_embedding_error_vs_saved_export': maximum_error,
    'ranking_repetitions': len(ranking), 'ranking_queries': len(queries), 'ranking_gallery_videos': len(gallery),
    'ranking_mean_seconds': float(np.mean(ranking)),
    'ranking_mean_seconds_per_query': float(np.mean(ranking) / len(queries)),
    'scope': 'CPU cached NPZ read/hash validation + frozen encoder/window/pooling; excludes raw RGB perception/teacher, one-time model restore and output-file writing. Ranking includes signer/content audit, normalization, gloss rank and metrics; excludes embedding generation and file loading. Dev protocol only.'})
print('CPU inference and ranking measured; vectors match saved dev export.')
