"""Resume-safe features and historical evaluation with explicit failure coverage."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from signrepr.export.api import export_checkpoint, SignRepresentationModel
from signrepr.data.schema import read_manifest, load_pose, sha256_file
from signrepr.training.job_support import atomic_json


def run(root):
    torch.set_num_threads(4)
    job = json.loads((root/'job.json').read_text())
    out = root/'evaluation'; out.mkdir(exist_ok=True)
    checkpoint = Path(job['train_root'])/'best_validation.pt'
    if not checkpoint.exists(): raise ValueError('Training must produce a validation-selected checkpoint')
    sha = sha256_file(checkpoint)
    bundle = out/'bundle'
    if not (bundle/'metadata.json').exists(): export_checkpoint(checkpoint, bundle)
    model = SignRepresentationModel(bundle, 'cuda')
    if model.metadata['source_checkpoint_sha256'] != sha: raise ValueError('Evaluation checkpoint changed')
    dataset = {r['sample_id']: r for r in read_manifest(Path(job['data_root'])/'train.jsonl')+read_manifest(Path(job['data_root'])/'val_all.jsonl')}
    stop = {'requested': False}
    def halt(sig, frame): stop['requested'] = True
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, halt)
    ids = list(dataset)
    cache = out/'clips'; cache.mkdir(exist_ok=True)
    started = time.time()
    for i, sid in enumerate(ids):
        if stop['requested'] or (root/'pause.request').exists(): return {'status': 'paused'}
        token = __import__('hashlib').sha256(sid.encode()).hexdigest()
        path = cache/f'{token}.npz'
        record = dataset[sid]
        valid_cached = False
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as data:
                    valid_cached = str(data['checkpoint_sha256']) == sha and str(data['pose_sha256']) == record['pose_sha256']
            except (OSError, ValueError, EOFError): pass
        if not valid_cached:
            result = model.encode_clip(load_pose(record['pose_path']), record)
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, raw_embedding=result['embedding'].numpy(), status=np.array(result['status']),
                                checkpoint_sha256=np.array(sha), pose_sha256=np.array(record['pose_sha256']))
            os.replace(temporary, path)
        if i % 20 == 0 or i+1 == len(ids):
            atomic_json(root/'progress.json', {'phase': 'evaluate_extract', 'status': 'running', 'completed': i+1,
                                               'total': len(ids), 'updated_unix': time.time(), 'elapsed_seconds': time.time()-started,
                                               'pid': os.getpid()})
            (root/'progress.txt').write_text(f'Pose-JEPA ASLC3000 | evaluate extract\nClips: {i+1}/{len(ids)}\nUpdated: {time.ctime()}\n')
    # Assemble exact original row order; evaluation labels/signers match the old
    # evaluator's own rows, independently from the training/selection manifests.
    baseline = Path(job['baseline_root'])
    require(sha256_file(baseline/'gallery.jsonl') == job['original_gallery_sha256'], 'Original gallery changed')
    require(sha256_file(baseline/'queries.jsonl') == job['original_queries_sha256'], 'Original queries changed')
    gallery = [json.loads(line) for line in (baseline/'gallery.jsonl').read_text().splitlines()]
    queries = [json.loads(line) for line in (baseline/'queries.jsonl').read_text().splitlines()]
    aslc = [{'sample_id': Path(r['video']).stem, 'video': r['video'], 'gloss': r['gloss'], 'label': r['label'],
             'signer': r['signer_id'], 'split': r['split'], 'role': role} for entries, role in [(gallery, 'gallery'), (queries, 'query')] for r in entries]
    import csv
    dataset_root = Path(next(iter(dataset.values()))['video_path']).parent.parent
    with (dataset_root/'splits/val.csv').open() as file:
        val = [{'sample_id': Path(r['Video file']).stem, 'video': str(dataset_root/'videos'/r['Video file']),
                'gloss': r['Gloss'], 'signer': r['Participant ID'], 'split': 'val'} for r in csv.DictReader(file)]
    def matrix(rows):
        vectors = []
        for row in rows:
            token = __import__('hashlib').sha256(row['sample_id'].encode()).hexdigest()
            with np.load(cache/f'{token}.npz') as data:
                vector = data['raw_embedding']
                valid = str(data['status']) == 'ok' and np.isfinite(vector).all() and np.linalg.norm(vector) > 1e-12
                row['feature_valid'] = bool(valid)
                row['feature_status'] = str(data['status'])
                vectors.append(vector if valid else np.zeros_like(vector))
        return np.stack(vectors)
    x = matrix(aslc); v = matrix(val)
    # Primary raw result remains separate from the historical whitening adapter.
    from signrepr.evaluation.retrieval import rank_dictionary
    raw_result = rank_dictionary(x[:3000], x[3000:], [r['sample_id'] for r in aslc[:3000]],
                                 [r['sample_id'] for r in aslc[3000:]], [r['gloss'] for r in aslc[:3000]],
                                 [r['gloss'] for r in aslc[3000:]])
    atomic_json(out/'raw_retrieval.json', raw_result)
    from signrepr.evaluation.loso import loso_metrics
    atomic_json(out/'raw_loso.json', loso_metrics(v, [r['gloss'] for r in val],
                                                [r['signer'] for r in val], [r['sample_id'] for r in val]))
    atomic_json(out/'coverage.json', {name: {'total': len(rows), 'valid': sum(r['feature_valid'] for r in rows),
                                          'failed_ids': [r['sample_id'] for r in rows if not r['feature_valid']]}
                                     for name, rows in [('gallery', aslc[:3000]), ('queries', aslc[3000:]), ('val', val)]})
    # scipy/sklearn evaluation uses the already installed, pinned legacy env.
    raw = out/'raw_features'; raw.mkdir(exist_ok=True)
    np.save(raw/'aslc.npy', x); np.save(raw/'val.npy', v)
    atomic_json(raw/'rows.json', {'aslc300': aslc, 'val': val})
    env = os.environ.copy(); env['PYTHONPATH'] = job['environment']['PYTHONPATH']
    command = [job['evaluation_python'], str(Path(__file__).resolve()), '--job-root', str(root), '--legacy-only']
    child = subprocess.Popen(command, env=env, start_new_session=False)
    while child.poll() is None:
        if stop['requested'] or (root/'pause.request').exists():
            child.terminate(); child.wait(); return {'status': 'paused'}
        time.sleep(1)
    require(child.returncode == 0, 'Legacy frozen evaluation failed; see evaluation log')
    result = {'status': 'complete', 'checkpoint_sha256': sha, 'raw_report': str(out/'raw_retrieval.json'),
              'legacy_report': str(out/'legacy/report.json'), 'finished_unix': time.time(),
              'selection_note': job['selection_note']}
    atomic_json(out/'complete.json', result)
    atomic_json(root/'progress.json', {'phase': 'complete', **result})
    (root/'progress.txt').write_text(f"Pose-JEPA ASLC3000 | COMPLETE\nResults: {out}\n")
    return result


def require(value, message):
    if not value: raise ValueError(message)


def failure_ranks(features, labels, query_indices, gallery_indices, valid, nclasses, chunk=64):
    """Historical pessimistic ties; invalid/missing-positive queries have rank inf."""
    qidx = np.asarray(query_indices); gidx = np.asarray(gallery_indices)
    gidx = gidx[valid[gidx]]
    ranks = np.full(len(qidx), np.inf)
    if not len(gidx): return ranks
    present = np.zeros(nclasses, bool); present[labels[gidx]] = True
    for start in range(0, len(qidx), chunk):
        indices = qidx[start:start+chunk]
        scores = np.full((len(indices), nclasses), -np.inf, np.float32)
        sim = features[indices] @ features[gidx].T
        np.maximum.at(scores, (np.arange(len(indices))[:, None], labels[gidx][None, :]), sim)
        truth = scores[np.arange(len(indices)), labels[indices]]
        good = valid[indices] & present[labels[indices]]
        ranks[start:start+len(indices)][good] = (scores[good] >= truth[good, None]).sum(1)
    return ranks


def install_failure_aware_evaluation(evaluator):
    """Keep original folds, pooling, whitening, probes and metrics; mask failures."""
    def inputs(name, split):
        clips, win, offs, spans = evaluator.load(name, split)
        require(np.all(np.diff(offs) == 1), 'Failure adapter requires one whole-clip window')
        valid = np.array([c['feature_valid'] for c in clips], bool)
        return clips, win, offs, spans, valid

    def whiten_valid(win, fit_mask, stride):
        sample = win[fit_mask][::stride]; mu = sample.mean(0)
        _, singular, axes = np.linalg.svd(sample-mu, full_matrices=False)
        singular = singular/np.sqrt(len(sample)); k = min(512, len(axes))
        return ((win-mu) @ axes[:k].T)/(singular[:k]+1e-3*singular[0])

    def eval_loso(name, with_probes=True):
        clips, win, offs, spans, valid = inputs(name, 'val')
        gloss = np.array([c['gloss'] for c in clips]); _, y = np.unique(gloss, return_inverse=True)
        sg = np.array([c['signer'] for c in clips]); result = {}
        for mode, src in [('white512', whiten_valid(win, valid, 7)), ('mean', win)]:
            features = evaluator.pool(src, offs, 'mean'); ranks = np.full(len(y), np.inf)
            for signer in np.unique(sg):
                q = np.flatnonzero(sg == signer); g = np.flatnonzero(sg != signer)
                ranks[q] = failure_ranks(features, y, q, g, valid, int(y.max())+1)
            result[mode] = evaluator.metrics_from_ranks(ranks, y)
            if mode == 'white512': white_ranks = ranks
            else: mean_features = features; result['geometry'] = evaluator.geometry(features[valid])
        if with_probes:
            # Diagnostic regression probes report valid-feature coverage separately.
            good = np.flatnonzero(valid)
            pri = evaluator.window_priors([clips[i] for i in good], np.arange(len(good)+1), spans[good])
            result['probe_R2'] = {k: v for k, v in evaluator.probes(win[valid], sg[valid], pri).items() if not k.endswith('alpha')}
            result['signer_probe_acc'] = evaluator.signer_probe(mean_features[valid], sg[valid], gloss[valid])
        result['coverage'] = {'total': len(valid), 'valid': int(valid.sum()), 'probe_scope': 'valid features only'}
        return result, white_ranks, sg

    def eval_aslc300(name):
        clips, win, offs, _, valid = inputs(name, 'aslc300')
        role = np.array([c['role'] for c in clips]); y = np.array([c['label'] for c in clips])
        sg = np.array([c['signer'] for c in clips]); gi = np.flatnonzero(role == 'gallery'); qi = np.flatnonzero(role == 'query')
        good_g = gi[valid[gi]]; fit = (role == 'gallery') & valid
        result = {}; mean_features = evaluator.pool(win, offs, 'mean')
        for mode, features in [('white512', evaluator.pool(whiten_valid(win, fit, 3), offs, 'mean')), ('mean', mean_features)]:
            ranks = failure_ranks(features, y, qi, gi, valid, int(y.max())+1)
            result['retrieval_'+mode] = evaluator.metrics_from_ranks(ranks, y[qi])
            if mode == 'white512': white_ranks = ranks
        X = mean_features
        Xs = (X-X[good_g].mean(0))/(X[good_g].std(0)+1e-6)
        sig_g = np.unique(sg[good_g]); rng = np.random.default_rng(0)
        hold = rng.choice(sig_g, max(1, len(sig_g)//5), replace=False)
        a, b = good_g[~np.isin(sg[good_g], hold)], good_g[np.isin(sg[good_g], hold)]
        best, scores = None, {}
        for C in (0.01, 0.1, 1., 10.):
            if not len(b): break
            clf = evaluator.LogisticRegression(C=C, max_iter=3000).fit(Xs[a], y[a])
            scores[C] = float((clf.predict(Xs[b]) == y[b]).mean())
            if best is None or scores[C] > scores[best]: best = C
        clf = evaluator.LogisticRegression(C=best or 1., max_iter=3000).fit(Xs[good_g], y[good_g])
        correct = (clf.predict(Xs[qi]) == y[qi]) & valid[qi]
        result['linear_probe'] = {'C': best, 'val_scores': scores, 'instance_top1': float(correct.mean()),
                                  'macro_top1': float(np.mean([correct[y[qi] == c].mean() for c in np.unique(y[qi])]))}
        result['_probe_correct'] = correct.astype(np.int8).tolist(); result['_probe_y'] = y[qi].tolist()
        result['coverage'] = {'gallery_total': len(gi), 'gallery_valid': len(good_g),
                              'query_total': len(qi), 'query_valid': int(valid[qi].sum())}
        return result, white_ranks, sg[qi]

    evaluator.eval_loso = eval_loso; evaluator.eval_aslc300 = eval_aslc300


def legacy(root):
    job = json.loads((root/'job.json').read_text()); workspace = Path(job['workspace'])
    sys.path.insert(0, str(workspace/'method/obs_jepa'))
    import oj.evaluate as evaluator
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits
    out = root/'evaluation'; legacy_root = out/'legacy'; name = 'PJ14_HYBRID_ASLC3000_grp'
    raw = out/'raw_features'; rows = json.loads((raw/'rows.json').read_text())
    x = np.load(raw/'aslc.npy'); v = np.load(raw/'val.npy')
    with threadpool_limits(limits=4):
        good_gallery = np.array([r['feature_valid'] for r in rows['aslc300'][:3000]], bool)
        pca = PCA(n_components=768, svd_solver='full').fit(x[:3000][good_gallery])
        projection = legacy_root/'projection.npz'; projection.parent.mkdir(parents=True, exist_ok=True)
        np.savez(projection, mean=pca.mean_, components=pca.components_)
        for split, values in [('aslc300', x), ('val', v)]:
            destination = legacy_root/'feats'/name/split; destination.mkdir(parents=True, exist_ok=True)
            projected = pca.transform(values).astype(np.float16)
            projected[~np.array([r['feature_valid'] for r in rows[split]], bool)] = 0
            np.save(destination/'windows.npy', projected)
            np.save(destination/'offsets.npy', np.arange(len(values)+1, dtype=np.int64))
            manifest = {r['sample_id']: r for r in read_manifest(Path(job['data_root'])/('val_all.jsonl' if split == 'val' else 'train.jsonl'))}
            if split == 'aslc300': manifest.update({r['sample_id']: r for r in read_manifest(Path(job['data_root'])/'queries.jsonl')})
            np.save(destination/'src_span.npy', np.array([[0, manifest[r['sample_id']]['num_frames']] for r in rows[split]], dtype=np.int64))
            (destination/'clips.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[split]))
            atomic_json(destination/'meta.json', {'readout': 'whole_clip_per_part_valid_mean_after_unique_segment_stitch',
                                                'projection': '768D PCA fit on valid clips within the 3000 gallery only',
                                                'one_whole_clip_window_per_sample': True, 'checkpoint': 'best_validation.pt'})
        evaluator.RUNTIME = legacy_root; evaluator.EVAL_DIR = legacy_root/'eval'
        install_failure_aware_evaluation(evaluator)
        results = evaluator.evaluate_model(name, 'both', True, True)
        references = {}
        for reference in ['S1_FULL_4k_grp', 'S2_FULL_4k_grp', 'PJ_PLctx_16k_grp', 'SR3D', 'JH3D', 'JV3D', 'B13']:
            for folder in [workspace/'method/phonojepa_s1/runtime/eval/models', workspace/'method/phonojepa_s2/runtime/eval/models', workspace/'method/obs_jepa/runtime/eval/models']:
                path = folder/f'{reference}.json'
                if path.exists(): references[reference] = json.loads(path.read_text()); break
        def serializable(value):
            if isinstance(value, np.ndarray): return value.tolist()
            if isinstance(value, np.generic): return value.item()
            raise TypeError(type(value).__name__)
        results = json.loads(json.dumps(results, default=serializable))
        references = json.loads(json.dumps(references, default=serializable))
        atomic_json(legacy_root/'report.json', {'model': name, 'results': results, 'references': references,
                                              'evaluator_sha256': sha256_file(Path(evaluator.__file__)),
                                              'input_projection': 'valid-gallery-only 1024->768 PCA, then historical whitening/linear probes',
                                              'failure_adapter': 'Invalid gallery excluded; invalid/missing-positive retrieval queries rank infinity; invalid LP queries count incorrect. Retrieval/LP keep every query. Geometry/body-part/signer probes use valid features and report coverage.',
                                              'selection_scope': job['selection_note'],
                                              'budget_note': '30k maximum updates, unlike 4k/16k baselines; not budget-matched'})
    return {'status': 'complete'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-root', required=True); parser.add_argument('--legacy-only', action='store_true')
    args = parser.parse_args(); root = Path(args.job_root).resolve()
    print(json.dumps(legacy(root) if args.legacy_only else run(root)), flush=True)
