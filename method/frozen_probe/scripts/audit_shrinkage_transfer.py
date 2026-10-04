"""H006_v1: fixed full-covariance controls for the rank8 PCA diagnostic."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import CacheReader, file_sha
from run_context import RunContext
from resources import scratch_preflight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import numpy as np
        import sklearn
        from geometry_transfer import apply_geometry, disjoint_cohorts, common_feature_recipe
        from shrinkage_geometry import NEW_MODES, fit_shrinkage, transformed, lookup_scores
        from window_budget import pooled_vector
        from metrics import evaluate, ranked_classes, score_ranking
        from analysis import participant_bootstrap
        for path, expected in c['input_hashes'].items():
            if file_sha(REPO / path) != expected:
                raise ValueError('Changed registered input ' + path)
        if sklearn.__version__ != c['sklearn_version']:
            raise ValueError('Unexpected covariance implementation version')
        resource = scratch_preflight('/dev/shm', 128 << 20)
        parent = REPO / c['parent_runtime']
        parent_config = json.loads((parent / 'config.json').read_text())
        old_status = json.loads((parent / 'status.json').read_text())
        for path, expected in parent_config['input_hashes'].items():
            if file_sha(REPO / path) != expected:
                raise ValueError('Changed original input ' + path)
        for path, expected in old_status['output_hashes'].items():
            if file_sha(parent / path) != expected:
                raise ValueError('Changed parent output ' + path)
        if old_status['status'] != 'completed':
            raise ValueError('Parent run incomplete')
        meta = json.loads((parent / 'fit_support_metadata.json').read_text())
        old_report = json.loads((parent / 'geometry_transfer.json').read_text())
        old_predictions = {(r['mode'], str(r['support_seed']), r['sample_id']): r['ranking']
                           for r in json.loads((parent / 'predictions.json').read_text())}
        original_fit = dict(np.load(parent / 'fitted_geometry.npz', allow_pickle=False))
        cohorts, readers, priors, samples = {}, {}, {}, {}
        for role in ('base', 'target'):
            prior = json.loads((REPO / parent_config[role + '_identity_config']).read_text())
            cache = json.loads((REPO / parent_config[role + '_cache_report']).read_text())
            if not cache['passed'] or file_sha(cache['cache_manifest']) != meta[role + '_manifest_sha256']:
                raise ValueError('Unavailable or changed cache')
            cohorts[role] = json.loads((REPO / prior['cohort_manifest']).read_text())
            audit = json.loads((REPO / prior['video_audit']).read_text())
            readers[role] = CacheReader(cache['cache_manifest'], prior['cache_identity'], cohorts[role], audit)
            priors[role] = prior
            samples[role] = {s['sample_id']: s for s in cohorts[role]['samples']}
        disjoint_cohorts(cohorts['base'], cohorts['target'])
        common_feature_recipe(priors['base']['cache_identity'], priors['target']['cache_identity'])
        fit_ids = sorted(k for k,s in samples['base'].items() if s['split'] == 'train')
        query_ids = sorted(k for k,s in samples['target'].items() if s['split'] == 'val')
        assert fit_ids == meta['fit_ids'] and query_ids == meta['query_ids']
        assert len(fit_ids) == 1313 and len(query_ids) == 383 and len(cohorts['target']['classes']) == 100
        labels = [samples['base'][k]['class_id'] for k in fit_ids]
        assert labels == meta['fit_labels']
        supports = meta['supports']
        assert set(supports) == {'1701','1702','1703'}
        for seed, support in supports.items():
            expected = [cohorts['target']['supports'][seed]['1'][g][0] for g in cohorts['target']['classes']]
            assert support == expected and len(set(support)) == 100
            assert [samples['target'][k]['class_id'] for k in support] == list(range(100))
            assert all(samples['target'][k]['split'] == 'train' for k in support)
        query_people = {samples['target'][k]['participant_id'] for k in query_ids}
        assert not (query_people & {samples['base'][k]['participant_id'] for k in fit_ids})
        assert not (query_people & {samples['target'][k]['participant_id'] for ids in supports.values() for k in ids})
        target_ids = sorted(set(query_ids) | {k for ids in supports.values() for k in ids})
        assert target_ids == meta['target_loaded_ids']

        def pool(role, ids):
            vectors = []
            windows = 0
            for key in ids:
                x, a, _ = readers[role].load(key)
                vectors.append(pooled_vector(x, a, 'full_activity', list(range(len(x)))))
                windows += len(x)
            return np.stack(vectors), windows

        x, base_windows = pool('base', fit_ids)
        values, target_windows = pool('target', target_ids)
        assert base_windows == meta['base_windows'] and target_windows == meta['target_loaded_windows']
        fitted, fit_details = fit_shrinkage(x, labels, args.seed)
        assert fit_details['shuffled_labels'] == meta['shuffled_labels']
        assert np.max(np.abs(fitted['mean'] - original_fit['mean'])) <= 1e-12
        whitening_errors = {}
        for family in ('residual','total','shuffled'):
            w = fitted[family + '_whitening']
            cov = fitted[family + '_covariance']
            error = float(np.max(np.abs(w.T @ cov @ w - np.eye(768))))
            if error > c['whitening_identity_atol']:
                raise ValueError('Whitening equation failed')
            whitening_errors[family] = error
        positions = {k:i for i,k in enumerate(target_ids)}
        classes = list(range(100))
        refs = [dict(sample_id=k, label=samples['target'][k]['class_id']) for k in query_ids]
        modes = ('raw','residual_pca') + NEW_MODES
        rows, metrics, dcg_by_query = [], {}, {}
        distance_rebuilds = 0
        for mode in modes:
            z = apply_geometry(values, original_fit, mode) if mode in ('raw','residual_pca') else transformed(values, fitted, mode)
            metrics[mode] = {}
            seed_dcg = []
            for seed, support in supports.items():
                q = z[[positions[k] for k in query_ids]]
                g = z[[positions[k] for k in support]]
                scores = q @ g.T if mode in ('raw','residual_pca') else lookup_scores(q, g, mode)
                predictions = [dict(sample_id=k, ranking=ranked_classes(v.tolist(),classes)) for k,v in zip(query_ids,scores)]
                measured = evaluate(predictions,refs,classes)
                metrics[mode][seed] = measured
                entries = []
                for qi,p in enumerate(predictions):
                    key = p['sample_id']
                    # All whitened cosine/Euclidean rows also use direct pair differences.
                    distances = np.sum((g-q[qi])**2,axis=1)
                    reference = np.lexsort((np.arange(100),distances)).tolist()
                    if p['ranking'] != reference:
                        raise ValueError('Direct-distance full ranking mismatch')
                    distance_rebuilds += 1
                    if mode in ('raw','residual_pca') and p['ranking'] != old_predictions[mode,seed,key]:
                        raise ValueError('Changed original geometry ranking')
                    entry = dict(**p,label=samples['target'][key]['class_id'],participant_id=samples['target'][key]['participant_id'],
                                 mode=mode,support_seed=int(seed),**score_ranking(p['ranking'],samples['target'][key]['class_id'],classes))
                    rows.append(entry)
                    entries.append(entry)
                seed_dcg.append([r['dcg'] for r in entries])
            dcg_by_query[mode] = np.mean(seed_dcg,axis=0)
        means = {m:float(v.mean()) for m,v in dcg_by_query.items()}
        for mode in ('raw','residual_pca'):
            assert abs(means[mode]-old_report['mean_DCG'][mode]) <= 1e-12
        deltas = {m:means['residual_pca']-means[m] for m in NEW_MODES}
        primary = ('residual_lw_cos','residual_lw_maha')
        nonnegative = {m:all(metrics['residual_pca'][s]['dcg'] >= metrics[m][s]['dcg'] for s in supports) for m in primary}
        passed = min(deltas[m] for m in primary) >= c['PCA_gain_floor'] and all(nonnegative.values())
        participants = [samples['target'][k]['participant_id'] for k in query_ids]
        uncertainty = {m:participant_bootstrap(dcg_by_query['residual_pca']-dcg_by_query[m],participants,c['bootstrap_reps'],args.seed) for m in NEW_MODES}
        conditioned_deltas = {}
        for metric in ('cos','maha'):
            conditioned = 'residual_lw_' + metric
            for family in ('total','shuffled'):
                control = family + '_lw_' + metric
                conditioned_deltas[conditioned + '_minus_' + control] = dict(delta_DCG=means[conditioned]-means[control],
                    conditional_bootstrap=participant_bootstrap(dcg_by_query[conditioned]-dcg_by_query[control],participants,c['bootstrap_reps'],args.seed))
        np.savez(run.output/'fitted_shrinkage.npz',**fitted)
        run.save('fit_support_metadata.json',dict(**meta,shrinkage_fit=fit_details))
        run.save('predictions.json',rows)
        run.save('shrinkage_transfer.json',dict(run_id=args.run_id,passed_PCA_advantage_diagnostic=bool(passed),
            query_count=383,prediction_rows=len(rows),metrics=metrics,mean_DCG=means,PCA_minus_LW_DCG=deltas,
            primary_seed_nonnegative=nonnegative,conditional_participant_bootstrap=uncertainty,
            class_conditioning_deltas=conditioned_deltas,whitening_identity_errors=whitening_errors,
            direct_distance_rankings_equal=distance_rebuilds,parent_raw_PCA_rankings_equal=2298,
            resource=resource,source_hashes=c['input_hashes'],sklearn=sklearn.__version__,numpy=np.__version__,
            GPU_hours=0,new_encoder_calls=0,test_access='none',base_dev_feature_access='none',
            target_unused_train_feature_access='none',
            interpretation_scope='same-corpus exploratory standard covariance controls; extra labeled base access; no SimpleCNAPS/LFD/FROST full reproduction, new method, population, optimal-shrinkage or causal claim'))
        run.status['output_hashes'] = {n:file_sha(run.output/n) for n in ('fitted_shrinkage.npz','fit_support_metadata.json','predictions.json','shrinkage_transfer.json')}
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
            raise ValueError('Unexpected CUDA')


if __name__ == '__main__':
    main()
