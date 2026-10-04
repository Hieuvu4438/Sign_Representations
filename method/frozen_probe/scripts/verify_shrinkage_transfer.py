"""Independent fourth-moment covariance and original-space metric reconstruction."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import CacheReader, file_sha
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import numpy as np
        for name, digest in c['input_hashes'].items():
            if file_sha(REPO/name) != digest:
                raise ValueError('Changed verification input ' + name)
        measured = REPO/c['measured_runtime']
        status = json.loads((measured/'status.json').read_text())
        assert status['status'] == 'completed' and status['exit_code'] == 0
        for name,digest in status['output_hashes'].items():
            assert file_sha(measured/name) == digest
        config = json.loads((measured/'config.json').read_text())
        for name,digest in config['input_hashes'].items():
            assert file_sha(REPO/name) == digest
        parent = REPO/config['parent_runtime']
        parent_config = json.loads((parent/'config.json').read_text())
        for name,digest in parent_config['input_hashes'].items():
            assert file_sha(REPO/name) == digest
        meta = json.loads((measured/'fit_support_metadata.json').read_text())
        fitted = dict(np.load(measured/'fitted_shrinkage.npz',allow_pickle=False))
        report = json.loads((measured/'shrinkage_transfer.json').read_text())
        rows = json.loads((measured/'predictions.json').read_text())
        indexed = {(r['mode'],str(r['support_seed']),r['sample_id']):r for r in rows}
        assert len(indexed) == len(rows) == 9192
        parent_meta = json.loads((parent/'fit_support_metadata.json').read_text())
        assert {k:meta[k] for k in parent_meta} == parent_meta
        matrices, samples = {}, {}
        for role,ids in [('base',meta['fit_ids']),('target',meta['target_loaded_ids'])]:
            prior = json.loads((REPO/parent_config[role+'_identity_config']).read_text())
            cache = json.loads((REPO/parent_config[role+'_cache_report']).read_text())
            cohort = json.loads((REPO/prior['cohort_manifest']).read_text())
            audit = json.loads((REPO/prior['video_audit']).read_text())
            samples[role] = {s['sample_id']:s for s in cohort['samples']}
            assert file_sha(cache['cache_manifest']) == meta[role+'_manifest_sha256']
            reader = CacheReader(cache['cache_manifest'],prior['cache_identity'],cohort,audit)
            pooled = []
            for key in ids:
                x,activity,_ = reader.load(key)
                x = x.astype(np.float64)
                weights = np.clip(activity.astype(np.float64),0,1).max(axis=1)
                value = x.mean(axis=0) if weights.sum() <= 1e-12 else np.sum(weights[:,None]*x,axis=0)/weights.sum()
                assert np.linalg.norm(value) > 1e-12
                pooled.append(value/np.linalg.norm(value))
            matrices[role] = np.stack(pooled)
        x = matrices['base']
        labels = [samples['base'][k]['class_id'] for k in meta['fit_ids']]
        assert labels == meta['fit_labels'] and len(labels) == 1313
        assert meta['shrinkage_fit']['shuffled_labels'] == np.random.default_rng(1701).permutation(labels).tolist()
        assert np.max(np.abs(x.mean(axis=0)-fitted['mean'])) <= c['numeric_atol']
        values = matrices['target']-fitted['mean']
        positions = {key:i for i,key in enumerate(meta['target_loaded_ids'])}
        queries = meta['query_ids']
        query = values[[positions[k] for k in queries]]
        errors, rebuilt = {}, 0
        for family in ('residual','total','shuffled'):
            r = x.copy()
            if family == 'total':
                r -= x.mean(axis=0)
            else:
                actual_labels = labels if family == 'residual' else meta['shrinkage_fit']['shuffled_labels']
                for label in set(actual_labels):
                    ids = np.where(np.asarray(actual_labels)==label)[0]
                    r[ids] -= x[ids].mean(axis=0)
            n,d = r.shape
            empirical = r.T@r/n
            tau = np.trace(empirical)/d
            delta = np.sum((empirical-tau*np.eye(d))**2)/d
            fourth = np.mean(np.sum(r*r,axis=1)**2)
            beta = (fourth-np.sum(empirical**2))/(n*d)
            coefficient = 0.0 if beta == 0 else min(beta,delta)/delta
            covariance = (1-coefficient)*empirical+coefficient*tau*np.eye(d)
            cov_error = float(np.max(np.abs(covariance-fitted[family+'_covariance'])))
            assert cov_error <= c['numeric_atol']
            assert abs(coefficient-meta['shrinkage_fit']['families'][family]['shrinkage']) <= c['numeric_atol']
            precision = np.linalg.solve(covariance,np.eye(d))
            assert np.max(np.abs(covariance@precision-np.eye(d))) <= c['inverse_identity_atol']
            errors[family] = dict(covariance_max_abs_error=cov_error,shrinkage=coefficient)
            qp = query@precision
            qnorm = np.sum(qp*query,axis=1)
            for seed,support in meta['supports'].items():
                gallery = values[[positions[k] for k in support]]
                gp = gallery@precision
                gnorm = np.sum(gp*gallery,axis=1)
                cross = qp@gallery.T
                metrics = {'cos':cross/np.sqrt(qnorm[:,None]*gnorm[None,:]),
                           'maha':-(qnorm[:,None]+gnorm[None,:]-2*cross)}
                for metric,scores in metrics.items():
                    mode = family+'_lw_'+metric
                    for i,key in enumerate(queries):
                        ranking = np.lexsort((np.arange(100),-scores[i])).tolist()
                        assert ranking == indexed[mode,seed,key]['ranking']
                        rebuilt += 1
        # Recount every row, including both unchanged prior references.
        group_metrics = {}
        for row in rows:
            assert row['label'] == samples['target'][row['sample_id']]['class_id']
            assert row['participant_id'] == samples['target'][row['sample_id']]['participant_id']
            assert sorted(row['ranking']) == list(range(100))
            rank = row['ranking'].index(row['label'])+1
            metrics = {'dcg':1/math.log2(rank+1),'mrr':1/rank,
                       **{'top'+str(k):float(rank<=k) for k in (1,5,10,20)}}
            assert all(abs(row[k]-v)<=c['numeric_atol'] for k,v in metrics.items())
            group_metrics.setdefault((row['mode'],str(row['support_seed'])),[]).append(metrics)
        assert len(group_metrics) == 24 and all(len(v)==383 for v in group_metrics.values())
        for (mode,seed),entries in group_metrics.items():
            for k in entries[0]:
                assert abs(math.fsum(r[k] for r in entries)/383-report['metrics'][mode][seed][k]) <= c['numeric_atol']
        means = {mode:math.fsum(report['metrics'][mode][s]['dcg'] for s in meta['supports'])/3 for mode in report['metrics']}
        assert all(abs(means[m]-report['mean_DCG'][m])<=c['numeric_atol'] for m in means)
        primary = ('residual_lw_cos','residual_lw_maha')
        passed = all(means['residual_pca']-means[m]>=config['PCA_gain_floor'] and
                     all(report['metrics']['residual_pca'][s]['dcg']>=report['metrics'][m][s]['dcg'] for s in meta['supports']) for m in primary)
        assert passed == report['passed_PCA_advantage_diagnostic']
        run.save('verification.json',dict(passed=True,recounted_rows=9192,original_space_LW_rankings_equal=rebuilt,
                 independent_fourth_moment_fit_errors=errors,input_output_digests_verified=True,
                 parent_fit_support_contract_equal=True,mean_DCG=means,GPU_hours=0,
                 scope='independent arithmetic implementation by same agent; no new recipe, trial, label access or independent scientific review'))
        run.status['output_hashes']={'verification.json':file_sha(run.output/'verification.json')}


if __name__=='__main__':
    main()
