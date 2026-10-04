"""H006_v2 fixed-factorial cached-feature diagnostic, bounded CPU artifacts."""
import argparse
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from bounded_cpu_context import BoundedCPUContext
from cache import CacheReader, file_sha
from axis_spectrum import factor, transforms, unit_rows, MODES
from geometry_transfer import disjoint_cohorts, common_feature_recipe
from window_budget import pooled_vector
from metrics import evaluate, ranked_classes, score_ranking
from analysis import participant_bootstrap
from compact_geometry_artifacts import ranking_metadata, pack_predictions
from resources import scratch_preflight


def main():
    parser = argparse.ArgumentParser()
    for option in ('config', 'run-id', 'output-dir'):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    with BoundedCPUContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        c = run.config
        if c.get('software_only'):
            suite = unittest.defaultTestLoader.discover('method/frozen_probe/tests', pattern='test_axis_spectrum.py')
            result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
            run.save('axis_tests.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
                                            failures=len(result.failures), errors=len(result.errors)))
            if not result.wasSuccessful(): raise ValueError('Axis/spectrum oracles failed')
            return
        resource = scratch_preflight('/dev/shm', 128 << 20)
        parent = Path(c['parent_runtime'])
        parent_c = json.loads((parent / 'config.json').read_text())
        ancestor = Path(parent_c['parent_runtime'])
        ancestor_c = json.loads((ancestor / 'config.json').read_text())
        for root, config in ((parent, parent_c), (ancestor, ancestor_c)):
            status = json.loads((root / 'status.json').read_text())
            if status['status'] != 'completed': raise ValueError('Parent incomplete')
            for name, sha in status['output_hashes'].items():
                if file_sha(root/name) != sha: raise ValueError('Changed parent output')
            for name, sha in config['input_hashes'].items():
                if file_sha(name) != sha: raise ValueError('Changed parent input')
        meta = json.loads((parent/'fit_support_metadata.json').read_text())
        fitted = dict(np.load(parent/'fitted_shrinkage.npz', allow_pickle=False))
        prior_rows = json.loads((parent/'predictions.json').read_text())
        old_ranks = {(r['mode'], r['support_seed'], r['sample_id']): r['ranking'] for r in prior_rows}
        bases, spectra, checks = {}, {}, {}
        for letter, family in (('R','residual'), ('T','total')):
            spectra[letter], bases[letter], checks[letter] = factor(
                fitted[family+'_covariance'], c['relative_gap_floor'], c['factor_atol'])
        w = transforms(bases, spectra)
        for mode, family in (('RR','residual'), ('TT','total')):
            if np.max(np.abs(w[mode]-fitted[family+'_whitening'])) > c['reference_whitening_atol']:
                raise ValueError('Original whitening changed')
        # Base parameter provenance is bound; base arrays need not be reloaded/refit.
        cohorts, priors, samples, readers = {}, {}, {}, {}
        for role in ('base', 'target'):
            prior = json.loads(Path(ancestor_c[role+'_identity_config']).read_text())
            cache = json.loads(Path(ancestor_c[role+'_cache_report']).read_text())
            if not cache['passed'] or file_sha(cache['cache_manifest']) != meta[role+'_manifest_sha256']:
                raise ValueError('Missing/changed native cache')
            cohort = json.loads(Path(prior['cohort_manifest']).read_text())
            audit = json.loads(Path(prior['video_audit']).read_text())
            cohorts[role], priors[role] = cohort, prior
            samples[role] = {s['sample_id']: s for s in cohort['samples']}
            readers[role] = CacheReader(cache['cache_manifest'], prior['cache_identity'], cohort, audit)
        disjoint_cohorts(cohorts['base'],cohorts['target'])
        common_feature_recipe(priors['base']['cache_identity'],priors['target']['cache_identity'])
        fit_ids = sorted(k for k,s in samples['base'].items() if s['split']=='train')
        query_ids = sorted(k for k,s in samples['target'].items() if s['split']=='val')
        assert fit_ids == meta['fit_ids'] and len(fit_ids)==1313 and query_ids==meta['query_ids'] and len(query_ids)==383
        assert [samples['base'][k]['class_id'] for k in fit_ids] == meta['fit_labels']
        support_seeds = [1701,1702,1703]; supports = meta['supports']; classes = list(range(100))
        assert set(supports)==set(map(str,support_seeds))
        for seed in support_seeds:
            ids = supports[str(seed)]
            assert ids == [cohorts['target']['supports'][str(seed)]['1'][g][0] for g in cohorts['target']['classes']]
            assert [samples['target'][k]['class_id'] for k in ids]==classes
            assert all(samples['target'][k]['split']=='train' for k in ids)
        people = {samples['target'][k]['participant_id'] for k in query_ids}
        assert not people & {samples['base'][k]['participant_id'] for k in fit_ids}
        assert not people & {samples['target'][k]['participant_id'] for ids in supports.values() for k in ids}
        target_ids = sorted(set(query_ids)|{k for ids in supports.values() for k in ids})
        assert target_ids == meta['target_loaded_ids'] and len(target_ids)==661
        vectors = []; windows = 0
        for key in target_ids:
            x,a,_ = readers['target'].load(key)
            vectors.append(pooled_vector(x,a,'full_activity',list(range(len(x))))); windows+=len(x)
        assert windows==meta['target_loaded_windows']
        values=np.stack(vectors); positions={key:i for i,key in enumerate(target_ids)}
        rows,metrics,dcg={}, {}, {}; reference_equal=0; distance_equal=0
        flat=[]; refs=[dict(sample_id=k,label=samples['target'][k]['class_id']) for k in query_ids]
        for mode in MODES:
            z=unit_rows((values-fitted['mean'])@w[mode]); metrics[mode]={}; by_seed=[]
            for seed in support_seeds:
                q=z[[positions[k] for k in query_ids]]; g=z[[positions[k] for k in supports[str(seed)]]]
                scores=q@g.T; predictions=[]; entries=[]
                for qi,key in enumerate(query_ids):
                    ranking=ranked_classes(scores[qi].tolist(),classes)
                    direct=np.lexsort((np.arange(100),np.sum((g-q[qi])**2,axis=1))).tolist()
                    if ranking!=direct: raise ValueError('Direct-distance ranking differs')
                    distance_equal+=1
                    if mode in ('RR','TT'):
                        family='residual' if mode=='RR' else 'total'
                        if ranking!=old_ranks[(family+'_lw_cos',seed,key)]: raise ValueError('Original fullrank differs')
                        reference_equal+=1
                    p=dict(sample_id=key,ranking=ranking); predictions.append(p)
                    entry=dict(**p,label=samples['target'][key]['class_id'],participant_id=samples['target'][key]['participant_id'],
                               mode=mode,support_seed=seed,**score_ranking(ranking,samples['target'][key]['class_id'],classes))
                    entries.append(entry);flat.append(entry)
                metrics[mode][str(seed)]=evaluate(predictions,refs,classes)
                by_seed.append([e['dcg'] for e in entries])
            dcg[mode]=np.mean(by_seed,axis=0)
        labels=[samples['target'][k]['class_id'] for k in query_ids]; participants=[samples['target'][k]['participant_id'] for k in query_ids]
        shared=ranking_metadata(classes,list(MODES),support_seeds,query_ids,labels,participants)
        packed=pack_predictions(flat,shared)
        deltas={a+'_minus_'+b:dict(delta_DCG=float(np.mean(dcg[a]-dcg[b])),
                    conditional_bootstrap=participant_bootstrap(dcg[a]-dcg[b],participants,2000,args.seed))
                for a,b in (('RR','TR'),('RT','TT'),('RR','RT'),('TR','TT'))}
        per_seed={str(s):metrics['RR'][str(s)]['dcg']-metrics['TR'][str(s)]['dcg'] for s in support_seeds}
        passed=deltas['RR_minus_TR']['delta_DCG']>=c['primary_gain_floor'] and min(per_seed.values())>=0
        arrays=dict(packed,**{'basis_R.npy':bases['R'],'basis_T.npy':bases['T'],
            'spectrum_R.npy':spectra['R'],'spectrum_T.npy':spectra['T'],'mean.npy':fitted['mean'],'target_vectors.npy':values})
        for name,array in arrays.items():run.save_array(name,array,c['array_specs'][name])
        run.save('shared_metadata.json',dict(ranking=shared,target_loaded_ids=target_ids,supports=supports,
                 source_fit_metadata_path=str(parent/'fit_support_metadata.json'),native_recipe='unchanged full_activity FP64 pool/L2',
                 base_arrays_loaded=0,target_vectors_count=len(values),target_windows_loaded=windows))
        run.save('axis_spectrum.json',dict(passed_orientation_diagnostic=bool(passed),mean_DCG={m:float(dcg[m].mean()) for m in MODES},
                 metrics=metrics,deltas=deltas,primary_per_seed_DCG=per_seed,eigensystem_checks=checks,
                 parent_reference_rankings_equal=reference_equal,direct_distance_rankings_equal=distance_equal,
                 resource=resource,GPU_hours=0,new_encoder_calls=0,test_access='none',
                 scope='fixed same-corpus exploratory computational factorial; extra labeled base fit; no novel method, unseen-pretraining, independent confirmation or linguistic causality'))


if __name__=='__main__':main()
