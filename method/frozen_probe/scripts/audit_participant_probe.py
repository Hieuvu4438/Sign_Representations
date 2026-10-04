"""H005_v0 train-only cached diagnostic; never reads test or adapts the encoder."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import CacheReader, file_sha, sample_map
from participant_probe import centroid_predictions, gloss_folds, permute_within_gloss
from resources import scratch_preflight
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import numpy as np
        import sklearn
        from window_budget import pooled_vector
        for path, expected in config['input_hashes'].items():
            if file_sha(REPO/path) != expected:
                raise ValueError(f'Changed registered input: {path}')
        resource = scratch_preflight('/dev/shm', 32 << 20)
        prior = json.loads((REPO/config['cache_identity_config']).read_text())
        report = json.loads((REPO/config['cache_report']).read_text())
        if not report['passed'] or file_sha(report['cache_manifest']) != report['cache_manifest_sha256']:
            raise ValueError('Native cache unavailable or changed')
        cohort = json.loads((REPO/prior['cohort_manifest']).read_text())
        audit = json.loads((REPO/prior['video_audit']).read_text())
        indexed = sample_map(cohort)
        reader = CacheReader(report['cache_manifest'], prior['cache_identity'], cohort, audit)
        ids = sorted(k for k,s in indexed.items() if s['split']=='train')
        people = [indexed[k]['participant_id'] for k in ids]
        glosses = [indexed[k]['gloss'] for k in ids]
        if len(ids)!=config['train_count'] or len(set(people))!=config['participant_count']:
            raise ValueError('Registered train coverage differs')
        folds = gloss_folds(cohort['classes'],args.seed)
        partitions = [([i for i,g in enumerate(glosses) if g in set(fit)],
                       [i for i,g in enumerate(glosses) if g not in set(fit)]) for fit in folds]
        if sorted(i for _,ev in partitions for i in ev)!=list(range(len(ids))):
            raise ValueError('Incomplete/duplicate cross-fit evaluation')
        for fit,ev in partitions:
            if ({indexed[ids[i]]['source_sha256'] for i in fit}
                    & {indexed[ids[i]]['source_sha256'] for i in ev}):
                raise ValueError('Byte-identical video across gloss folds')
        vectors=[]
        windows=0
        for key in ids:
            x,a,_=reader.load(key)
            vectors.append(pooled_vector(x,a,'full_activity',list(range(len(x)))))
            windows+=len(x)
        x=np.stack(vectors)
        if x.shape!=(config['train_count'],768) or not np.isfinite(x).all():
            raise ValueError('Pooled feature contract differs')
        real=[centroid_predictions(x,people,fit,ev) for fit,ev in partitions]
        rng=np.random.default_rng(args.seed)
        controls=[]
        for trial in range(config['permutations']):
            labels=permute_within_gloss(people,glosses,rng)
            results=[centroid_predictions(x,labels,fit,ev) for fit,ev in partitions]
            controls.append(dict(trial=trial,balanced_by_fold=[r['scores']['balanced'] for r in results],
                                 mean_balanced=float(np.mean([r['scores']['balanced'] for r in results]))))
        actual=float(np.mean([r['scores']['balanced'] for r in real]))
        null=[r['mean_balanced'] for r in controls]
        delta=actual-float(np.median(null))
        coverage=[r['candidate_coverage'] for r in real]
        passed=delta>=config['balanced_delta_floor'] and min(coverage)>=config['coverage_floor']
        predictions=[]
        for fold, ((fit,ev),r) in enumerate(zip(partitions,real)):
            for i,p in zip(ev,r['predictions']):
                predictions.append(dict(sample_id=ids[i],gloss=glosses[i],participant_id=people[i],
                                        predicted_participant=p,fold=fold,correct=p==people[i]))
        predictions.sort(key=lambda r:r['sample_id'])
        run.save('predictions.json',predictions)
        run.save('permutation_controls.json',controls)
        run.save('participant_diagnostic.json',dict(run_id=args.run_id,passed_diagnostic=bool(passed),
                 mean_balanced=actual,null_median=float(np.median(null)),balanced_delta=delta,
                 null_range=[float(min(null)),float(max(null))],
                 null_empirical_025_975=np.quantile(null,[.025,.975]).tolist(),
                 folds=[dict(fit_glosses=folds[i],eval_glosses=folds[1-i],fit_count=len(fit),
                             eval_count=len(ev),**{k:v for k,v in r.items() if k!='predictions'})
                        for i,((fit,ev),r) in enumerate(zip(partitions,real))],
                 train_count=len(ids),participant_count=len(set(people)),cached_windows=windows,
                 cache_manifest_sha256=file_sha(reader.path),source_hashes=config['input_hashes'],
                 resource=resource,packages={'numpy':np.__version__,'sklearn':sklearn.__version__},
                 GPU_hours=0,new_encoder_calls=0,test_access='none',dev_feature_access='none',
                 uncertainty_scope='descriptive conditional permutation controls; no p-value or population CI',
                 interpretation_scope='participant predictability only, not harmful nuisance or retrieval improvement'))
        run.status['output_hashes']={name:file_sha(run.output/name) for name in
                                    ('predictions.json','permutation_controls.json','participant_diagnostic.json')}
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
            raise ValueError('Unexpected CUDA initialization')


if __name__=='__main__': main()
