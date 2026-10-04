"""H005_v1 train-fit/held-gloss dev retrieval with prespecified geometry controls."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT))
from cache import CacheReader,file_sha,sample_map
from run_context import RunContext
from resources import scratch_preflight


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--seed',required=True,type=int)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    c=json.loads(args.config.read_text())
    with RunContext(args.output_dir,args.run_id,args.config,args.seed) as run:
        import numpy as np
        from participant_probe import centroid_predictions,gloss_folds
        from participant_removal import MODES,fit_removal,transform
        from window_budget import pooled_vector
        from metrics import evaluate,ranked_classes,score_ranking
        from analysis import participant_bootstrap
        for path,expected in c['input_hashes'].items():
            if file_sha(REPO/path)!=expected:raise ValueError('Changed registered input '+path)
        resource=scratch_preflight('/dev/shm',64<<20)
        prior=json.loads((REPO/c['cache_identity_config']).read_text())
        cache=json.loads((REPO/c['cache_report']).read_text())
        if not cache['passed'] or file_sha(cache['cache_manifest'])!=cache['cache_manifest_sha256']:
            raise ValueError('Unavailable or changed native cache')
        cohort=json.loads((REPO/prior['cohort_manifest']).read_text())
        audit=json.loads((REPO/prior['video_audit']).read_text())
        indexed=sample_map(cohort)
        reader=CacheReader(cache['cache_manifest'],prior['cache_identity'],cohort,audit)
        ids=sorted(indexed)
        if len(ids)!=1691:raise ValueError('Cohort differs')
        vectors=[]
        for key in ids:
            x,a,_=reader.load(key)
            vectors.append(pooled_vector(x,a,'full_activity',list(range(len(x)))))
        x=np.stack(vectors)
        positions={key:i for i,key in enumerate(ids)}
        people=[indexed[k]['participant_id'] for k in ids]
        train=[i for i,k in enumerate(ids) if indexed[k]['split']=='train']
        query_ids=sorted(k for k in ids if indexed[k]['split']=='val')
        if len(train)!=1313 or len(query_ids)!=378:raise ValueError('Coverage differs')
        all_predictions=[]
        fold_metadata=[]
        fitted_arrays={}
        for fold,fit_glosses in enumerate(gloss_folds(cohort['classes'],args.seed)):
            fit_set=set(fit_glosses)
            classes=sorted(i for i,g in enumerate(cohort['classes']) if g not in fit_set)
            fit=[i for i in train if indexed[ids[i]]['gloss'] in fit_set]
            eval_train=[i for i in train if indexed[ids[i]]['gloss'] not in fit_set]
            queries=[k for k in query_ids if indexed[k]['class_id'] in classes]
            fit_hashes={indexed[ids[i]]['source_sha256'] for i in fit}
            if fit_hashes & {indexed[k]['source_sha256'] for k in queries+[ids[i] for i in eval_train]}:
                raise ValueError('Identical source across fit/eval folds')
            f=fit_removal(x[fit],[indexed[ids[i]]['class_id'] for i in fit],
                          [people[i] for i in fit],c['rank'],args.seed+fold)
            fitted_arrays[f'fold{fold}_mean']=f['mean']
            for mode in ('participant','residual_pca','random'):
                fitted_arrays[f'fold{fold}_{mode}']=f[mode]
            metadata=dict(fold=fold,fit_glosses=fit_glosses,eval_classes=classes,
                          fit_ids=[ids[i] for i in fit],eval_train_count=len(eval_train),
                          query_ids=queries,fit_count=len(fit),participant_singular=f['participant_singular'].tolist(),
                          residual_singular=f['residual_singular'].tolist(),identity_probe={},supports={})
            for mode in MODES:
                transformed=transform(x,f,mode)
                id_probe=centroid_predictions(transformed,people,fit,eval_train)
                metadata['identity_probe'][mode]={k:v for k,v in id_probe.items() if k!='predictions'}
                for seed in c['support_seeds']:
                    support=[cohort['supports'][str(seed)]['1'][cohort['classes'][cl]][0] for cl in classes]
                    if len(support)!=50 or len(set(support))!=50 or any(indexed[k]['split']!='train' for k in support):
                        raise ValueError('One-example gallery differs')
                    if any(indexed[k]['class_id']!=cl for k,cl in zip(support,classes)):
                        raise ValueError('Support class ordering differs')
                    if set(support)&set(queries) or any(indexed[k]['gloss'] in fit_set for k in support):
                        raise ValueError('Support/fit/query overlap')
                    if ({indexed[k]['source_sha256'] for k in support}
                            & {indexed[k]['source_sha256'] for k in queries}):
                        raise ValueError('Identical source across support/query')
                    metadata['supports'][str(seed)]=support
                    scores=transformed[[positions[k] for k in queries]]@transformed[[positions[k] for k in support]].T
                    for key,values in zip(queries,scores):
                        ranking=ranked_classes(values.tolist(),classes)
                        all_predictions.append(dict(sample_id=key,participant_id=indexed[key]['participant_id'],
                            label=indexed[key]['class_id'],fold=fold,mode=mode,support_seed=seed,ranking=ranking,
                            **score_ranking(ranking,indexed[key]['class_id'],classes)))
            fold_metadata.append(metadata)
        metrics={}
        per_query={}
        by_fold={}
        for mode in MODES:
            metrics[mode]={}
            paired=[]
            for seed in c['support_seeds']:
                rows=sorted([r for r in all_predictions if r['mode']==mode and r['support_seed']==seed],key=lambda r:r['sample_id'])
                if [r['sample_id'] for r in rows]!=query_ids:raise ValueError('Incomplete/duplicate query predictions')
                for fold,meta in enumerate(fold_metadata):
                    part=[r for r in rows if r['fold']==fold]
                    by_fold[f'{mode}_s{seed}_f{fold}']=evaluate(part,[{'sample_id':r['sample_id'],'label':r['label']} for r in part],meta['eval_classes'])
                metrics[mode][str(seed)]={name:float(np.mean([r[name] for r in rows])) for name in ('dcg','top1','top5','mrr')}
                paired.append([r['dcg'] for r in rows])
            per_query[mode]=np.mean(paired,axis=0)
        means={mode:float(per_query[mode].mean()) for mode in MODES}
        differences={mode:means['participant']-means[mode] for mode in MODES if mode!='participant'}
        identity={mode:float(np.mean([m['identity_probe'][mode]['scores']['balanced'] for m in fold_metadata])) for mode in MODES}
        identity_drop=identity['center']-identity['participant']
        all_seeds_nonnegative=all(metrics['participant'][str(seed)]['dcg']>=metrics['raw'][str(seed)]['dcg'] for seed in c['support_seeds'])
        passed=(differences['raw']>=c['raw_gain_floor'] and
                min(differences[k] for k in ('center','random','residual_pca'))>=c['control_gain_floor'] and
                identity_drop>=c['identity_drop_floor'] and all_seeds_nonnegative)
        uncertainties={mode:participant_bootstrap(per_query['participant']-per_query[mode],
                       [indexed[k]['participant_id'] for k in query_ids],c['bootstrap_reps'],args.seed)
                       for mode in MODES if mode!='participant'}
        for value in uncertainties.values():
            value['limitation']='five dev participants; conditional on fixed folds/supports/recipe; exploratory, no population or causal claim'
        np.savez(run.output/'fitted_transforms.npz',**fitted_arrays)
        run.save('predictions.json',all_predictions)
        run.save('fold_metadata.json',fold_metadata)
        run.save('removal_diagnostic.json',dict(run_id=args.run_id,passed_diagnostic=bool(passed),rank=c['rank'],
                 query_count=len(query_ids),pairs=len(all_predictions),metrics=metrics,mean_DCG=means,
                 participant_minus_controls_DCG=differences,identity_balanced=identity,
                 identity_center_minus_participant=identity_drop,all_support_seeds_nonnegative=all_seeds_nonnegative,
                 conditional_participant_bootstrap=uncertainties,by_fold=by_fold,resource=resource,
                 cache_manifest_sha256=file_sha(reader.path),source_hashes=c['input_hashes'],
                 numpy=np.__version__,test_access='none',GPU_hours=0,new_encoder_calls=0,
                 interpretation_scope='known projection recipe on reused train/dev; no causal nuisance/novelty/benchmark or latency claim'))
        run.status['output_hashes']={n:file_sha(run.output/n) for n in
            ('fitted_transforms.npz','predictions.json','fold_metadata.json','removal_diagnostic.json')}
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():raise ValueError('Unexpected CUDA')


if __name__=='__main__':main()
