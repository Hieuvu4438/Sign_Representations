"""H006: base-only fit, fixed new-class support galleries and complete dev lookup."""
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
        from geometry_transfer import MODES,fit_geometry,apply_geometry,disjoint_cohorts,common_feature_recipe
        from window_budget import pooled_vector
        from metrics import evaluate,ranked_classes,score_ranking
        from analysis import participant_bootstrap
        for path,expected in c['input_hashes'].items():
            if file_sha(REPO/path)!=expected:raise ValueError('Changed registered input '+path)
        resource=scratch_preflight('/dev/shm',64<<20)
        cohorts,readers,priors={},{},{}
        for role in ('base','target'):
            prior=json.loads((REPO/c[role+'_identity_config']).read_text())
            cache=json.loads((REPO/c[role+'_cache_report']).read_text())
            if not cache['passed'] or file_sha(cache['cache_manifest'])!=cache['cache_manifest_sha256']:
                raise ValueError('Unavailable or changed '+role+' native cache')
            cohorts[role]=json.loads((REPO/prior['cohort_manifest']).read_text())
            audit=json.loads((REPO/prior['video_audit']).read_text())
            readers[role]=CacheReader(cache['cache_manifest'],prior['cache_identity'],cohorts[role],audit)
            priors[role]=prior
        disjoint_cohorts(cohorts['base'],cohorts['target'])
        common_feature_recipe(priors['base']['cache_identity'],priors['target']['cache_identity'])
        base=sample_map(cohorts['base']);target=sample_map(cohorts['target'])
        fit_ids=sorted(k for k,s in base.items() if s['split']=='train')
        query_ids=sorted(k for k,s in target.items() if s['split']=='val')
        if len(fit_ids)!=1313 or len(query_ids)!=383 or len(cohorts['target']['classes'])!=100:
            raise ValueError('Registered coverage differs')
        if {base[k]['participant_id'] for k in fit_ids}&{target[k]['participant_id'] for k in query_ids}:
            raise ValueError('Base-fit/query participant overlap')
        supports={}
        for seed in c['support_seeds']:
            support=[cohorts['target']['supports'][str(seed)]['1'][g][0] for g in cohorts['target']['classes']]
            if (len(support)!=100 or len(set(support))!=100
                    or any(target[k]['split']!='train' or target[k]['class_id']!=cl for cl,k in enumerate(support))):
                raise ValueError('Gallery identity/order differs')
            if ({target[k]['participant_id'] for k in support}&{target[k]['participant_id'] for k in query_ids}
                    or {target[k]['source_sha256'] for k in support}&{target[k]['source_sha256'] for k in query_ids}):
                raise ValueError('Support/query participant/source overlap')
            supports[str(seed)]=support
        target_ids=sorted(set(query_ids)|{k for support in supports.values() for k in support})
        def pool(reader,ids):
            vectors=[];windows=0
            for key in ids:
                x,a,_=reader.load(key);vectors.append(pooled_vector(x,a,'full_activity',list(range(len(x)))));windows+=len(x)
            return np.stack(vectors),windows
        x,base_windows=pool(readers['base'],fit_ids)
        values,target_windows=pool(readers['target'],target_ids)
        labels=[base[k]['class_id'] for k in fit_ids]
        f=fit_geometry(x,labels,c['rank'],args.seed)
        positions={k:i for i,k in enumerate(target_ids)}
        classes=list(range(100))
        refs=[dict(sample_id=k,label=target[k]['class_id']) for k in query_ids]
        rows=[];metrics={};dcg_by_query={}
        for mode in MODES:
            transformed=apply_geometry(values,f,mode)
            metrics[mode]={};seed_dcg=[]
            for seed,support in supports.items():
                scores=transformed[[positions[k] for k in query_ids]]@transformed[[positions[k] for k in support]].T
                predictions=[dict(sample_id=k,ranking=ranked_classes(v.tolist(),classes)) for k,v in zip(query_ids,scores)]
                measured=evaluate(predictions,refs,classes)
                metrics[mode][seed]=measured
                entries=[]
                for p in predictions:
                    k=p['sample_id'];entry=dict(**p,label=target[k]['class_id'],participant_id=target[k]['participant_id'],
                        mode=mode,support_seed=int(seed),**score_ranking(p['ranking'],target[k]['class_id'],classes))
                    entries.append(entry);rows.append(entry)
                seed_dcg.append([r['dcg'] for r in entries])
            dcg_by_query[mode]=np.mean(seed_dcg,axis=0)
        means={mode:float(v.mean()) for mode,v in dcg_by_query.items()}
        deltas={mode:means['residual_pca']-means[mode] for mode in MODES if mode!='residual_pca'}
        seed_nonnegative=all(metrics['residual_pca'][s]['dcg']>=metrics['raw'][s]['dcg'] for s in supports)
        passed=(deltas['raw']>=c['raw_gain_floor'] and
                min(deltas[m] for m in ('center','random','total_pca','shuffled_pca'))>=c['control_gain_floor'] and seed_nonnegative)
        uncertainty={mode:participant_bootstrap(dcg_by_query['residual_pca']-dcg_by_query[mode],
            [target[k]['participant_id'] for k in query_ids],c['bootstrap_reps'],args.seed) for mode in deltas}
        for item in uncertainty.values():
            item['limitation']='same-corpus reused five dev participants; fixed bases/labels/gallery/rank; exploratory, no population or causal inference'
        arrays={k:v for k,v in f.items() if isinstance(v,np.ndarray)}
        np.savez(run.output/'fitted_geometry.npz',**arrays)
        run.save('predictions.json',rows)
        run.save('fit_support_metadata.json',dict(fit_ids=fit_ids,fit_labels=labels,shuffled_labels=f['shuffled_labels'],
                 query_ids=query_ids,target_loaded_ids=target_ids,supports=supports,base_windows=base_windows,
                 target_loaded_windows=target_windows,base_manifest_sha256=file_sha(readers['base'].path),
                 target_manifest_sha256=file_sha(readers['target'].path)))
        run.save('geometry_transfer.json',dict(run_id=args.run_id,passed_diagnostic=bool(passed),
                 query_count=len(query_ids),prediction_rows=len(rows),rank=c['rank'],metrics=metrics,mean_DCG=means,
                 residual_minus_controls_DCG=deltas,all_support_seeds_nonnegative=seed_nonnegative,
                 conditional_participant_bootstrap=uncertainty,resource=resource,source_hashes=c['input_hashes'],
                 GPU_hours=0,new_encoder_calls=0,test_access='none',base_dev_feature_access='none',
                 target_unused_train_feature_access='none',numpy=np.__version__,
                 interpretation_scope='class-disjoint same-corpus cached diagnostic, known projection family and extra base labels; no novelty/independent-population/causal/latency claim'))
        run.status['output_hashes']={n:file_sha(run.output/n) for n in
            ('fitted_geometry.npz','predictions.json','fit_support_metadata.json','geometry_transfer.json')}
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():raise ValueError('Unexpected CUDA')


if __name__=='__main__':main()
