"""Independent original-space metric and saved rank/aggregate recount."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from bounded_cpu_context import BoundedCPUContext
from bounded_artifacts import verify_store
from cache import file_sha


def main():
    parser=argparse.ArgumentParser()
    for option in ('config','run-id','output-dir'):parser.add_argument('--'+option,required=True)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    with BoundedCPUContext(args.output_dir,args.run_id,args.config,args.seed) as run:
        root=Path(run.config['measured_runtime']); c=json.loads((root/'config.json').read_text())
        status=json.loads((root/'status.json').read_text())
        assert status['status']=='completed' and status['exit_code']==0 and status['GPU_hours']==0
        checked=verify_store(root)
        manifest=json.loads((root/'artifact_manifest.json').read_text())
        assert manifest['plan']['file_caps']=={**c['artifact_file_caps'],'artifact_manifest.json':c['artifact_manifest_cap']}
        assert manifest['plan']['ceiling_bytes']==c['artifact_ceiling_bytes']==20971520
        assert file_sha(root/'config.json')==status['config_sha256']
        for path,digest in c['input_hashes'].items():assert file_sha(path)==digest,path
        for path,digest in status['output_hashes'].items():assert file_sha(root/path)==digest,path
        report=json.loads((root/'axis_spectrum.json').read_text()); meta=json.loads((root/'shared_metadata.json').read_text())
        shared=meta['ranking']; load=lambda n:np.load(root/n,allow_pickle=False)
        ranks=load('ranks.npy'); metrics=load('row_metrics.npy'); values=load('target_vectors.npy')-load('mean.npy')
        assert ranks.shape==(4596,100) and ranks.dtype==np.dtype('<i2') and metrics.shape==(4596,6)
        assert shared['modes']==['RR','TR','RT','TT'] and shared['support_seeds']==[1701,1702,1703]
        assert shared['class_ids']==list(range(100)) and len(shared['query_ids'])==383
        assert shared['metric_columns']==['dcg','top1','top5','top10','top20','mrr']
        assert np.array_equal(np.sort(ranks,axis=1),np.broadcast_to(np.arange(100),ranks.shape))
        positions={k:i for i,k in enumerate(meta['target_loaded_ids'])}; assert len(positions)==len(values)==661
        parent=Path(c['parent_runtime']); parentmeta=json.loads((parent/'fit_support_metadata.json').read_text())
        assert meta['target_loaded_ids']==parentmeta['target_loaded_ids'] and meta['supports']==parentmeta['supports']
        parentconfig=json.loads((parent/'config.json').read_text())
        ancestorconfig=json.loads((Path(parentconfig['parent_runtime'])/'config.json').read_text())
        prior=json.loads(Path(ancestorconfig['target_identity_config']).read_text())
        cohort=json.loads(Path(prior['cohort_manifest']).read_text()); samples={s['sample_id']:s for s in cohort['samples']}
        assert shared['query_ids']==sorted(k for k,s in samples.items() if s['split']=='val')
        assert shared['labels']==[samples[k]['class_id'] for k in shared['query_ids']]
        assert shared['participants']==[samples[k]['participant_id'] for k in shared['query_ids']]
        original={(r['mode'],r['support_seed'],r['sample_id']):r['ranking'] for r in json.loads((parent/'predictions.json').read_text())}
        q=values[[positions[k] for k in shared['query_ids']]]
        dcg={}; index=0; reference_rows=0; row_metric_error=0.
        for mode in shared['modes']:
            v=load('basis_'+mode[0]+'.npy'); lam=load('spectrum_'+mode[1]+'.npy')
            # No inverse-square-root transform or unit-vector implementation reused.
            inverse=(v/lam)@v.T
            qmetric=q@inverse; qnorm=np.sqrt(np.einsum('ij,ij->i',qmetric,q))
            by_seed=[]
            for seed in shared['support_seeds']:
                ids=meta['supports'][str(seed)]; g=values[[positions[k] for k in ids]]
                assert [samples[k]['class_id'] for k in ids]==list(range(100))
                gnorm=np.sqrt(np.einsum('ij,ij->i',g@inverse,g))
                scores=qmetric@g.T/(qnorm[:,None]*gnorm[None,:])
                measured=[]
                for qi,key in enumerate(shared['query_ids']):
                    ordered=sorted(range(100),key=lambda cl:(-float(scores[qi,cl]),cl))
                    if ordered!=ranks[index].tolist():raise ValueError('Original-space ranking mismatch '+mode+' '+str(seed)+' '+key)
                    if mode in ('RR','TT'):
                        family='residual' if mode=='RR' else 'total'
                        assert ordered==original[(family+'_lw_cos',seed,key)];reference_rows+=1
                    rank=ordered.index(shared['labels'][qi])+1
                    row=[1/math.log2(rank+1),float(rank<=1),float(rank<=5),float(rank<=10),float(rank<=20),1/rank]
                    error=float(np.max(np.abs(np.asarray(row)-metrics[index])))
                    row_metric_error=max(row_metric_error,error);assert error<=1e-12
                    measured.append(row);index+=1
                a=np.asarray(measured);by_seed.append(a[:,0])
                for column,name in enumerate(shared['metric_columns']):
                    mean=math.fsum(row[column] for row in measured)/383
                    assert abs(mean-report['metrics'][mode][str(seed)][name])<=1e-12
            dcg[mode]=np.mean(by_seed,axis=0)
            assert abs(math.fsum(dcg[mode])/383-report['mean_DCG'][mode])<=1e-12
        assert index==4596 and reference_rows==2298
        groups=sorted(set(shared['participants'])); people=np.asarray(shared['participants'])
        for a,b in (('RR','TR'),('RT','TT'),('RR','RT'),('TR','TT')):
            differences=dcg[a]-dcg[b]; delta=float(np.mean(differences)); saved=report['deltas'][a+'_minus_'+b]
            assert abs(delta-saved['delta_DCG'])<=1e-12
            rng=np.random.default_rng(1701); counts={g:int(np.count_nonzero(people==g)) for g in groups}
            sums={g:math.fsum(differences[people==g]) for g in groups}; draws=[]
            for _ in range(2000):
                chosen=[groups[i] for i in rng.integers(0,len(groups),size=len(groups))]
                draws.append(math.fsum(sums[g] for g in chosen)/sum(counts[g] for g in chosen))
            interval=np.quantile(draws,[.025,.975])*100
            assert np.max(np.abs(interval-saved['conditional_bootstrap']['interval95_pp']))<=1e-10
        per_seed={str(s):report['metrics']['RR'][str(s)]['dcg']-report['metrics']['TR'][str(s)]['dcg'] for s in shared['support_seeds']}
        passed=np.mean(dcg['RR']-dcg['TR'])>=c['primary_gain_floor'] and min(per_seed.values())>=0
        assert bool(passed)==report['passed_orientation_diagnostic']
        assert per_seed==report['primary_per_seed_DCG']
        run.save('axis_verification.json',dict(passed=True,original_space_fullrank_rows=index,parent_reference_rows=reference_rows,
                 maximum_row_metric_error=row_metric_error,all_metrics_deltas_intervals_primary_gate_verified=True,
                 measured_store=checked,registered_file_caps_match=True,all_source_config_output_bindings_verified=True,
                 GPU_hours=0,new_encoder_calls=0,scope='same-agent independent original-space numerical implementation; no independent scientific review'))


if __name__=='__main__':main()
