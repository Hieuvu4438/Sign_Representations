"""Independent original-space full score/rank/metric/grouped-interval replay."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from bounded_cpu_context import BoundedCPUContext
from bounded_artifacts import verify_store
from cache import file_sha


def main():
    p=argparse.ArgumentParser()
    for key in ('config','run-id','output-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with BoundedCPUContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        c=run.config;measured=Path(c['measured_runtime']);mc=json.loads((measured/'config.json').read_text())
        status=json.loads((measured/'status.json').read_text())
        assert status['status']=='completed' and status['GPU_hours']==0 and status['exit_code']==0
        store=verify_store(measured)
        assert file_sha(measured/'config.json')==status['config_sha256']
        for path,digest in mc['input_hashes'].items():assert file_sha(path)==digest,path
        for path,digest in status['output_hashes'].items():assert file_sha(measured/path)==digest,path
        meta=json.loads((measured/'shared_metadata.json').read_text());shared=meta['ranking']
        report=json.loads((measured/'geometry_eval.json').read_text())
        dictionary=json.loads((Path(c['dictionary_runtime'])/'dictionary_manifest.json').read_text())
        items=json.loads((Path(c['dictionary_runtime'])/'extraction_manifest.json').read_text())
        feature=Path(c['feature_runtime']);fc=json.loads((feature/'config.json').read_text())
        assert json.loads((feature/'status.json').read_text())['status']=='completed'
        feature_store=verify_store(feature)
        values=np.concatenate([np.load(feature/s['array_name'],allow_pickle=False) for s in fc['shards']])
        indexed={x['sample_id']:x for x in items};positions={x['sample_id']:i for i,x in enumerate(items)}
        queries=dictionary['query_ids'];labels=[indexed[k]['class_id'] for k in queries]
        people=[str(indexed[k]['signer_id']) for k in queries];groups=sorted(set(people))
        modes=['RR','TR','RT','TT','raw','centered'];seeds=[1701,1702,1703];classes=list(range(100))
        assert meta['target_loaded_ids']==[x['sample_id'] for x in items]==fc['sample_ids']
        assert meta['supports']==dictionary['supports'] and shared['query_ids']==queries and len(queries)==748
        assert shared['labels']==labels and shared['participants']==people and len(groups)==19
        assert shared['modes']==modes and shared['support_seeds']==seeds and shared['class_ids']==classes
        columns=['dcg','top1','top5','top10','top20','mrr'];assert shared['metric_columns']==columns
        assert shared['row_order']==['mode','support_seed','query_id'] and shared['prediction_rows']==13464
        load=lambda n:np.load(measured/n,allow_pickle=False)
        saved_scores,ranks,saved_rows=load('scores.npy'),load('ranks.npy'),load('row_metrics.npy')
        assert saved_scores.shape==ranks.shape==(13464,100) and saved_rows.shape==(13464,6)
        assert saved_scores.dtype==saved_rows.dtype==np.float64 and ranks.dtype==np.int16
        assert np.isfinite(saved_scores).all() and np.isfinite(saved_rows).all()
        assert np.array_equal(np.sort(ranks,axis=1),np.broadcast_to(np.arange(100),ranks.shape))
        base=Path(c['base_runtime']);base_store=verify_store(base)
        base_load=lambda n:np.load(base/n,allow_pickle=False)
        mean=base_load('mean.npy');centered=values-mean;index=0;max_score_error=max_row_error=max_aggregate_error=0.
        dcg={};per_gallery={};tie_totals={}
        for mode in modes:
            x=values if mode=='raw' else centered
            q=x[[positions[k] for k in queries]]
            if mode in modes[:4]:
                v=base_load('basis_'+mode[0]+'.npy');lam=base_load('spectrum_'+mode[1]+'.npy')
                inverse=(v/lam)@v.T
                qm=q@inverse;qn=np.sqrt(np.einsum('ij,ij->i',qm,q))
            else:
                inverse=None;qm=q;qn=np.sqrt(np.einsum('ij,ij->i',q,q))
            by_seed=[];per_gallery[mode]={};tie_totals[mode]={}
            for support_seed in seeds:
                ids=dictionary['supports'][str(support_seed)]
                assert [indexed[k]['class_id'] for k in ids]==classes
                g=x[[positions[k] for k in ids]]
                gm=g if inverse is None else g@inverse
                gn=np.sqrt(np.einsum('ij,ij->i',gm,g))
                scores=qm@g.T/(qn[:,None]*gn[None,:])
                assert np.isfinite(scores).all() and np.all(qn>0) and np.all(gn>0)
                error=float(np.max(np.abs(scores-saved_scores[index:index+748])));max_score_error=max(max_score_error,error)
                assert error<=c['numerical_score_atol'],(mode,support_seed,error)
                rows=[];class_rows={cl:[] for cl in classes}
                # Rank replay uses independent sorted tuples; saved-score diagnostics must match exactly.
                tie_pairs=0;min_gap=None
                for qi,key in enumerate(queries):
                    ordered=sorted(classes,key=lambda cl:(-float(scores[qi,cl]),cl))
                    assert ordered==ranks[index+qi].tolist(),(mode,support_seed,key)
                    rank=ordered.index(labels[qi])+1
                    row=[1/math.log2(rank+1),float(rank==1),float(rank<=5),float(rank<=10),float(rank<=20),1/rank]
                    error=float(np.max(np.abs(np.asarray(row)-saved_rows[index+qi])));max_row_error=max(max_row_error,error)
                    assert error<=c['row_metric_atol'];rows.append(row);class_rows[labels[qi]].append(row)
                    gaps=np.diff(np.sort(saved_scores[index+qi]));tie_pairs+=int(np.count_nonzero(gaps==0))
                    row_min=float(np.min(gaps));min_gap=row_min if min_gap is None else min(min_gap,row_min)
                diagnostics=report['numerical_diagnostics'][mode][str(support_seed)]
                assert tie_pairs==diagnostics['exact_adjacent_tie_pairs'] and min_gap==diagnostics['minimum_adjacent_score_gap']
                tie_totals[mode][str(support_seed)]=tie_pairs
                measured_metrics=report['metrics'][mode][str(support_seed)]
                assert measured_metrics['query_count']==748 and measured_metrics['class_count']==100
                for j,metric in enumerate(columns):
                    micro=math.fsum(r[j] for r in rows)/748
                    macro=math.fsum(math.fsum(r[j] for r in class_rows[cl])/len(class_rows[cl]) for cl in classes)/100
                    for kind,observed in (('micro',micro),('macro',macro)):
                        err=abs(observed-measured_metrics[kind][metric]);max_aggregate_error=max(max_aggregate_error,err)
                        assert err<=c['row_metric_atol']
                    for cl in classes:
                        assert len(class_rows[cl])==measured_metrics['class_support'][str(cl)]
                        observed=math.fsum(r[j] for r in class_rows[cl])/len(class_rows[cl])
                        assert abs(observed-measured_metrics['per_class'][str(cl)][metric])<=c['row_metric_atol']
                per_gallery[mode][str(support_seed)]=math.fsum(r[0] for r in rows)/748
                by_seed.append([r[0] for r in rows]);index+=748
            dcg[mode]=np.mean(by_seed,axis=0)
            for kind in ('micro','macro'):
                for metric in columns:
                    observed=math.fsum(report['metrics'][mode][str(s)][kind][metric] for s in seeds)/3
                    assert abs(observed-report['mean_over_fixed_galleries'][mode][kind][metric])<=c['row_metric_atol']
        assert index==13464
        differences=dcg['RR']-dcg['TR'];estimate=math.fsum(differences)/748
        primary=report['primary'];gallery={str(s):per_gallery['RR'][str(s)]-per_gallery['TR'][str(s)] for s in seeds}
        assert abs(estimate-primary['delta_DCG'])<=c['row_metric_atol']
        for s in seeds:assert abs(gallery[str(s)]-primary['per_gallery_delta_DCG'][str(s)])<=c['row_metric_atol']
        arrays={g:[float(differences[i]) for i,p in enumerate(people) if p==g] for g in groups}
        rng=np.random.default_rng(a.seed);boot=[]
        for _ in range(2000):
            chosen=[groups[i] for i in rng.integers(0,len(groups),size=len(groups))]
            boot.append(math.fsum(math.fsum(arrays[g]) for g in chosen)/sum(len(arrays[g]) for g in chosen))
        interval=np.quantile(boot,[.025,.975]);saved=primary['conditional_bootstrap']
        assert abs(saved['estimate_DCG']-estimate)<=c['row_metric_atol']
        assert np.max(np.abs(interval-saved['interval95_DCG']))<=c['interval_atol']
        assert saved['signer_count']==19 and saved['replicates']==2000 and saved['seed']==a.seed
        assert primary['practical_gain_floor']==c['primary_gain_floor']==.005 and primary['every_gallery_nonnegative_required']
        assert bool(estimate>=.005 and min(gallery.values())>=0)==primary['passed']
        assert bool(min(gallery.values())>=0)==primary['every_gallery_nonnegative_observed']
        run.save('geometry_eval_verification.json',dict(passed=True,original_space_fullrank_rows=index,
            maximum_score_error=max_score_error,maximum_row_metric_error=max_row_error,maximum_aggregate_error=max_aggregate_error,
            all_metrics_classes_deltas_interval_gate_verified=True,exact_tie_pairs=tie_totals,
            measured_store=store,feature_store=feature_store,base_store=base_store,GPU_hours=0,new_encoder_calls=0,
            scope='Same-agent independent quadraticmetric/rank/manualmetric/grouped-bootstrap savedartifact replay; no external scientificreview or basis portability guarantee'))


if __name__=='__main__':main()
