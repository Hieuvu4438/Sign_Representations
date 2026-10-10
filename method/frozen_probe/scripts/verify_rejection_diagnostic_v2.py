"""Independent saved-input scalar scoring/count/statistic recount for H007."""
import argparse
import json
from math import exp, fsum, log2, sqrt
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from bounded_artifacts import verify_store
from cache import file_sha


def read(path):return json.loads(Path(path).read_text())


def close(actual,expected):
    assert abs(float(actual)-float(expected))<=1e-12,(actual,expected)


def main():
    p=argparse.ArgumentParser()
    for k in ('config','run-id','output-dir'):p.add_argument('--'+k,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with BoundedCPUContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        root=Path(run.config['measured_runtime']);c=read(root/'config.json');s=read(root/'status.json')
        assert s['status']=='completed' and s['exit_code']==0 and file_sha(root/'config.json')==s['config_sha256']
        store=verify_store(root)
        for n,h in s['output_hashes'].items():assert file_sha(root/n)==h
        meta=read(root/'rejection_metadata.json');report=read(root/'rejection_eval.json')
        original=read(Path(c['metadata_runtime'])/'metadata_rows.json');tasks=read(Path(c['metadata_runtime'])/'candidate_partitions.json')
        indexed={role:{r['sample_id']:r for r in rows} for role,rows in original.items()}
        arrays={n:np.load(root/n,allow_pickle=False) for n in c['array_specs']}
        for n,spec in c['array_specs'].items():
            x=arrays[n];assert x.shape==tuple(spec['shape']) and x.dtype.str==spec['dtype'] and np.isfinite(x).all()
        ids,y,signers=meta['query_ids'],meta['labels'],meta['signers'];parts=meta['parts']
        assert len(ids)==len(y)==len(signers)==1412
        expected_parts=[('base_calibration','ASLCitizen_base',tasks['ASLCitizen_base']['calibration_ids']),
                        ('base_dev','ASLCitizen_base',tasks['ASLCitizen_base']['dev_ids']),('MS_dev','MSASL100',tasks['MSASL100']['dev_ids'])]
        cursor=0
        for part,(name,corpus,expected) in zip(parts,expected_parts):
            assert (part['name'],part['corpus'],part['start'],part['stop'])==(name,corpus,cursor,cursor+len(expected))
            assert ids[cursor:part['stop']]==expected
            for i,key in enumerate(expected,cursor):
                source=indexed[corpus][key];assert (y[i],signers[i])==(source['class_id'],source['signer'])
            cursor=part['stop']
        assert cursor==1412 and meta['base_ids']==c['base_ids']
        base=arrays['base_vectors.npy'];basepos={key:i for i,key in enumerate(c['base_ids'])};poolerror=0.;windows=0
        for i,key in enumerate(c['base_ids']):
            row=indexed['ASLCitizen_base'][key]
            with np.load(row['feature']['path'],allow_pickle=False) as cache:
                assert set(cache.files)=={'features','raw_activity','centers_s'}
                x=cache['features'].astype(np.float64);activity=cache['raw_activity'].astype(np.float64);centers=cache['centers_s']
            assert x.shape==(row['windows'],768) and activity.shape==(len(x),2) and len(centers)==len(x)
            assert np.isfinite(x).all() and np.isfinite(activity).all()
            weights=np.maximum(np.minimum(np.maximum(activity[:,0],0),1),np.minimum(np.maximum(activity[:,1],0),1))
            total=fsum(float(w) for w in weights)
            value=x.mean(axis=0) if total<=1e-12 else np.einsum('ij,i->j',x,weights)/total
            norm=sqrt(fsum(float(v)*float(v) for v in value));assert norm>1e-12
            poolerror=max(poolerror,float(np.max(np.abs(value/norm-base[i]))));windows+=len(x)
        assert poolerror<=1e-12 and windows==meta['base_windows_loaded']
        fc=read(Path(c['feature_runtime'])/'config.json');items=read(fc['extraction_manifest']);ms={}
        for shard in fc['shards']:
            x=np.load(Path(c['feature_runtime'])/shard['array_name'],allow_pickle=False)
            for row,vector in zip(items[shard['start']:shard['stop']],x):
                assert row['sample_id'] not in ms;ms[row['sample_id']]=vector
        assert len(ms)==1037
        cosine=arrays['cosine50.npy'];coserror=scoreerror=0.;prediction_rows=0
        scores=arrays['scores.npy'];pred=arrays['predictions.npy'];ranks=arrays['known_rank.npy']
        for si,seed in enumerate(c['support_seeds']):
            for part in parts:
                corpus=part['corpus'];start,stop=part['start'],part['stop'];entry=tasks[corpus]['seeds'][str(seed)];full=entry['galleries']['50']
                q=np.stack([base[basepos[key]] if corpus=='ASLCitizen_base' else ms[key] for key in ids[start:stop]])
                g=np.stack([base[basepos[key]] if corpus=='ASLCitizen_base' else ms[key] for key in full['support_ids']])
                ref=np.clip(np.einsum('id,jd->ij',q,g),-1,1)
                coserror=max(coserror,float(np.max(np.abs(ref-cosine[si,start:stop]))))
                for ni,size in enumerate(c['gallery_sizes']):
                    classes=entry['galleries'][str(size)]['class_ids'];cols=[full['class_ids'].index(cls) for cls in classes]
                    for i in range(start,stop):
                        z=[float(cosine[si,i,j]) for j in cols];order=sorted(range(size),key=lambda j:(-z[j],classes[j]));top=z[order[0]]
                        ranking=[classes[j] for j in order]
                        assert int(pred[si,ni,i])==ranking[0]
                        assert int(ranks[si,ni,i])==(ranking.index(y[i])+1 if y[i] in classes else 0)
                        expected=[top,top-z[order[1]],1/fsum(exp(z[j]-top) for j in order),1/fsum(exp(z[j]-top) for j in order[:min(16,size)])]
                        scoreerror=max(scoreerror,max(abs(scores[si,ni,mi,i]-v) for mi,v in enumerate(expected)));prediction_rows+=1
        assert coserror<=1e-12 and scoreerror<=1e-12 and prediction_rows==12708
        cal=parts[0]
        for si,seed in enumerate(c['support_seeds']):
            for ni,size in enumerate(c['gallery_sizes']):
                classes=set(tasks['ASLCitizen_base']['seeds'][str(seed)]['galleries'][str(size)]['class_ids'])
                for mi,method in enumerate(c['score_names']):
                    selected=sorted([float(scores[si,ni,mi,i]) for i in range(cal['start'],cal['stop']) if y[i] in classes],reverse=True)
                    n=len(selected);k=(19*n+19)//20;tau=selected[k-1];saved=report['calibration'][f'{seed}:{size}:{method}']
                    assert tau==saved['threshold'] and n==saved['known_count'] and k==saved['minimum_accept_count']
                    assert sum(v>=tau for v in selected)==saved['accepted_known_count']
                    close(saved['empirical_known_acceptance'],saved['accepted_known_count']/n)
                    assert saved['tie_overshoot']==saved['accepted_known_count']-k
        partmap={p['name']:p for p in parts};counts=np.asarray(arrays['count_curves.npy']);curvepoints=0
        for segment in meta['curve_segments']:
            part=partmap[segment['part']];si=c['support_seeds'].index(segment['seed']);ni=c['gallery_sizes'].index(segment['size']);mi=c['score_names'].index(segment['method'])
            classes=set(tasks[part['corpus']]['seeds'][str(segment['seed'])]['galleries'][str(segment['size'])]['class_ids']);indices=list(range(part['start'],part['stop']))
            local=[float(scores[si,ni,mi,i]) for i in indices];thresholds=[float('inf')]+sorted(set(local),reverse=True)
            actual=counts[segment['start']:segment['stop']];assert len(thresholds)==len(actual)
            for j,tau in enumerate(thresholds):
                expected=[0 if j==0 else tau,
                          sum(y[i] in classes and scores[si,ni,mi,i]>=tau for i in indices),
                          sum(y[i] in classes and pred[si,ni,i]==y[i] and scores[si,ni,mi,i]>=tau for i in indices),
                          sum(y[i] not in classes and scores[si,ni,mi,i]>=tau for i in indices)]
                assert np.array_equal(actual[j],expected)
            nk=sum(y[i] in classes for i in indices);nu=len(indices)-nk
            assert (nk,nu)==(segment['n_known'],segment['n_unknown'])
            area=fsum((right[3]-left[3])/nu*(left[2]+right[2])/nk/2 for left,right in zip(actual,actual[1:]))
            for row in report['metric_rows']:
                if (row['part'],row['seed'],row['size'],row['method'])==(segment['part'],segment['seed'],segment['size'],segment['method']):close(row['adapted_grouped_trapezoid_AUC'],area)
            curvepoints+=len(actual)
        assert curvepoints==meta['curve_points_used'] and not counts[curvepoints:].any()
        dcg_errata=[]
        for row in report['metric_rows']:
            part=partmap[row['part']];si=c['support_seeds'].index(row['seed']);ni=c['gallery_sizes'].index(row['size']);mi=c['score_names'].index(row['method'])
            entry=tasks[part['corpus']]['seeds'][str(row['seed'])];classes=set(entry['galleries'][str(row['size'])]['class_ids']);indices=list(range(part['start'],part['stop']))
            known=[i for i in indices if y[i] in classes];unknown=[i for i in indices if y[i] not in classes]
            tau=report['calibration'][f'{row["seed"]}:{50 if row["policy"]=="FROZEN50" else row["size"]}:{row["method"]}']['threshold'];point=row['point']
            ka=sum(scores[si,ni,mi,i]>=tau for i in known);kca=sum(scores[si,ni,mi,i]>=tau and pred[si,ni,i]==y[i] for i in known);ua=sum(scores[si,ni,mi,i]>=tau for i in unknown)
            assert (point['n_known'],point['n_unknown'],point['known_accept_count'],point['known_correct_accept_count'],point['unknown_accept_count'])==(len(known),len(unknown),ka,kca,ua)
            close(point['FAR'],ua/len(unknown));close(point['CCR'],kca/len(known));close(point['known_acceptance'],ka/len(known))
            for key,value in [('Top1',sum(ranks[si,ni,i]==1 for i in known)/len(known)),('Top5',sum(ranks[si,ni,i]<=5 for i in known)/len(known)),
                              ('MRR',fsum(1/int(ranks[si,ni,i]) for i in known)/len(known))]:close(row['known_retrieval'][key],value)
            original_fp32=float(np.mean(1/np.log2(np.asarray([int(ranks[si,ni,i]) for i in known],dtype=np.int16)+1)))
            corrected=fsum(1/log2(int(ranks[si,ni,i])+1) for i in known)/len(known)
            assert row['known_retrieval']['DCG']==original_fp32
            dcg_errata.append(dict(part=row['part'],seed=row['seed'],size=row['size'],method=row['method'],policy=row['policy'],
                                   reported_FP32_DCG=row['known_retrieval']['DCG'],replayed_FP32_DCG=original_fp32,
                                   corrected_FP64_DCG=corrected,FP64_minus_FP32=corrected-original_fp32))
            assert point['threshold']==tau
            close(point['closed_known_accuracy'],sum(pred[si,ni,i]==y[i] for i in known)/len(known))
            common=[i for i in indices if y[i] in entry['common_known_classes']];persistent=[i for i in indices if y[i] in entry['common_unknown_classes']]
            assert len(common)==row['common_known_count'] and len(persistent)==row['persistent_unknown_count']
            close(row['persistent_unknown_FAR'],sum(scores[si,ni,mi,i]>=tau for i in persistent)/len(persistent))
            close(row['common_known_acceptance'],sum(scores[si,ni,mi,i]>=tau for i in common)/len(common))
            close(row['common_known_CCR'],sum(scores[si,ni,mi,i]>=tau and pred[si,ni,i]==y[i] for i in common)/len(common))
            close(row['common_known_closed_Top1'],sum(pred[si,ni,i]==y[i] for i in common)/len(common))
        rng=np.random.default_rng(a.seed);draws={}
        for part,file in [(parts[1],'bootstrap_base.npy'),(parts[2],'bootstrap_MS.npy')]:
            names=sorted(set(signers[part['start']:part['stop']]));w=rng.multinomial(len(names),np.full(len(names),1/len(names)),size=2000)
            assert names==meta['group_order'][part['name']] and np.array_equal(w,arrays[file]);draws[part['name']]=(names,w)
        def independently_bootstrap(part,ni,mi,policy,persistent=False):
            names,w=draws[part['name']];num=[];den=[];size=c['gallery_sizes'][ni]
            for si,seed in enumerate(c['support_seeds']):
                entry=tasks[part['corpus']]['seeds'][str(seed)];classes=set(entry['common_unknown_classes'] if persistent else entry['galleries'][str(size)]['class_ids'])
                tau=report['calibration'][f'{seed}:{50 if policy=="FROZEN50" else size}:{c["score_names"][mi]}']['threshold']
                counts_by_group=[]
                for group in names:
                    eligible=[i for i in range(part['start'],part['stop']) if signers[i]==group and ((y[i] in classes) if persistent else (y[i] not in classes))]
                    counts_by_group.append((sum(scores[si,ni,mi,i]>=tau for i in eligible),len(eligible)))
                num.append([v[0] for v in counts_by_group]);den.append([v[1] for v in counts_by_group])
            n=w@np.asarray(num).T;d=w@np.asarray(den).T;ok=np.all(d>0,axis=1)
            out=np.divide(n,d,out=np.zeros(n.shape,dtype=float),where=d>0).mean(axis=1)
            return out,ok
        base,okb=independently_bootstrap(parts[1],2,0,'FROZEN50');ms,okm=independently_bootstrap(parts[2],2,0,'FROZEN50');delta=ms-base;valid=okb&okm
        assert np.array_equal(valid,arrays['bootstrap_primary_valid.npy']) and np.max(np.abs(delta-arrays['bootstrap_primary.npy']))<=1e-12
        interval=np.quantile(delta,[.025,.975]).tolist() if valid.all() else None
        assert int((~valid).sum())==report['primary']['conditional_bootstrap']['invalid_draws']
        if interval is not None:
            for actual,expected in zip(report['primary']['conditional_bootstrap']['interval95'],interval):close(actual,expected)
        primary_deltas=[]
        for seed in c['support_seeds']:
            vals={r['part']:r['point']['FAR'] for r in report['metric_rows'] if (r['seed'],r['size'],r['method'],r['policy'])==(seed,50,'maxcosine','FROZEN50')}
            primary_deltas.append(vals['MS_dev']-vals['base_dev']);close(report['primary']['per_seed_signed_delta'][str(seed)],primary_deltas[-1])
        point=fsum(primary_deltas)/3;close(point,report['primary']['mean_FAR_delta_MS_minus_base'])
        same=all(v>0 for v in primary_deltas) or all(v<0 for v in primary_deltas)
        passed=abs(point)>=.05 and same and interval is not None and (interval[0]>0 or interval[1]<0)
        assert passed==report['primary']['screen_passed']
        for row in report['paired_persistent_unknown']:
            part=partmap[row['part']];mi=c['score_names'].index(row['method']);hi,okh=independently_bootstrap(part,2,mi,row['policy'],True);lo,okl=independently_bootstrap(part,0,mi,row['policy'],True)
            ok=okh&okl;assert int((~ok).sum())==row['conditional_bootstrap']['invalid_draws']
            if ok.all():
                for actual,expected in zip(row['conditional_bootstrap']['interval95'],np.quantile(hi-lo,[.025,.975])):close(actual,expected)
            for j,seed in enumerate(c['support_seeds']):
                vals={r['size']:r['persistent_unknown_FAR'] for r in report['metric_rows'] if (r['part'],r['seed'],r['method'],r['policy'])==(row['part'],seed,row['method'],row['policy'])}
                close(row['per_seed'][j],vals[50]-vals[10])
            close(row['mean_persistent_unknown_delta50_minus10'],fsum(row['per_seed'])/3)
        run.save('dcg_precision_erratum.json',dict(version='P007_v1.secondary_DCG_FP64_r1',parent_run=s['run_id'],
            affected_field='metric_rows[].known_retrieval.DCG',original_outputs_unchanged=True,primary_unchanged=True,
            rows=dcg_errata,maximum_absolute_correction=max(abs(x['FP64_minus_FP32']) for x in dcg_errata),
            scope='Explicit reporting precision correction on saved ranks, not new prediction/primary/threshold measurement'))
        run.save('rejection_verification.json',dict(passed=True,base_repooled_clips=805,maximum_independent_pool_error=poolerror,
            maximum_einsum_cosine_error=coserror,maximum_scalar_score_error=scoreerror,
            exact_prediction_and_knownrank_rows=prediction_rows,calibration_cells_verified=len(report['calibration']),
            metric_rows_verified=len(report['metric_rows']),curve_segments_verified=len(meta['curve_segments']),curve_points_verified=curvepoints,
            primary_bootstrap_replicates=2000,paired_persistent_unknown_rows=len(report['paired_persistent_unknown']),
            primary=report['primary'],measured_store=store,original_DCG_precision='FP32 replayed, FP64 separately corrected',
            scope='Same-agent independent input-pooling/einsum numeric parity and saved-logit scalar scoring/ordered-ID/threshold/count/bootstrap recount. Exact decisions verified on saved numerical values; no cross-environment bitwise guarantee, external review, raw-video redecoding or new hypothesis.'))


if __name__=='__main__':main()
