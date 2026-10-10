"""First fixed H007 rejection scores on full registered source/target cohorts."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from cache import CacheReader, file_sha
from pooling import activity_mean
from probes import normalize_rows
from msasl_saved_features import load_complete_features
from resources import scratch_preflight
from frozen_rejection import fixed_scores, fit_source_known_threshold, SCORE_NAMES
from open_set_evaluation import JointOpenSet
from rejection_analysis import cluster_draws, cluster_mean_rate, conditional_interval, pack_count_curve


def read(path): return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser()
    for k in ('config','run-id','output-dir'):p.add_argument('--'+k,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with BoundedCPUContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        c=run.config;resource=scratch_preflight('/dev/shm',c['workspace_plan_bytes'])
        meta=Path(c['metadata_runtime']);all_rows=read(meta/'metadata_rows.json');tasks=read(meta/'candidate_partitions.json')
        lookup={role:{r['sample_id']:r for r in rows} for role,rows in all_rows.items()}
        prior=read(c['base_inputs']['identity_config']);report=read(c['base_inputs']['report'])
        cohort=read(prior['cohort_manifest']);audit=read(prior['video_audit'])
        assert report['passed'] and file_sha(report['cache_manifest'])==report['cache_manifest_sha256']
        reader=CacheReader(report['cache_manifest'],prior['cache_identity'],cohort,audit)
        vectors=[];windows=0
        for i,key in enumerate(c['base_ids']):
            x,activity,_=reader.load(key);r=lookup['ASLCitizen_base'][key]
            assert len(x)==r['windows'] and x.shape[1]==768
            assert reader.samples[key]['class_id']==r['class_id'] and reader.samples[key]['participant_id']==r['signer']
            vectors.append(normalize_rows(activity_mean(x,activity)[0][None])[0]);windows+=len(x)
            if (i+1)%100==0:print('base pooling',i+1,'/',len(c['base_ids']))
        base_values=np.stack(vectors);base_positions={key:i for i,key in enumerate(c['base_ids'])}
        run.save_array('base_vectors.npy',base_values,c['array_specs']['base_vectors.npy'])
        fc=read(Path(c['feature_runtime'])/'config.json');items=read(fc['extraction_manifest'])
        ms_values,_,feature_check=load_complete_features(c['feature_runtime'],items)
        ms_positions={r['sample_id']:i for i,r in enumerate(items)}
        parts=[];query_ids=[];labels=[];signers=[];offset=0
        for name,corpus,ids in [('base_calibration','ASLCitizen_base',tasks['ASLCitizen_base']['calibration_ids']),
                                ('base_dev','ASLCitizen_base',tasks['ASLCitizen_base']['dev_ids']),
                                ('MS_dev','MSASL100',tasks['MSASL100']['dev_ids'])]:
            assert ids==sorted(set(ids))
            qrows=[lookup[corpus][key] for key in ids]
            part=dict(name=name,corpus=corpus,start=offset,stop=offset+len(ids));parts.append(part)
            query_ids.extend(ids);labels.extend(r['class_id'] for r in qrows);signers.extend(r['signer'] for r in qrows);offset+=len(ids)
        assert offset==c['expected_total_queries']==1412 and list(SCORE_NAMES)==c['score_names']
        y=np.asarray(labels,dtype=np.int64);cos=np.zeros((3,offset,50),dtype=np.float64);gallery_meta=[]
        for si,seed in enumerate(c['support_seeds']):
            for part in parts:
                corpus=part['corpus'];entry=tasks[corpus]['seeds'][str(seed)];g=entry['galleries']['50']
                ids=query_ids[part['start']:part['stop']];support=[lookup[corpus][key] for key in g['support_ids']]
                assert [r['class_id'] for r in support]==g['class_ids'] and len(g['class_ids'])==50
                queries=[lookup[corpus][key] for key in ids]
                for field in ('sample_id','signer','source_group'):
                    assert not {r[field] for r in support}&{r[field] for r in queries}
                if corpus=='ASLCitizen_base':
                    q=base_values[[base_positions[key] for key in ids]];sv=base_values[[base_positions[key] for key in g['support_ids']]]
                else:
                    q=ms_values[[ms_positions[key] for key in ids]];sv=ms_values[[ms_positions[key] for key in g['support_ids']]]
                cos[si,part['start']:part['stop']]=np.clip(q@sv.T,-1,1)
                gallery_meta.append(dict(part=part['name'],seed=seed,**entry))
        scores=np.zeros((3,3,4,offset),dtype=np.float64);predictions=np.zeros((3,3,offset),dtype=np.int16);ranks=np.zeros_like(predictions)
        for si,seed in enumerate(c['support_seeds']):
            for ni,size in enumerate(c['gallery_sizes']):
                for part in parts:
                    sl=slice(part['start'],part['stop']);entry=tasks[part['corpus']]['seeds'][str(seed)]
                    full=entry['galleries']['50']['class_ids'];classes=entry['galleries'][str(size)]['class_ids'];columns=[full.index(cls) for cls in classes]
                    local=cos[si,sl][:,columns];s,pred,ranking=fixed_scores(local,classes,c['nac_k'],c['score_scale'])
                    scores[si,ni,:,sl]=s.T;predictions[si,ni,sl]=pred
                    relevant=ranking==y[sl,None];ranks[si,ni,sl]=np.where(relevant.any(axis=1),relevant.argmax(axis=1)+1,0)
        calibration={};cal=parts[0];cal_sl=slice(cal['start'],cal['stop'])
        for si,seed in enumerate(c['support_seeds']):
            for ni,size in enumerate(c['gallery_sizes']):
                classes=tasks['ASLCitizen_base']['seeds'][str(seed)]['galleries'][str(size)]['class_ids'];known=np.isin(y[cal_sl],classes)
                for mi,method in enumerate(SCORE_NAMES):
                    calibration[f'{seed}:{size}:{method}']=fit_source_known_threshold(scores[si,ni,mi,cal_sl],known,*c['target_fraction'])
        counts_array=np.zeros(c['array_specs']['count_curves.npy']['shape'],dtype=np.float64);cursor=0;curve_segments=[];metric_rows=[]
        for si,seed in enumerate(c['support_seeds']):
            for ni,size in enumerate(c['gallery_sizes']):
                for part in parts:
                    sl=slice(part['start'],part['stop']);entry=tasks[part['corpus']]['seeds'][str(seed)];classes=entry['galleries'][str(size)]['class_ids']
                    known=np.isin(y[sl],classes);common_known=np.isin(y[sl],entry['common_known_classes']);persistent=np.isin(y[sl],entry['common_unknown_classes'])
                    assert known.any() and (~known).any() and common_known.any() and persistent.any()
                    known_ranks=ranks[si,ni,sl][known];closed=dict(Top1=float(np.mean(known_ranks==1)),Top5=float(np.mean(known_ranks<=5)),
                        DCG=float(np.mean(1/np.log2(known_ranks+1))),MRR=float(np.mean(1/known_ranks)))
                    for mi,method in enumerate(SCORE_NAMES):
                        evaluator=JointOpenSet(query_ids[part['start']:part['stop']],y[sl],predictions[si,ni,sl],scores[si,ni,mi,sl],classes,query_ids[part['start']:part['stop']])
                        curve=evaluator.curve();packed=pack_count_curve(curve['points']);stop=cursor+len(packed);assert stop<=len(counts_array)
                        counts_array[cursor:stop]=packed
                        curve_segments.append(dict(part=part['name'],seed=seed,size=size,method=method,start=cursor,stop=stop,
                            n_known=int(known.sum()),n_unknown=int((~known).sum()),first_threshold_kind='+inf; stored0meansreject_all'))
                        cursor=stop
                        for policy in c['calibration_policies']:
                            fit=calibration[f'{seed}:{50 if policy=="FROZEN50" else size}:{method}'];point=evaluator.at_threshold(fit['threshold'])
                            accepted=scores[si,ni,mi,sl]>=fit['threshold'];correct=predictions[si,ni,sl]==y[sl]
                            metric_rows.append(dict(part=part['name'],seed=seed,size=size,method=method,policy=policy,point=point,known_retrieval=closed,
                                persistent_unknown_count=int(persistent.sum()),persistent_unknown_FAR=float(np.mean(accepted[persistent])),
                                common_known_count=int(common_known.sum()),common_known_acceptance=float(np.mean(accepted[common_known])),
                                common_known_CCR=float(np.mean((accepted&correct)[common_known])),
                                common_known_closed_Top1=float(np.mean(correct[common_known])),
                                adapted_grouped_trapezoid_AUC=curve['grouped_trapezoid_AUC']))
        rng=np.random.default_rng(a.seed);draws={};groups={}
        for part in parts[1:]:
            local=signers[part['start']:part['stop']];names,w=cluster_draws(local,c['bootstrap_replicates'],rng);groups[part['name']]=names;draws[part['name']]=w
        assert len(groups['base_dev'])==5 and len(groups['MS_dev'])==19
        def bootstrap(part,ni,mi,policy,persistent_only=False):
            sl=slice(part['start'],part['stop']);accepted=[];masks=[];size=c['gallery_sizes'][ni]
            for si,seed in enumerate(c['support_seeds']):
                entry=tasks[part['corpus']]['seeds'][str(seed)];classes=entry['common_unknown_classes'] if persistent_only else entry['galleries'][str(size)]['class_ids']
                mask=np.isin(y[sl],classes) if persistent_only else ~np.isin(y[sl],classes)
                tau=calibration[f'{seed}:{50 if policy=="FROZEN50" else size}:{SCORE_NAMES[mi]}']['threshold']
                accepted.append(scores[si,ni,mi,sl]>=tau);masks.append(mask)
            return cluster_mean_rate(np.stack(accepted),np.stack(masks),signers[part['start']:part['stop']],draws[part['name']])
        def find(part,seed,size,method,policy):
            return next(r for r in metric_rows if (r['part'],r['seed'],r['size'],r['method'],r['policy'])==(part,seed,size,method,policy))
        base_b,base_ok=bootstrap(parts[1],2,0,'FROZEN50');ms_b,ms_ok=bootstrap(parts[2],2,0,'FROZEN50');delta=ms_b-base_b;valid=base_ok&ms_ok
        interval=conditional_interval(delta,valid)
        per_seed=[find('MS_dev',seed,50,'maxcosine','FROZEN50')['point']['FAR']-find('base_dev',seed,50,'maxcosine','FROZEN50')['point']['FAR'] for seed in c['support_seeds']]
        point=float(np.mean(per_seed));ci=interval['interval95'];same_sign=bool(np.all(np.asarray(per_seed)>0) or np.all(np.asarray(per_seed)<0))
        passed=abs(point)>=c['primary_drift_floor'] and same_sign and ci is not None and (ci[0]>0 or ci[1]<0)
        primary=dict(mean_FAR_delta_MS_minus_base=point,per_seed_signed_delta=dict(zip(map(str,c['support_seeds']),per_seed)),
                     absolute_drift_floor=c['primary_drift_floor'],all_seed_same_nonzero_sign=same_sign,conditional_bootstrap=interval,screen_passed=bool(passed),
                     direction='higher target FAR' if point>0 else 'lower target FAR' if point<0 else 'no point change',method='maxcosine',policy='FROZEN50',gallery_size=50)
        paired=[]
        for part in parts[1:]:
            for mi,method in enumerate(SCORE_NAMES):
                for policy in c['calibration_policies']:
                    high,okh=bootstrap(part,2,mi,policy,True);low,okl=bootstrap(part,0,mi,policy,True)
                    differences=[find(part['name'],seed,50,method,policy)['persistent_unknown_FAR']-find(part['name'],seed,10,method,policy)['persistent_unknown_FAR'] for seed in c['support_seeds']]
                    paired.append(dict(part=part['name'],method=method,policy=policy,mean_persistent_unknown_delta50_minus10=float(np.mean(differences)),
                                       per_seed=differences,conditional_bootstrap=conditional_interval(high-low,okh&okl),secondary_exploratory=True))
        arrays={'cosine50.npy':cos,'scores.npy':scores,'predictions.npy':predictions,'known_rank.npy':ranks,'count_curves.npy':counts_array,
                'bootstrap_base.npy':draws['base_dev'].astype(np.int16),'bootstrap_MS.npy':draws['MS_dev'].astype(np.int16),
                'bootstrap_primary.npy':delta,'bootstrap_primary_valid.npy':valid}
        for name,array in arrays.items():run.save_array(name,array,c['array_specs'][name])
        run.save('rejection_metadata.json',dict(parts=parts,query_ids=query_ids,labels=labels,signers=signers,base_ids=c['base_ids'],
            support_seeds=c['support_seeds'],gallery_sizes=c['gallery_sizes'],score_names=list(SCORE_NAMES),gallery_metadata=gallery_meta,
            curve_segments=curve_segments,curve_points_used=cursor,curve_unused_tail='zero; exclude from all analyses',
            group_order=groups,base_windows_loaded=windows,base_numeric_clips=805,MS_complete_vectors=1037,
            source_label_budget='286 calibration labeled clips + one support/class; sourcepool classlabels available for support sampling, evaluationlabels not fitted',
            known_rank_zero='query class absent from dictionary; retained in rejection denominator'))
        run.save('rejection_eval.json',dict(primary=primary,calibration=calibration,metric_rows=metric_rows,paired_persistent_unknown=paired,
            resource=resource,feature_check=feature_check,GPU_seconds=0,new_encoder_calls=0,target_fit_calls=0,HPO_trials=0,
            test_access='none; reused exploratory train/validation',comparability='adapted protocol',
            scope='Fixed baseline operating-point diagnostic, not new method or fresh confirmation. Both corporaASL; input crops/physical identities/pretraining overlap unresolved. Bootstrap conditional fixed calibration/supports.'))
        print('all rows',offset,'curve points',cursor,'primary',json.dumps(primary))


if __name__=='__main__':main()
