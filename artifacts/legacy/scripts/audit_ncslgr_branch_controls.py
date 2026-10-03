"""Read-only source/cache/normalization/saved-head audit of the locked A03 run.

No fitting, model selection, changed population or encoder re-execution.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch

from _common import ROOT
from run_ncslgr_branch_ablation import CONFIG, FEATURES, OUTPUT, MANIFEST, NATIVE, PREP, SOURCE_MANIFEST, identities
from signrepr.cue_ablation import CONDITIONS, STREAMS, pool_conditions
from signrepr.io import read_jsonl, sha256, write_json
from signrepr.native_cache import verified_preprocessing
from signrepr.ncslgr import inclusive_frame_bounds
from signrepr.statistics import grouped_bootstrap
from signrepr.timestamps import source_frame_intervals

REPORT = ROOT / 'reports/ncslgr_branch_controls_audit_v3.json'
STATE = ROOT / 'reports/ncslgr_branch_controls_audit_v3.status.json'
PIPELINE = ROOT / 'state/ncslgr_branch_ablation_v3'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(a, b, message, atol=1e-6):
    require(np.asarray(a).shape == np.asarray(b).shape and np.allclose(a, b, atol=atol, rtol=1e-6), message)


def nested_near(a, b, message):
    if isinstance(a, dict):
        require(isinstance(b, dict) and a.keys()==b.keys(), message)
        for key in a:
            nested_near(a[key], b[key], message+': '+key)
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a)==len(b), message)
        for x, y in zip(a, b):
            nested_near(x, y, message)
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        near(a, b, message)
    else:
        require(a==b, message)


def read(path):
    return json.loads(path.read_text())


def ready():
    """Terminal failures never become waits or trigger a scientific retry."""
    pipeline=read(PIPELINE/'status.json')
    if pipeline['status']=='FAIL':
        raise ValueError('Scientific pipeline failed; audit cannot retry it')
    if pipeline['status']!='PASS_EXPLORATORY_CUE_ABLATION_PIPELINE':
        require(pipeline['status'] in ['RUNNING_CPU_INTERVENTION_EXTRACTION','RUNNING_CPU_FIXED_READOUTS'], 'Unknown scientific pipeline state')
        with (PIPELINE/'.pipeline.lock').open('r+') as worker:
            try:
                fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            else:
                fcntl.flock(worker,fcntl.LOCK_UN)
                raise ValueError('Running scientific pipeline has no held worker lock; inspect before recovery')
    summary=read(OUTPUT/'summary.json')
    require(summary['status']=='PASS_EXPLORATORY_SAME_SOURCE_CUE_ABLATION' and read(OUTPUT/'status.json')['status']==summary['status'], 'Terminal fixed readout PASS required')
    require(sha256(OUTPUT/'summary.json')==pipeline['summary_sha256'], 'Pipeline terminal summary changed')
    return True


def validate_selection(selected, policy):
    require(selected['test_used_for_selection'] is False, 'Test used for selection')
    candidates=selected['candidates']
    require([r['lr'] for r in candidates]==policy['lr_grid'], 'Declared validation grid differs')
    require(all(np.isfinite(r['validation']['macro_recall']) and 0<=r['validation']['macro_recall']<=1 for r in candidates), 'Invalid validation metric')
    best=max(candidates,key=lambda r:r['validation']['macro_recall'])
    require(best['lr']==selected['selected_lr'] and best['validation']==selected['validation'], 'First validation winner differs')


def class_metrics(gold, prediction):
    recalls={str(c):float(np.mean(prediction[gold==c]==c)) for c in np.unique(gold)}
    return dict(top1=float(np.mean(prediction==gold)),macro_recall=float(np.mean(list(recalls.values()))),per_class_recall=recalls)


def audit():
    policy=read(CONFIG);summary=read(OUTPUT/'summary.json');coverage=read(OUTPUT/'coverage.json')
    require(identities()==policy['identity_sha256']==summary['identity_sha256'], 'Pinned source/input identity differs')
    require(summary['config_sha256']==sha256(CONFIG) and summary['coverage_sha256']==sha256(OUTPUT/'coverage.json'), 'Protocol/coverage summary hash differs')
    feature_report=read(FEATURES/'extraction_report.json')
    require(feature_report['status']=='PASS_SAME_SOURCE_CUE_ABLATION_FEATURES' and read(FEATURES/'status.json')['status']==feature_report['status'], 'Full feature PASS required')
    require(feature_report['success']==feature_report['samples']==175 and feature_report['config_sha256']==sha256(CONFIG), 'Full registered feature population differs')
    require(coverage['feature_report_sha256']==sha256(FEATURES/'extraction_report.json') and coverage['feature_index_sha256']==sha256(FEATURES/'index.jsonl'), 'Readout source cache snapshot differs')
    index_rows=[r for _,r in read_jsonl(FEATURES/'index.jsonl')]
    index={r['utterance_sample_id']:r for r in index_rows}
    require(len(index)==len(index_rows)==175 and set(index)==set(policy['eligible_ids'])==set(coverage['eligible_ids']), 'Duplicate or changed common population')
    rows=[r for _,r in read_jsonl(MANIFEST) if r['view_role']=='body' and r['utterance_sample_id'] in index]
    rows.sort(key=lambda r:r['utterance_sample_id'])
    require(len(rows)==175 and [r['utterance_sample_id'] for r in rows]==coverage['eligible_ids'], 'Canonical label order/population differs')
    source_rows={r['sample_id']:r for _,r in read_jsonl(SOURCE_MANIFEST)}
    native={r['sample_id']:r for _,r in read_jsonl(NATIVE/'index.jsonl')}
    prep_identity=read(PREP/'cohort.lock.json');raw=[];source_audits=[];hashes={}
    for row,covered in zip(rows,coverage['rows']):
        entry=index[row['utterance_sample_id']];path=Path(entry['shard'])
        require(path.resolve().is_relative_to(FEATURES.resolve()) and read(path.with_suffix('.json'))==entry and sha256(path)==entry['shard_sha256'], 'Immutable intervention shard/index metadata differs')
        require(entry['status']=='SUCCESS' and entry['sample_id']==row['sample_id'] and entry['config_sha256']==sha256(CONFIG), 'Intervention sample/config differs')
        source=source_rows[row['sample_id']]
        audit_path=PREP/hashlib.sha256(row['sample_id'].encode()).hexdigest()[:20]/'audit.json'
        prepared=verified_preprocessing(audit_path,source,prep_identity)
        require(entry['source_sha256']==prepared['source_sha256'] and entry['preprocessing_audit_sha256']==sha256(audit_path), 'Canonical source/preprocessing linkage differs')
        reference=native[row['sample_id']];ref=Path(reference['shard'])
        require(ref.resolve().is_relative_to(NATIVE.resolve()) and read(ref.with_suffix('.json'))==reference and sha256(ref)==entry['reference_shard_sha256']==reference['shard_sha256'], 'Original reference provenance differs')
        with np.load(path,allow_pickle=False) as data,np.load(ref,allow_pickle=False) as original:
            values,observed=data['embeddings'],data['stream_observed_mask']
            require(values.shape==(len(CONDITIONS),prepared['frames'],768) and np.isfinite(values).all(), 'Invalid intervention feature dimensions/values')
            require(data['condition_names'].tolist()==list(CONDITIONS) and policy['conditions']=={k:list(v) for k,v in CONDITIONS.items()}, 'Condition axes/protocol differs')
            require(np.array_equal(data['condition_kept_streams'],np.asarray([[s in kept for s in STREAMS] for kept in CONDITIONS.values()],bool)), 'Intervention availability differs')
            require(observed.dtype==bool and observed.shape==(prepared['frames'],4), 'Original observation mask type/axes invalid')
            for key in ['stream_observed_mask','center_sec','window_start_sec','window_end_sec']:
                require(np.array_equal(data[key],original[key]), 'Source clock/observation changed: '+key)
            clock,_=source_frame_intervals(prepared['source_path'],prepared['frames'],prepared['fps'])
            near(data['window_start_sec'],clock[:,0],'Original PTS start differs',atol=1e-10)
            near(data['window_end_sec'],clock[:,1],'Original PTS end differs',atol=1e-10)
            near(data['center_sec'],clock.mean(1),'Original PTS center differs',atol=1e-10)
            first,last=inclusive_frame_bounds(row['utterance_start_ms'],row['utterance_end_ms'],30)
            pooled,keep=pool_conditions(values,observed,data['center_sec'],first/30,last/30)
            independently_selected=observed.all(1)&(data['center_sec']>=first/30)&(data['center_sec']<last/30)
            require(np.array_equal(keep,independently_selected) and int(keep.sum())==covered['pooled_frames']>0, 'Common supported frame selection differs')
            independent_mean=values[:,independently_selected].astype(np.float64).mean(1)
            independent_mean/=np.linalg.norm(independent_mean,axis=1,keepdims=True)
            near(pooled,independent_mean,'Float64 independent pooling/L2 definition differs')
            require(entry['observed_frames']==int(observed.all(1).sum()), 'Observed frame count differs')
            difference=float(np.abs(values[0]-original['embeddings']).max())
            require(difference==entry['all_reproduction_max_abs_difference']==covered['all_reproduction_max_abs_difference'], 'Recorded descriptive old-cache drift differs')
            require(np.isfinite(entry['within_run_all_repeat_max_abs_difference']) and entry['within_run_all_repeat_max_abs_difference']<=policy['within_run_repetition_tolerance'], 'Producer repetition measurement exceeds locked tolerance')
            require(entry['original_cache_equivalence']=='NOT_CERTIFIED', 'Old-cache numerical equivalence cannot be promoted')
            raw.append(pooled)
            source_audits.append(dict(sample_id=row['utterance_sample_id'],split=row['split'],pooled_frames=int(keep.sum()),source_frames=len(keep),
                                     old_cache_max_abs_difference=difference,producer_repeat_difference=entry['within_run_all_repeat_max_abs_difference']))
        hashes[str(path.relative_to(ROOT))]=entry['shard_sha256']
        hashes[str(audit_path.relative_to(ROOT))]=sha256(audit_path)
    raw=np.stack(raw);gold=np.asarray([policy['classes'].index(r['label']) for r in rows])
    split={s:np.asarray([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','val','test']}
    require({s:len(i) for s,i in split.items()}==summary['split_counts']==dict(train=65,val=27,test=83), 'Split counts differ')
    counts={s:np.bincount(gold[i],minlength=3).tolist() for s,i in split.items()}
    require(counts==coverage['class_counts'] and all(min(c)>0 for c in counts.values()), 'Class support differs')
    require(set(summary['results'])==set(CONDITIONS) and summary['trained_heads']==24, 'Missing/extra conditions or heads')
    train,val,test=[split[s] for s in ['train','val','test']];groups=[rows[i]['recording_group'] for i in test]
    correct={};verified_heads=[]
    for ci,condition in enumerate(CONDITIONS):
        value=raw[:,ci];mean,sd=value[train].mean(0),value[train].std(0)
        x=torch.from_numpy(((value-mean)/np.maximum(sd,1e-5)).astype(np.float32));tables=[]
        records=summary['results'][condition]['seeds']
        require([r['seed'] for r in records]==policy['seeds'],'Training seed coverage/order differs')
        for record in records:
            seed=record['seed'];directory=OUTPUT/f'{condition}_seed{seed}';selected=read(directory/'selection.json')
            require(selected==record['selection'] and selected['condition']==condition and selected['config_sha256']==sha256(CONFIG), 'Selection provenance differs')
            validate_selection(selected,policy)
            head=torch.load(directory/'head.pt',map_location='cpu',weights_only=False)
            require(head['selection']==selected,'Saved head selection provenance differs')
            require(np.array_equal(head['train_mean'],mean) and np.array_equal(head['train_sd'],sd),'Train-only normalization differs')
            model=torch.nn.Linear(768,3);model.load_state_dict(head['state_dict'],strict=True);model.eval().requires_grad_(False)
            require(sum(p.numel() for p in model.parameters())==policy['head_parameters']==2307,'Matched readout capacity differs')
            with torch.inference_mode():
                validation=model(x[val]).argmax(1).numpy();probability=model(x[test]).softmax(1).numpy();prediction=probability.argmax(1)
            nested_near(class_metrics(gold[val],validation),selected['validation'],'Selected-head validation reconstruction differs')
            measured=class_metrics(gold[test],prediction)
            nested_near(measured,record['metrics'],'Summary test metric differs');nested_near(measured,read(directory/'metrics.json'),'Head test metric differs')
            with (directory/'predictions.csv').open(newline='') as f:predictions=list(csv.DictReader(f))
            require([r['sample_id'] for r in predictions]==[rows[i]['utterance_sample_id'] for i in test], 'Prediction identity/order differs')
            for j,(table,i) in enumerate(zip(predictions,test)):
                require(table['split']=='test' and int(table['gold'])==gold[i] and int(table['prediction'])==prediction[j] and int(table['correct'])==int(prediction[j]==gold[i]),'Prediction/gold/correctness differs')
                require(table['recording_group']==rows[i]['recording_group'] and table['signer']==rows[i]['signer'] and table['conservative_xml_group']==rows[i]['conservative_xml_group'],'Prediction source grouping differs')
                near([float(table['probability_'+c]) for c in policy['classes']],probability[j],'Prediction probabilities differ')
            tables.append((prediction==gold[test]).astype(float))
            for filename in ['head.pt','selection.json','predictions.csv','metrics.json']:
                hashes[str((directory/filename).relative_to(ROOT))]=sha256(directory/filename)
            verified_heads.append(dict(condition=condition,seed=seed,selected_lr=selected['selected_lr'],parameters=2307,test_samples=len(test)))
        correct[condition]=np.stack(tables)
        nested_near(grouped_bootstrap(gold[test],groups,correct[condition]),summary['results'][condition]['bootstrap'],'Condition recording bootstrap differs')
    comparisons={condition:grouped_bootstrap(gold[test],groups,correct['all'],reference=correct[condition]) for condition in CONDITIONS if condition!='all'}
    nested_near(comparisons,summary['secondary_all_minus_condition'],'Paired recording comparisons differ')
    nested_near(comparisons['without_face'],summary['primary'],'Registered primary contrast differs')
    prior=int(np.bincount(gold[train],minlength=3).argmax())
    prior_correct=np.tile((gold[test]==prior).astype(float),(len(policy['seeds']),1))
    require(policy['classes'][prior]==summary['train_only_majority_prior']['label'],'Train-only prior differs')
    nested_near(grouped_bootstrap(gold[test],groups,prior_correct),summary['train_only_majority_prior']['bootstrap'],'Train-prior metrics differ')
    return dict(status='PASS_SOURCE_CLOCK_MASK_NORMALIZATION_AND_24_SAVED_HEADS_WITH_LIMITS',config_sha256=sha256(CONFIG),
                summary_sha256=sha256(OUTPUT/'summary.json'),coverage_sha256=sha256(OUTPUT/'coverage.json'),
                feature_report_sha256=sha256(FEATURES/'extraction_report.json'),feature_index_sha256=sha256(FEATURES/'index.jsonl'),
                source_audits=source_audits,verified_heads=verified_heads,verified_head_count=len(verified_heads),
                cached_shards=175,conditions=8,test_samples=83,old_cache_equivalence='NOT_CERTIFIED',files_sha256=hashes,
                script_sha256=sha256(Path(__file__)),limits=[
                    'Source/cache/timestamps/masks and selected-head predictions reconstructed; no fitting or population change.',
                    'Within-run encoder repetition values are producer measurements; audit does not independently rerun DINO/encoder or certify old-cache numerical equivalence.',
                    'Discarded candidate-head weights not retained: audit verifies declared grid, deterministic winner accounting and selected-head validation; it does not replay training.',
                    'Recomputed bootstrap checks estimator execution, not population coverage or independent signer/acquisition/pretraining exclusion.',
                    'Zeroing distribution shift, one test signer, original imputed context and historical positive-function gold remain; no causal branch importance, representation adaptation or method novelty.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wait-for-controls',action='store_true');args=parser.parse_args()
    require(not REPORT.exists() and not STATE.exists(),'Preserve previous audit attempt')
    with (PIPELINE/'.audit_worker.lock').open('a') as worker:
        fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.perf_counter();state=dict(status='WAITING_FOR_FIXED_READOUTS',pid=os.getpid(),device='cpu',script_sha256=sha256(Path(__file__)))
        try:
            while not ready():
                require(args.wait_for_controls,'Scientific controls incomplete; no partial audit metrics')
                state.update(elapsed_seconds=time.perf_counter()-started,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
                write_json(STATE,state);time.sleep(30)
            state['status']='AUDITING_SAVED_EVIDENCE';write_json(STATE,state);torch.set_num_threads(1)
            result=audit();result['elapsed_seconds_including_wait']=time.perf_counter()-started
            write_json(REPORT,result);state.update(status=result['status'],report_sha256=sha256(REPORT),elapsed_seconds=time.perf_counter()-started)
            write_json(STATE,state);print(json.dumps(dict(status=result['status'],verified_heads=result['verified_head_count'],shards=result['cached_shards'])))
        except Exception as error:
            state.update(status='FAIL',failure_reason=type(error).__name__+': '+str(error),elapsed_seconds=time.perf_counter()-started)
            write_json(STATE,state);raise


if __name__=='__main__':main()
