"""CPU frozen cue interventions on the exact previously audited common cohort."""
import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from _common import ROOT
from run_ncslgr_branch_ablation import CONFIG, FEATURES, NATIVE, PREP, SOURCE_MANIFEST, identities
from signrepr.cue_ablation import CONDITIONS, STREAMS, intervene
from signrepr.io import read_jsonl, sha256, write_json, write_jsonl
from signrepr.native_cache import verified_preprocessing


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--limit-new-clips',type=int)
    args=parser.parse_args();p=json.loads(CONFIG.read_text())
    if p['status']!='LOCKED_EXPLORATORY_SAME_SOURCE_CUE_ABLATION' or identities()!=p['identity_sha256']:
        raise ValueError('Locked experiment source identity differs')
    if args.limit_new_clips is not None and args.limit_new_clips != 2:
        raise ValueError('Only two deterministic training smoke cases allowed before full run')
    rows=[r for _,r in read_jsonl(SOURCE_MANIFEST) if r['utterance_sample_id'] in p['eligible_ids']]
    if len(rows)!=len(p['eligible_ids']) or len({r['sample_id'] for r in rows})!=len(rows) or any(r['official_split']!='train' for r in rows[:2]):
        raise ValueError('Locked source population or training smoke order differs')
    if FEATURES.exists():
        if not args.resume or json.loads((FEATURES/'status.json').read_text())['status']!='PARTIAL_TRAIN_SMOKE':
            raise ValueError('Preserve attempts; resume only a successfully reviewed bounded training smoke')
    else:
        if args.resume:
            raise ValueError('Resume requires reviewed existing smoke')
        FEATURES.mkdir(parents=True)
    import shutil
    if shutil.disk_usage(FEATURES).free < (p['minimum_free_disk_gib']+1)*2**30:
        raise ValueError('Reserved disk guard before loading models')
    lock_file=FEATURES/'.worker.lock'
    with lock_file.open('a') as worker:
        fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
        begun=time.perf_counter();config_hash=sha256(CONFIG)
        attempt=FEATURES/f'attempt_{len(list(FEATURES.glob("attempt_*.json")))+1:04d}.json'
        state=dict(status='RUNNING',pid=os.getpid(),device='cpu',workers=1,samples=len(rows),success=0,new_clips=0,
                   config_sha256=config_hash,torch_threads=p['extraction_torch_threads'],conditions=list(CONDITIONS))
        write_json(FEATURES/'status.json',state)
        entries=[]
        try:
            # Reuse verified native DINO decoding/transforms and frozen upstream encoder.
            import yaml
            from extract_shubert_pilot import dino_features, load_native_dino, NativeSHuBERT, torch, np
            cfg=yaml.safe_load((ROOT/'configs/shubert.yaml').read_text())
            encoder=NativeSHuBERT(ROOT,cfg,device='cpu')
            face,face_audit=load_native_dino(ROOT,cfg,'W03',device='cpu')
            hand,hand_audit=load_native_dino(ROOT,cfg,'W04',device='cpu')
            torch.set_num_threads(p['extraction_torch_threads'])
            index={r['sample_id']:r for _,r in read_jsonl(NATIVE/'index.jsonl')}
            prep_identity=json.loads((PREP/'cohort.lock.json').read_text())
            for row in rows:
                if args.limit_new_clips is not None and state['new_clips']>=args.limit_new_clips:
                    break
                audit_path=PREP/hashlib.sha256(row['sample_id'].encode()).hexdigest()[:20]/'audit.json'
                prep=verified_preprocessing(audit_path,row,prep_identity)
                old=index[row['sample_id']];old_path=Path(old['shard'])
                if old['status']!='SUCCESS' or sha256(old_path)!=old['shard_sha256'] or json.loads(old_path.with_suffix('.json').read_text())!=old:
                    raise ValueError('Original native reference shard integrity differs')
                key=hashlib.sha256(row['sample_id'].encode()).hexdigest()[:20];path=FEATURES/(key+'.npz');meta=path.with_suffix('.json')
                if meta.exists():
                    entry=json.loads(meta.read_text())
                    if entry['config_sha256']!=config_hash or entry['source_sha256']!=prep['source_sha256'] or entry['reference_shard_sha256']!=old['shard_sha256'] or sha256(path)!=entry['shard_sha256']:
                        raise ValueError('Reviewed intervention cache identity differs')
                else:
                    if shutil.disk_usage(FEATURES).free < (p['minimum_free_disk_gib']+1)*2**30:
                        raise ValueError('Reserved disk guard reached')
                    started=time.perf_counter()
                    streams={name:np.load(text,allow_pickle=False).astype(np.float32) if name=='body_posture'
                             else dino_features(face if name=='face' else hand,Path(text)) for name,text in prep['streams'].items()}
                    with np.load(old_path,allow_pickle=False) as reference:
                        observed=reference['stream_observed_mask'].copy();clock=reference['center_sec'].copy()
                        start=reference['window_start_sec'].copy();end=reference['window_end_sec'].copy()
                        original=reference['embeddings'].copy()
                    features=np.stack([encoder.forward(intervene(streams,condition)) for condition in CONDITIONS])
                    difference=float(np.abs(features[0]-original).max())
                    repeat_difference=float(np.abs(features[0]-encoder.forward(intervene(streams,'all'))).max())
                    if repeat_difference>p['within_run_repetition_tolerance']:
                        raise ValueError('Within-run all-stream repetition exceeds unchanged tolerance: '+str(repeat_difference))
                    if features.shape!=(len(CONDITIONS),prep['frames'],768) or not np.isfinite(features).all() or observed.dtype!=bool or observed.shape!=(prep['frames'],4):
                        raise ValueError('Intervention alignment/dimensions/masks invalid')
                    temporary=path.with_suffix('.npz.partial')
                    with temporary.open('wb') as f:
                        np.savez_compressed(f,embeddings=features,condition_names=np.asarray(list(CONDITIONS)),
                            condition_kept_streams=np.asarray([[s in kept for s in STREAMS] for kept in CONDITIONS.values()],dtype=bool),
                            stream_observed_mask=observed,center_sec=clock,window_start_sec=start,window_end_sec=end)
                        f.flush();os.fsync(f.fileno())
                    os.replace(temporary,path)
                    entry=dict(status='SUCCESS',sample_id=row['sample_id'],utterance_sample_id=row['utterance_sample_id'],
                        shard=str(path),shard_sha256=sha256(path),config_sha256=config_hash,source_sha256=prep['source_sha256'],
                        preprocessing_audit_sha256=sha256(audit_path),reference_shard_sha256=old['shard_sha256'],
                        frames=prep['frames'],observed_frames=int(observed.all(1).sum()),conditions=list(CONDITIONS),
                        all_reproduction_max_abs_difference=difference,within_run_all_repeat_max_abs_difference=repeat_difference,
                        original_cache_equivalence='NOT_CERTIFIED',feature_wall_seconds=time.perf_counter()-started)
                    write_json(meta,entry);state['new_clips']+=1
                entries.append(entry);state.update(success=len(entries),last_sample_id=row['sample_id'],elapsed_seconds=time.perf_counter()-begun)
                write_json(FEATURES/'status.json',state);write_jsonl(FEATURES/'index.jsonl',entries)
                print(json.dumps(dict(success=len(entries),samples=len(rows),sample_id=row['sample_id'],all_reproduction_max_abs_difference=entry['all_reproduction_max_abs_difference'])),flush=True)
            complete=len(entries)==len(rows)
            state.update(status='PASS_SAME_SOURCE_CUE_ABLATION_FEATURES' if complete else 'PARTIAL_TRAIN_SMOKE',elapsed_seconds=time.perf_counter()-begun)
            report=dict(**state,encoder_load_audit=encoder.audit,dino_load_audits=[face_audit,hand_audit],identity_sha256=identities(),
                        limits=p['limits'],scope='Frozen input zeroing, same source clocks and observation masks; no grammar metrics in extraction.')
            write_json(FEATURES/'extraction_report.json',report);write_json(attempt,state);write_json(FEATURES/'status.json',state)
        except Exception as error:
            state.update(status='FAIL',failure_reason=type(error).__name__+': '+str(error),elapsed_seconds=time.perf_counter()-begun)
            write_jsonl(FEATURES/'index.jsonl',entries);write_json(attempt,state);write_json(FEATURES/'status.json',state);raise


if __name__=='__main__':main()
