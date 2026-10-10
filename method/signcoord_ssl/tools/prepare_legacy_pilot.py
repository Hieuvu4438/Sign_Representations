"""Versioned legacy How2Sign conversion for a bounded real target audit.

Quality = clip(raw backend score / train-only per-part 90th percentile, 0,1).
It is a heuristic quality score, not a probability. Original estimator unknown.
Multiple/zero-person frames are invalid; no motion-based identity selection.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"src"))
from signcoord.data.video_io import video_metadata
from signcoord.provenance import atomic_json, execution_stamp, object_hash, sha256_file
from signcoord.targets.schema import SCHEMA_HASH

POLICY="legacy_image_normalized_xy_train_part_q90_quality_v1"
PARTS={"body":np.arange(23),"face":np.arange(23,91),"L":np.arange(91,112),"R":np.arange(112,133)}


def source_id(stem):
    return re.sub(r"_\d+-\d+-rgb_front$","",stem)


def read_legacy(path):
    with path.open('rb') as f: d=pickle.load(f)
    if set(d) != {'keypoints','scores'} or len(d['keypoints']) != len(d['scores']):
        raise ValueError('legacy structure mismatch')
    n=len(d['keypoints']); xy=np.zeros((n,133,2),dtype=np.float32); score=np.zeros((n,133),dtype=np.float32)
    observed=np.zeros((n,133),dtype=bool); status=np.full((n,133),2,dtype=np.uint8)
    for t,(kp,sc) in enumerate(zip(d['keypoints'],d['scores'])):
        kp,sc=np.asarray(kp),np.asarray(sc)
        if kp.ndim != 3 or kp.shape[1:] != (133,2) or sc.shape != kp.shape[:2]:
            raise ValueError(f'legacy frame {t} schema mismatch')
        if len(kp)==1:
            xy[t],score[t]=kp[0],sc[0]
            observed[t]=np.isfinite(xy[t]).all(-1) & np.isfinite(score[t])
            status[t,observed[t]]=0
    return xy,score,observed,status


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True)
    parser.add_argument('--trace-report',default=str(PROJECT/'runs/legacy_pose_trace_v1/trace_report.json'))
    parser.add_argument('--seed',type=int,default=3407); parser.add_argument('--train-clips',type=int,default=100)
    parser.add_argument('--dev-clips',type=int,default=20); args=parser.parse_args()
    root=Path(args.output).resolve(); root.mkdir(parents=True,exist_ok=False)
    (root/'pose').mkdir(); (root/'manifests').mkdir()
    h2s=Path('/home/shared_data/sign_language/How2Sign')
    names=sorted(p.stem for p in (h2s/'train/train_pose').glob('*.pkl'))
    groups={}
    for stem in names: groups.setdefault(source_id(stem),[]).append(stem)
    # Exclude any known source recordings in official How2Sign eval/test.
    held_sources={source_id(p.stem) for split in ('eval','test') for p in (h2s/split/'raw_videos').glob('*.mp4')}
    sources=sorted(set(groups)-held_sources); rng=np.random.default_rng(args.seed); rng.shuffle(sources)
    selected={'ssl_train':[],'ssl_dev':[]}; taken=set()
    # One clip/source produces diverse source groups; deterministic group holdout.
    for split,want in (('ssl_train',args.train_clips),('ssl_dev',args.dev_clips)):
        for source in sources:
            if source in taken: continue
            for stem in groups[source]:
                vp=h2s/'train/raw_videos'/(stem+'.mp4')
                if vp.exists():
                    selected[split].append(stem); taken.add(source); break
            if len(selected[split])==want: break
        if len(selected[split]) != want: raise ValueError('insufficient group-disjoint local clips')
    # Fit quality on train observations only; retain raw score distribution/counts.
    values={part:[] for part in PARTS}
    pose_inputs={}
    for stem in selected['ssl_train']:
        p=h2s/'train/train_pose'/(stem+'.pkl'); xy,sc,obs,status=read_legacy(p)
        pose_inputs[stem]=sha256_file(p)
        for part,idx in PARTS.items():
            v=sc[:,idx]; use=obs[:,idx] & np.isfinite(v) & (v>0)
            values[part].append(v[use])
    calibration={part:float(np.quantile(np.concatenate(v),.90)) for part,v in values.items()}
    if any(not np.isfinite(v) or v<=0 for v in calibration.values()): raise ValueError('invalid score scale')
    quality={**execution_stamp(),'policy':POLICY,'fit_split':'ssl_train','sample_ids':selected['ssl_train'],
             'pose_input_hashes':pose_inputs,'part_q90':calibration,
             'score_semantics':'heuristic_bounded_backend_quality_not_probability',
             'coordinate_evidence_sha256':sha256_file(args.trace_report)}
    atomic_json(root/'quality_calibration.json',quality); quality_hash=sha256_file(root/'quality_calibration.json')
    reports=[]
    for split,names in selected.items():
        rows=[]
        for stem in names:
            pp=h2s/'train/train_pose'/(stem+'.pkl'); vp=h2s/'train/raw_videos'/(stem+'.mp4')
            try:
                xy,sc,obs,status=read_legacy(pp); pts,w,h=video_metadata(vp)
                if len(xy)!=len(pts): raise ValueError('legacy/video frame count mismatch')
                xy*=np.array([w,h],dtype=np.float32)
                quality_score=np.zeros_like(sc)
                for part,idx in PARTS.items(): quality_score[:,idx]=np.clip(sc[:,idx]/calibration[part],0,1)
                quality_score=np.where(obs,quality_score,0); xy=np.where(obs[...,None],xy,0)
                digest=sha256_file(vp); posehash=sha256_file(pp)
                metadata={'schema_version':'pose_cache_v1','schema_hash':SCHEMA_HASH,'media_sha256':digest,
                          'extractor_id':'legacy_unknown_backend','extractor_weights_hash':None,
                          'source_pose_sha256':posehash,'coordinate_units':'pixels_from_xW_yH_overlay_verified',
                          'score_semantics':quality['score_semantics'],'conversion_policy':POLICY,
                          'quality_calibration_sha256':quality_hash,'tracking_policy':'single_person_only_no_persistent_id',
                          'tracking_identity_verified':False,'pose_frame_order':'assumed_decode_order_count_and_midframe_overlay_verified'}
                key=object_hash({'media':digest,'pose':posehash,'schema':SCHEMA_HASH,'quality':quality_hash,'policy':POLICY})+'.npz'
                np.savez_compressed(root/'pose'/key,xy_px=xy,confidence=quality_score,observed=obs,status=status,pts_s=pts,
                                    metadata=json.dumps(metadata),raw_backend_score=sc)
                frame_step=np.median(np.diff(pts)) if len(pts)>1 else 1/25
                end=float(pts[-1]+frame_step)*1000
                rows.append({'schema_version':'video_manifest_v1','sample_id':'how2sign:'+stem,'dataset_id':'how2sign',
                             'source_video_id':source_id(stem),'source_group_id':source_id(stem),'media_path':str(vp),
                             'media_sha256':digest,'start_ms':max(0,float(pts[0]*1000)),'end_ms':end,'signer_id':None,
                             'sign_language':'ase','split':split,'pose_cache_key':key,'duplicate_group_id':None,
                             'label_ref':None,'access_status':'available','license_ref':'https://how2sign.github.io/',
                             'decode_status':'verified','pose_status':'legacy_converted_with_unknown_extractor'})
                reports.append({'sample_id':stem,'split':split,'status':'converted','source_frames':len(pts),
                                'single_person_fraction':float(obs.any(-1).mean()),'pose_cache_key':key})
            except Exception as exc: reports.append({'sample_id':stem,'split':split,'status':'failed','error':str(exc)})
        (root/'manifests'/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    base=yaml.safe_load((PROJECT/'configs/base.yaml').read_text())
    base['model'].update(backbone='signrep_hiera_adapter',initialization_mode='signrep_pretrained',
                         initialization_checkpoint=str(PROJECT.parents[1]/'checkpoints/signrep/ckpt.pt'))
    base['data'].update(ssl_train_manifest=str(root/'manifests/ssl_train.jsonl'),ssl_dev_manifest=str(root/'manifests/ssl_dev.jsonl'),
                        pose_cache_root=str(root/'pose'),target_stats_path=str(root/'stats/target_stats.json'),
                        phase_calibration_path=str(root/'stats/phase_calibration.json'))
    base['data']['augmentation']['color_jitter']=0.0  # explicit smoke limitation, no hidden unsupported jitter
    base['train'].update(device='cpu',precision='fp32',microbatch_per_gpu=2,accumulation_steps=1,dataloader_workers=0,
                          max_updates=4,warmup_updates=1,save_every_updates=2,eval_every_updates=2)
    base['loss']['relation'].update(warmup_zero_updates=1,ramp_end_update=3)
    base['run_dir']=str(root/'runs/A7_real_smoke')
    (root/'real_smoke.yaml').write_text(yaml.safe_dump(base,sort_keys=False))
    report={**execution_stamp(),'status':'prepared_for_audit','policy':POLICY,'quality_calibration_hash':quality_hash,
            'signer_disjoint_verified':False,'upstream_signrep_pretraining_overlap':'unknown',
            'source_groups_excluded_for_official_h2s_heldout':len(held_sources),'records':reports}
    atomic_json(root/'conversion_report.json',report)
    print(json.dumps({'output':str(root),'converted':sum(r['status']=='converted' for r in reports),
                      'failed':sum(r['status']=='failed' for r in reports),'quality_q90':calibration}))


if __name__=='__main__': main()
