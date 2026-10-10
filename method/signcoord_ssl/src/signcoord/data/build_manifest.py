"""Freeze selected local media/PTS; report every missing or undecodable record."""
import json
import os
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from signcoord.provenance import atomic_json,execution_stamp,object_hash,sha256_file
from .adapters import load_candidates,select_candidates
from .video_index import create_video_index
from .manifest import read_manifest


def build_manifest(adapter,root,split,output,annotation=None,seed=3407,
                   max_classes=None,max_per_class=None,limit=None,workers=2,exclude_groups_from=None):
    path=Path(output).resolve()
    if path.exists():raise ValueError('manifest exists; use a new immutable output')
    if workers<1:raise ValueError('manifest workers must be positive')
    candidates,metadata_paths=load_candidates(adapter,root,annotation,seed)
    rows,classes=select_candidates(candidates,split,seed,max_classes,max_per_class,limit)
    before_exclusion=len(rows);excluded=[]
    if exclude_groups_from:
        blocked={r['source_group_id'] for r in read_manifest(exclude_groups_from) if r['dataset_id']==adapter}
        excluded=[r['id'] for r in rows if r['source_group'] in blocked]
        rows=[r for r in rows if r['source_group'] not in blocked]
    if not rows:raise ValueError('no selected candidates')
    index_root=path.parent/'video_index';label_path=path.with_suffix('.labels.json')
    def freeze(row):
        sid=adapter+':'+row['id']
        try:
            if row.get('annotation_problem'):raise ValueError(row['annotation_problem'])
            digest=sha256_file(row['media'])
            index,pts,w,h=create_video_index(row['media'],index_root,digest)
            step=float(np.median(np.diff(pts))) if len(pts)>1 else 1/25
            record={'schema_version':'video_manifest_v1','sample_id':sid,'dataset_id':adapter,
                'source_video_id':row['source_id'],'source_group_id':row['source_group'],
                'media_path':str(row['media'].resolve()),'media_sha256':digest,
                'start_ms':max(0,float(pts[0]*1000)),'end_ms':float((pts[-1]+step)*1000),
                'signer_id':row['signer_id'],'sign_language':row['sign_language'],'split':split,
                'width':w,'height':h,'decoded_frames':len(pts),'nominal_fps':1/step,
                'pose_cache_key':None,'legacy_pose_path':str(row['legacy_pose'].resolve()),
                'license_ref':row['license_ref'],'label_ref':str(label_path) if row['label'] is not None else None,
                'video_index_path':str(index),'video_index_sha256':sha256_file(index),
                'decode_status':'full_PTS_decode_passed','access_status':'available',
                'annotation_source':row['annotation_source']}
            record['source_interval_verified']=adapter=='asl_citizen'
            for key in ('original_source_url','original_frame_start','original_frame_end','interval_policy'):
                if key in row:record[key]=row[key]
            return record,row['label'],None
        except (ValueError,OSError,KeyError,RuntimeError) as exc:
            return None,None,{'sample_id':sid,'media_path':str(row['media']),'error':str(exc)}
    with ThreadPoolExecutor(max_workers=workers) as pool:results=list(pool.map(freeze,rows))
    records=[r for r,_,_ in results if r is not None]
    failures=[e for _,_,e in results if e is not None]
    if not records:
        atomic_json(str(path)+'.failures.json',failures)
        raise ValueError('no decodable selected records; see failure ledger')
    labels={r['sample_id']:label for r,label,_ in results if r is not None and label is not None}
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.')
    try:
        with os.fdopen(fd,'w') as f:
            for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    if labels:atomic_json(label_path,labels)
    report={**execution_stamp(),'status':'passed_manifest_with_disclosed_failures' if failures else 'passed',
        'adapter':adapter,'root':str(Path(root).resolve()),'split':split,'manifest':str(path),
        'manifest_sha256':sha256_file(path),'labels_path':str(label_path) if labels else None,
        'metadata_hashes':{str(p.resolve()):sha256_file(p) for p in metadata_paths},
        'declared_split':sum(r['split']==split for r in candidates),'selected':len(rows),
        'available':len(records),'failures':failures,'class_counts':dict(Counter(labels.values())),
        'signer_count':len({r['signer_id'] for r in records if r['signer_id'] is not None}),
        'ordered_ids_hash':object_hash([r['sample_id'] for r in records]),
        'selection':{'seed':seed,'max_classes':max_classes,'max_per_class':max_per_class,'limit':limit,
                     'classes':classes,'policy':'metadata_only_shared_train_dev_classes_then_hashed_ID_order',
                     'selected_before_source_group_exclusion':before_exclusion,'excluded_source_group_sample_ids':excluded,
                     'exclude_groups_manifest_sha256':sha256_file(exclude_groups_from) if exclude_groups_from else None},
        'full_corpus_benchmark':max_classes is None and max_per_class is None and limit is None and not failures and not excluded,
        'signer_disjoint_verified':False,'near_duplicate_verification':'not_run'}
    atomic_json(str(path)+'.report.json',report)
    return report
