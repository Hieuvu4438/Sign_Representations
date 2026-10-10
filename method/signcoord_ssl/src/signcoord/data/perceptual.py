"""Versioned multi-keyframe pHash candidate screen; review remains explicit."""
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn

from signcoord.provenance import atomic_json,object_hash,sha256_file
from .video_index import record_metadata
from .video_io import decode_frames

POLICY={'schema_version':'perceptual_screen_v1','keyframes':5,'resize':32,'low_frequency':8,
        'dc_bit':0,'hamming_max':6,'min_matched_keyframes':3,
        'candidate_search':'seven_disjoint_bit_blocks_then_exact_hamming',
        'review':'candidate_only_no_automatic_duplicate_truth'}


def phash(frame):
    gray=np.asarray(Image.fromarray(frame).convert('L').resize((32,32),Image.Resampling.BILINEAR),dtype=np.float64)
    values=dctn(gray,norm='ortho')[:8,:8].reshape(-1)
    bits=values>np.median(values[1:]);bits[0]=False
    return sum(int(bit)<<i for i,bit in enumerate(bits))


def fingerprints(record,cache_root):
    pts,_,_=record_metadata(record)
    key=object_hash({'media':record['media_sha256'],'start':record['start_ms'],'end':record['end_ms'],'policy':POLICY})
    path=Path(cache_root)/(key+'.json')
    if path.exists():
        import json
        result=json.loads(path.read_text())
        if result['key']!=key:raise ValueError('perceptual cache provenance mismatch')
        return result
    use=np.flatnonzero((pts>=record['start_ms']/1000)&(pts<record['end_ms']/1000))
    if not len(use):raise ValueError('no source frames in perceptual interval')
    indices=use[np.unique(np.rint(np.linspace(0,len(use)-1,5)).astype(int))]
    frames=decode_frames(record['media_path'],indices)
    result={'key':key,'media_sha256':record['media_sha256'],'frame_indices':indices.tolist(),
            'pts_s':pts[indices].tolist(),'hashes':[phash(frame) for frame in frames],'policy':POLICY}
    atomic_json(path,result);return result


def candidate_matches(left,right,max_distance=6,min_matches=3):
    """Indexed pair candidates with distinct keyframes on both sides."""
    tables=[defaultdict(list) for _ in range(7)]
    boundaries=[0,10,19,28,37,46,55,64]
    def parts(value):
        return [(value>>a)&((1<<(b-a))-1) for a,b in zip(boundaries,boundaries[1:])]
    for j,item in enumerate(right):
        for frame,value in enumerate(item['hashes']):
            for table,part in zip(tables,parts(value)):table[part].append((j,frame,value))
    matches=defaultdict(list)
    for i,item in enumerate(left):
        for frame,value in enumerate(item['hashes']):
            possible=set()
            for table,part in zip(tables,parts(value)):possible.update(table.get(part,()))
            for j,other_frame,other in possible:
                distance=(value^other).bit_count()
                if distance<=max_distance:matches[(i,j)].append((frame,other_frame,distance))
    candidates=[]
    for (i,j),edges in sorted(matches.items()):
        # Longest increasing chain requires distinct, chronologically matched
        # keyframes. One matching still frame cannot count multiple times.
        paths={}
        for a,b,distance in sorted(edges):
            previous=[p for (i,j),p in paths.items() if i<a and j<b]
            best=max(previous,key=lambda p:(len(p),-sum(x[2] for x in p)),default=[])
            paths[(a,b)]=best+[(a,b,distance)]
        selected=max(paths.values(),key=lambda p:(len(p),-sum(x[2] for x in p)))
        if len(selected)>=min_matches:
            candidates.append({'left_index':i,'right_index':j,'matched_keyframes':selected,
                               'minimum_hamming':min(d for _,_,d in selected)})
    return candidates


def audit_perceptual(left,right,cache_root,decisions=None):
    features_left=[fingerprints(r,cache_root) for r in left]
    features_right=[fingerprints(r,cache_root) for r in right]
    candidates=candidate_matches(features_left,features_right)
    decisions=decisions or {};unresolved=[];duplicates=[];reviewed=[]
    for item in candidates:
        a,b=left[item.pop('left_index')],right[item.pop('right_index')]
        key=object_hash({'left_id':a['sample_id'],'right_id':b['sample_id'],
                         'left_media':a['media_sha256'],'right_media':b['media_sha256'],'policy':POLICY})
        row={'candidate_key':key,'ssl_id':a['sample_id'],'eval_id':b['sample_id'],
             'ssl_media':a['media_path'],'eval_media':b['media_path'],**item}
        decision=decisions.get(key)
        if decision is None:unresolved.append(row)
        else:
            if decision.get('decision') not in ('duplicate','different_content') or not decision.get('evidence') or not decision.get('reviewer'):
                raise ValueError('perceptual decision needs outcome, reviewer and evidence')
            row['review']=decision;reviewed.append(row)
            if decision['decision']=='duplicate':duplicates.append(row)
    return {'policy':POLICY,'policy_hash':object_hash(POLICY),'candidate_count':len(candidates),
            'unresolved_candidates':unresolved,'duplicates':duplicates,'reviewed_candidates':reviewed,
            'passed':not unresolved and not duplicates,'screen_scope':'these_frozen_manifest_intervals_only',
            'near_duplicate_verification':'candidate_screen_completed_with_explicit_review',
            'no_detection_is_not_proof_of_absence':True}
