"""Local dataset adapters; labels remain in a separate downstream table."""
import csv
import json
import re
from pathlib import Path

from signcoord.provenance import object_hash

SPLITS={'train':'train','val':'dev','dev':'dev','test':'test'}


def how2sign_source(stem):
    return re.sub(r'_\d+-\d+-rgb_front$','',stem)


def asl_citizen(root):
    root=Path(root); candidates=[]; metadata=[]
    for original,split in (('train','train'),('val','dev'),('test','test')):
        path=root/'splits'/(original+'.csv');metadata.append(path)
        with path.open(newline='') as f:
            for row in csv.DictReader(f):
                stem=Path(row['Video file']).stem
                candidates.append({'id':stem,'split':split,'media':root/'videos'/(stem+'.mp4'),
                    'signer_id':row['Participant ID'] or None,'source_id':stem,'source_group':stem,
                    'label':row['Gloss'],'legacy_pose':root/'keypoints'/(stem+'.pkl'),
                    'license_ref':'https://www.microsoft.com/en-us/research/project/asl-citizen/',
                    'annotation_source':'official_split_csv','sign_language':'ase'})
    return candidates,metadata


def wlasl2000(root,annotation):
    root=Path(root); annotation=Path(annotation)
    official=json.loads(annotation.read_text())
    instances={i['video_id']:{'gloss':entry['gloss'],**i} for entry in official for i in entry['instances']}
    local_path=root/'preprocess/nslt_2000.json';class_path=root/'preprocess/wlasl_class_list.txt'
    local=json.loads(local_path.read_text())
    classes={int(line.replace('\\t','\t').split('\t',1)[0]):line.replace('\\t','\t').split('\t',1)[1]
             for line in class_path.read_text().splitlines() if line}
    candidates=[]
    for sid,item in local.items():
        info=instances.get(sid);label=classes[int(item['action'][0])];split=SPLITS[item['subset']]
        problem=None
        if info is None: problem='local_video_id_missing_from_pinned_official_annotation'
        elif info['gloss']!=label or SPLITS[info['split']]!=split:
            problem='local_class_or_split_differs_from_pinned_official_annotation'
        url=info['url'] if info else None
        # MP4 files are instances. Original URL/frame bounds are provenance;
        # never apply original frame_start again to these locally trimmed files.
        source=object_hash({'source_url':url}) if url else sid
        candidates.append({'id':sid,'split':split,'media':root/'WLASL2000'/(sid+'.mp4'),
            'signer_id':str(info['signer_id']) if info else None,'source_id':source,'source_group':source,
            'label':label,'legacy_pose':root/'keypoint'/(sid+'.pkl'),
            'license_ref':'https://github.com/dxli94/WLASL/blob/ac00e6be631c1a2a486621b65f202219f3964d6b/start_kit/C-UDA-1.0.pdf',
            'annotation_source':str(annotation.resolve()),'annotation_problem':problem,
            'original_source_url':url,'original_frame_start':info.get('frame_start') if info else None,
            'original_frame_end':info.get('frame_end') if info else None,
            'interval_policy':'entire_local_instance_MP4_without_reapplying_original_crop', 'sign_language':'ase'})
    return candidates,[local_path,class_path,annotation]


def how2sign(root,seed=3407,dev_fraction=.1):
    root=Path(root)
    held={how2sign_source(p.stem) for s in ('eval','test') for p in (root/s/'raw_videos').glob('*.mp4')}
    candidates=[]
    for original in ('train','eval','test'):
        for path in sorted((root/original/'raw_videos').glob('*.mp4')):
            source=how2sign_source(path.stem)
            if original=='train':
                split=('ssl_dev' if int(object_hash({'seed':seed,'source':source})[:16],16)/2**64<dev_fraction else 'ssl_train')
                problem='source_group_in_official_heldout' if source in held else None
            else:
                split='dev' if original=='eval' else 'test';problem=None
            candidates.append({'id':path.stem,'split':split,'media':path,'signer_id':None,
                'source_id':source,'source_group':source,'label':None,
                'legacy_pose':root/original/(original+'_pose')/(path.stem+'.pkl'),
                'license_ref':'https://how2sign.github.io/','annotation_source':'local_split_directory',
                'annotation_problem':problem,'sign_language':'ase'})
    return candidates,[]


def load_candidates(adapter,root,annotation=None,seed=3407):
    if adapter=='asl_citizen':return asl_citizen(root)
    if adapter=='wlasl2000':
        if not annotation:raise ValueError('WLASL adapter requires pinned official --annotation')
        return wlasl2000(root,annotation)
    if adapter=='how2sign':return how2sign(root,seed)
    raise ValueError(f'unknown dataset adapter: {adapter}')


def select_candidates(candidates,split,seed=3407,max_classes=None,max_per_class=None,limit=None):
    selected_classes=None
    if max_classes is not None:
        if max_classes<1:raise ValueError('max_classes must be positive')
        # Metadata-only class availability intersection, independent of scores.
        available={s:{r['label'] for r in candidates if r['split']==s and r['media'].is_file() and not r.get('annotation_problem')}
                   for s in ('train','dev')}
        common=available['train'] & available['dev'];common.discard(None)
        selected_classes=set(sorted(common,key=lambda label:object_hash({'seed':seed,'class':label}))[:max_classes])
    selected=[r for r in candidates if r['split']==split and (selected_classes is None or r['label'] in selected_classes)]
    selected.sort(key=lambda r:object_hash({'seed':seed,'id':r['id']}))
    if max_per_class is not None:
        if max_per_class<1:raise ValueError('max_per_class must be positive')
        counts={};bounded=[]
        for row in selected:
            label=row['label'];counts[label]=counts.get(label,0)+1
            if counts[label]<=max_per_class:bounded.append(row)
        selected=bounded
    if limit is not None:
        if limit<1:raise ValueError('limit must be positive')
        selected=selected[:limit]
    return selected,sorted(selected_classes) if selected_classes is not None else None
