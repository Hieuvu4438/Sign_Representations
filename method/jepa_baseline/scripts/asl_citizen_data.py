"""Fixed official-split ASL-Citizen pilot and persistent original-teacher workers."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, rows):
    Path(path).write_text(''.join(json.dumps(row, sort_keys=True)+'\n' for row in rows))


def inventory(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = args.dataset.resolve()
    csvs = {part: source/'splits'/f'{part}.csv' for part in ['train','val','test']}
    annotations = {}
    annotation_pins = {part:digest(path) for part,path in csvs.items()}
    for part, path in csvs.items():
        with path.open() as file:
            annotations[part] = list(csv.DictReader(file))
    classes = sorted({row['Gloss'] for rows in annotations.values() for row in rows})
    labels = {name: index for index, name in enumerate(classes)}
    manifests, participants = {}, {}
    for part, rows in annotations.items():
        participants[part] = {row['Participant ID'] for row in rows}
        manifests[part] = []
        for row in rows:
            filename = row['Video file']
            manifests[part].append({
                'video_id': 'asl_citizen:'+Path(filename).stem,
                'source_id': Path(filename).stem, 'dataset': 'asl_citizen',
                'source_split': part, 'split': part,
                'video': str(source/'videos'/filename), 'label': labels[row['Gloss']],
                'gloss': row['Gloss'],
                'signer_id': hashlib.sha256(('asl_citizen:'+row['Participant ID']).encode()).hexdigest()[:16],
                'annotation_sha256': annotation_pins[part]})
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if participants[a] & participants[b]:
            raise ValueError(f'Official signer split overlap: {a}/{b}')
        if {row['video_id'] for row in manifests[a]} & {row['video_id'] for row in manifests[b]}:
            raise ValueError(f'Official source-video split overlap: {a}/{b}')
    write(output/'official_train.jsonl', manifests['train'])
    write(output/'official_held_out.jsonl', manifests['val']+manifests['test'])
    grouped = {}
    for part, rows in manifests.items():
        grouped[part] = defaultdict(list)
        for row in rows:
            grouped[part][row['label']].append(row)
    candidates = [label for label in range(len(classes))
                  if len(grouped['train'][label]) >= args.train_per_class
                  and len(grouped['val'][label]) >= args.val_per_class]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = sorted(candidates[:args.classes])
    if len(selected) != args.classes:
        raise ValueError('Insufficient eligible classes')
    membership, reserves = {}, []
    for part, limit in [('train',args.train_per_class),('val',args.val_per_class)]:
        membership[part] = []
        for label in selected:
            rows = grouped[part][label].copy()
            rng.shuffle(rows)
            # Prefer different training signers, using annotation only, before any score.
            if part == 'train':
                seen, diverse, rest = set(), [], []
                for row in rows:
                    if row['signer_id'] not in seen:
                        diverse.append(row);seen.add(row['signer_id'])
                    else:
                        rest.append(row)
                rows = diverse+rest
                reserves.extend(rows[limit:])
            membership[part].extend(rows[:limit])
    for rows in membership.values():
        for row in rows:
            if not Path(row['video']).is_file():
                raise ValueError(f"Selected video missing: {row['video_id']}")
    write(output/'source.jsonl', [{**row,'split':'pretrain'} for row in membership['train']])
    write(output/'gallery.jsonl', membership['train'])
    write(output/'queries.jsonl', membership['val'])
    write(output/'training_reserves.jsonl', reserves)
    (output/'labels.json').write_text(json.dumps({'official_classes':classes,'pilot_labels':selected},indent=2))
    report = {'scope':'fixed pilot, not full ASL-Citizen benchmark','seed':args.seed,
              'classes':len(selected),'train_videos':len(membership['train']),
              'val_queries':len(membership['val']),'test_queries':0,
              'official_split_counts':{part:len(rows) for part,rows in manifests.items()},
              'official_signer_counts':{part:len(values) for part,values in participants.items()},
              'official_signer_and_source_splits_disjoint':True,
              'csv_sha256':annotation_pins,
              'manifest_sha256':{name:digest(output/name) for name in ['source.jsonl','gallery.jsonl','queries.jsonl','official_train.jsonl','official_held_out.jsonl']},
              'selection':'seeded classes and videos; train signer diversity preferred; no model scores used',
              'created_unix':time.time()}
    (output/'inventory.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


TEACHER = None


def initialize_teacher(weights):
    global TEACHER
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['OMP_NUM_THREADS'] = '2'
    sys.path.insert(0,str(ROOT/'scripts'))
    from pose_worker import Teacher
    TEACHER = Teacher(weights)


def pose_one(task):
    from pose_worker import extract
    import numpy as np
    index, row, output = task
    raw = Path(output)/f'{index:08d}.npz'
    started = time.monotonic()
    if not raw.is_file():
        extract(TEACHER,row['video'],raw)
    with np.load(raw,allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata_json']))
        if metadata['video_sha256'] != digest(row['video']) or metadata.get('body_root_mode') != 'canonical':
            raise ValueError('Existing raw-cache source/convention mismatch')
        observed = data['body_detected']
        windows = int((np.convolve(observed.astype(int),np.ones(16,int),mode='valid')==16).sum()) if len(observed)>=16 else 0
    return {'video_id':row['video_id'],'index':index,'decoded_frames':len(observed),
            'valid_body_windows':windows,'seconds':time.monotonic()-started,'raw':str(raw),
            'status':'PASS' if windows else 'NO_VALID_WINDOW'}


def poses(args):
    output = args.output.resolve()/ 'corpus/raw'
    output.mkdir(parents=True,exist_ok=True)
    rows = [json.loads(line) for line in (args.output/'source.jsonl').read_text().splitlines() if line]
    tasks = [(index,row,str(output)) for index,row in enumerate(rows)]
    results = []
    started = time.monotonic()
    log = args.output/'pose_progress.jsonl'
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_teacher,initargs=(args.weights,)) as executor:
        futures = [executor.submit(pose_one,task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            with log.open('a') as file:
                file.write(json.dumps(result)+'\n')
            print(json.dumps({'completed':len(results),'total':len(tasks),'elapsed_seconds':time.monotonic()-started,
                              'latest_status':result['status']}),flush=True)
    results.sort(key=lambda row:row['index'])
    report = {'results':results,'seconds':time.monotonic()-started,
              'all_usable':all(row['status']=='PASS' for row in results)}
    (args.output/'pose_report.json').write_text(json.dumps(report,indent=2))
    if not report['all_usable']:
        raise ValueError('Some fixed pilot videos lack valid pose windows; inspect report before changing membership')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['inventory','poses'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,default=Path('/home/dongvk/datasets/ASL_Citizen'))
    parser.add_argument('--weights',type=Path,default=ROOT/'runtime/teacher/models')
    parser.add_argument('--classes',type=int,default=64)
    parser.add_argument('--train-per-class',type=int,default=2)
    parser.add_argument('--val-per-class',type=int,default=1)
    parser.add_argument('--seed',type=int,default=20261007)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    inventory(args) if args.stage=='inventory' else poses(args)


if __name__=='__main__':
    main()
