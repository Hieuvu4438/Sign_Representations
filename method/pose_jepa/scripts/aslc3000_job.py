"""Durable ASLC3000 job: resumable conversion, training and queued frozen evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
if (PACKAGE/'src').is_dir(): sys.path.insert(0, str(PACKAGE/'src'))
from signrepr.training.job_support import atomic_json
from signrepr.data.schema import sha256_file, read_manifest, write_manifest, load_pose


def file_identity(path):
    stat = Path(path).stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def prepare_one(task):
    import numpy as np
    from signrepr.data.import_unisign import import_unisign_pose
    import av
    root, metadata = Path(task['root']), task['metadata']
    source, video = Path(task['source']), Path(metadata['video_path'])
    identity = {'pose': file_identity(source), 'video': file_identity(video), 'metadata': metadata}
    token = hashlib.sha256(metadata['sample_id'].encode()).hexdigest()
    receipt, output = root/'receipts'/f'{token}.json', root/'poses'/f'{token}.npz'
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous['input_identity'] != identity:
            raise ValueError(f'Source changed after conversion: {source}')
        row = previous['row']
        if output.is_file() and sha256_file(output) == row['pose_sha256']:
            load_pose(output)
            if sha256_file(row['raw_detector_scores_path']) == row['raw_detector_scores_sha256']:
                return row
    timestamps = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        width, height = stream.width, stream.height
        fps = float(stream.average_rate) if stream.average_rate else None
        stream.codec_context.thread_count = 1
        for frame in container.decode(stream):
            if frame.pts is None: raise ValueError(f'Missing native frame PTS: {video}')
            timestamps.append(float(frame.pts*frame.time_base))
    if not timestamps or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f'Invalid source timing: {video}')
    row, tracking = import_unisign_pose(source, {**metadata, 'timestamps_sec': timestamps, 'fps': fps,
                                                'image_width': width, 'image_height': height,
                                                'video_sha256': sha256_file(video)}, output)
    if row['num_frames'] != len(timestamps): raise ValueError(f'Pose/video frame mismatch: {source}')
    atomic_json(receipt, {'row': row, 'tracking': tracking, 'input_identity': identity})
    return row


def prepare(job, job_root):
    import multiprocessing as mp
    root = Path(job['data_root']); root.mkdir(parents=True, exist_ok=True)
    records = json.loads(Path(job['input_index']).read_text())
    stop = {'requested': False}
    def request_stop(sig, frame): stop['requested'] = True
    handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGTERM, signal.SIGINT)}
    rows = {}; iterator = iter(records)
    workers = job['prepare_workers']; started = time.time()
    try:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as pool:
            pending = {}
            def schedule():
                while not stop['requested'] and not (job_root/'pause.request').exists() and len(pending) < workers*2:
                    try: record = next(iterator)
                    except StopIteration: break
                    future = pool.submit(prepare_one, {'root': str(root), **record})
                    pending[future] = record['metadata']['sample_id']
            schedule()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED, timeout=2)
                for future in done:
                    sample = pending.pop(future); rows[sample] = future.result()
                atomic_json(job_root/'progress.json', {'phase': 'prepare', 'status': 'running', 'completed': len(rows),
                                                       'total': len(records), 'pid': os.getpid(), 'updated_unix': time.time(),
                                                       'elapsed_seconds': time.time()-started})
                (job_root/'progress.txt').write_text(f"Pose-JEPA ASLC3000 | prepare\nSamples: {len(rows)}/{len(records)}\nUpdated: {time.ctime()}\n")
                schedule()
        if stop['requested'] or (job_root/'pause.request').exists():
            return {'status': 'paused', 'completed': len(rows)}
        require(len(rows) == len(records), 'Incomplete pose conversion')
        manifests = {}
        for name, ids in job['splits'].items():
            path = root/f'{name}.jsonl'; write_manifest(path, [rows[i] for i in ids]); manifests[name] = str(path)
        train = [rows[i] for i in job['splits']['train']]
        validation = [rows[i] for i in job['splits']['selection_val']]
        queries = [rows[i] for i in job['splits']['queries']]
        require(not {r['source_video_id'] for r in train}&{r['source_video_id'] for r in validation+queries}, 'Train/evaluation source overlap')
        require(not {r['sample_id'] for r in validation}&{r['sample_id'] for r in queries}, 'Selection/query overlap')
        # Hash-identical poses/videos across train and held-out data are rejected.
        for key in ('pose_sha256', 'video_sha256'):
            require(not {r[key] for r in train}&{r[key] for r in validation+queries}, f'Train/heldout {key} overlap')
        coverage = {'samples': len(rows), 'frames': sum(r['num_frames'] for r in rows.values()),
                    'timing': 'source PTS, matching cache native frame count', 'manifests': manifests,
                    'split_counts': {k: len(v) for k,v in job['splits'].items()},
                    'score_profile': 'simcc_positive_rational_v1', 'raw_scores_preserved': True,
                    'source_overlap': False, 'selection_query_overlap': False,
                    'unverified_extractor_weights': True}
        from signrepr.data.normalization import canonicalize_sample
        coverage['normalization_failed'] = sum(canonicalize_sample(load_pose(r['pose_path']), r)['status'] != 'ok' for r in rows.values())
        atomic_json(root/'audit.json', coverage)
        result = {'status': 'complete', 'manifests_sha256': {k: sha256_file(v) for k,v in manifests.items()},
                  'input_index_sha256': sha256_file(job['input_index']), 'finished_unix': time.time()}
        atomic_json(root/'complete.json', result)
        return result
    finally:
        for sig, handler in handlers.items(): signal.signal(sig, handler)


def require(condition, message):
    if not condition: raise ValueError(message)


def supervisor(job_root):
    root = Path(job_root).resolve(); job = json.loads((root/'job.json').read_text())
    lock = (root/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None; stopped = {'value': False}
    def stop(sig, frame):
        stopped['value'] = True
        if child is not None and child.poll() is None: child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    env = os.environ.copy(); env.update(job['environment'])
    stages = [('prepare', [job['python'], job['frozen_script'], 'prepare', '--job-root', str(root)], Path(job['data_root'])/'complete.json'),
              ('train', [job['python'], '-m', 'signrepr.training.managed', '--config', job['train_config'], '--job-root', str(root), '--device', 'cuda', '--cpu-threads', '4'], Path(job['train_root'])/'complete.json'),
              ('evaluate', [job['python'], job['frozen_evaluation_script'], '--job-root', str(root)], root/'evaluation/complete.json')]
    for phase, command, marker in stages:
        if stopped['value'] or (root/'pause.request').exists(): return 0
        if marker.exists(): continue
        logpath = root/'logs'/f'{phase}.log'; logpath.parent.mkdir(exist_ok=True)
        with logpath.open('a') as log:
            log.write(f'\n--- attempt {time.ctime()} ---\n'); log.flush()
            child = subprocess.Popen(command, env=env, cwd=job['workspace'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            atomic_json(root/'supervisor.json', {'status': 'running', 'phase': phase, 'pid': os.getpid(), 'child_pid': child.pid,
                                                'command': command, 'started_unix': time.time(), 'log': str(logpath)})
            if stopped['value']: child.send_signal(signal.SIGTERM)
            code = child.wait()
        if stopped['value'] or (root/'pause.request').exists():
            atomic_json(root/'supervisor.json', {'status': 'paused', 'phase': phase, 'exit_code': code, 'updated_unix': time.time()})
            return 0
        if code or not marker.exists():
            atomic_json(root/'supervisor.json', {'status': 'failed', 'phase': phase, 'exit_code': code, 'updated_unix': time.time()})
            return code or 1
    atomic_json(root/'supervisor.json', {'status': 'complete', 'finished_unix': time.time()})
    return 0


def setup(args):
    import yaml
    root = Path(args.job_root).resolve()
    require(not (root/'job.json').exists(), 'Job already exists: use start/resume; setup never overwrites a run')
    dataset = Path(args.dataset_root).resolve(); baseline = Path(args.baseline_root).resolve()
    workspace = PACKAGE.parents[1]
    gallery = [json.loads(line) for line in (baseline/'gallery.jsonl').read_text().splitlines()]
    queries = [json.loads(line) for line in (baseline/'queries.jsonl').read_text().splitlines()]
    require(len(gallery) == 3000 and len(queries) == 1500, 'Expected the original 3000/1500 protocol')
    with (dataset/'splits/val.csv').open() as stream: official_val = list(csv.DictReader(stream))
    require(len(official_val) == 10304, 'Expected the full original validation split')
    records = {}; splits = {'train': [], 'queries': [], 'val_all': [], 'selection_val': []}
    for group, entries in [('train', gallery), ('queries', queries)]:
        for row in entries:
            sid = Path(row['video']).stem
            splits[group].append(sid)
            records[sid] = {'source': str(dataset/'keypoints'/f'{sid}.pkl'), 'metadata': {
                'sample_id': sid, 'source_video_id': sid, 'split': 'train' if group == 'train' else 'val',
                'corpus': 'asl_citizen', 'language': 'ASL', 'signer_id': row['signer_id'],
                'gloss_id': row['gloss'], 'gloss_namespace': 'ASL:asl_citizen', 'video_path': row['video'],
                'coordinate_system': 'viewport_normalized', 'joint_layout': 'coco_wholebody_133',
                'extractor_profile': 'existing_ASLC_133_joint_cache_weights_unverified', 'extractor_weights_sha256': None,
                'score_profile': 'simcc_positive_rational_v1', 'timing_source': 'decoded_source_frame_pts', 'synthetic': False}}
    for row in official_val:
        sid = Path(row['Video file']).stem; splits['val_all'].append(sid)
        if sid in records: continue
        splits['selection_val'].append(sid)
        records[sid] = {'source': str(dataset/'keypoints'/f'{sid}.pkl'), 'metadata': {
            'sample_id': sid, 'source_video_id': sid, 'split': 'val', 'corpus': 'asl_citizen', 'language': 'ASL',
            'signer_id': row['Participant ID'], 'gloss_id': row['Gloss'], 'gloss_namespace': 'ASL:asl_citizen',
            'video_path': str(dataset/'videos'/row['Video file']), 'coordinate_system': 'viewport_normalized',
            'joint_layout': 'coco_wholebody_133', 'extractor_profile': 'existing_ASLC_133_joint_cache_weights_unverified',
            'extractor_weights_sha256': None, 'score_profile': 'simcc_positive_rational_v1',
            'timing_source': 'decoded_source_frame_pts', 'synthetic': False}}
    require(len(splits['selection_val']) == 8804 and len(records) == 13304, 'Protocol split identity mismatch')
    missing = [p for item in records.values() for p in (item['source'], item['metadata']['video_path']) if not Path(p).is_file()]
    require(not missing, f'Missing input resources: {missing[:5]} (total {len(missing)})')
    root.mkdir(parents=True)
    atomic_json(root/'inputs.json', list(records.values()))
    code = root/'code'; shutil.copytree(PACKAGE/'src/signrepr', code/'signrepr', ignore=shutil.ignore_patterns('__pycache__'))
    ops = root/'ops'; ops.mkdir()
    frozen_script = ops/'aslc3000_job.py'; shutil.copyfile(Path(__file__), frozen_script)
    evaluation_script = ops/'evaluate_aslc3000_job.py'
    shutil.copyfile(PACKAGE/'scripts/evaluate_aslc3000_job.py', evaluation_script)
    config = yaml.safe_load((PACKAGE/'configs/ssl_pose_jepa.yaml').read_text())
    config['run'].update(name='pose_jepa_aslc3000_hybrid_30k', output_dir=str(root/'train'),
                         protocol='in_domain_ssl', model_selection_scope='target_validation')
    config['data'].update(train_manifest=str(root/'data/train.jsonl'), validation_manifest=str(root/'data/selection_val.jsonl'),
                          downstream_validation_protocol=None, sampling='isolated', num_workers=4)
    config['optim'].update(per_rank_batch=32, gradient_accumulation_steps=1, max_steps=30000, warmup_steps=1000)
    config['evaluation'].update(checkpoint_interval_steps=100, frozen_validation_interval_steps=2000, early_stopping_patience=5)
    (root/'train.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    job = {'job_root': str(root), 'workspace': str(workspace), 'python': str(Path(sys.executable).absolute()),
           'evaluation_python': str(workspace/'.venv-signrep/bin/python'), 'baseline_root': str(baseline),
           'data_root': str(root/'data'), 'train_root': str(root/'train'), 'input_index': str(root/'inputs.json'),
           'train_config': str(root/'train.yaml'), 'prepare_workers': args.workers,
           'frozen_script': str(frozen_script), 'frozen_evaluation_script': str(evaluation_script), 'splits': splits,
           'original_gallery_sha256': sha256_file(baseline/'gallery.jsonl'), 'original_queries_sha256': sha256_file(baseline/'queries.jsonl'),
           'environment': {'PYTHONPATH': str(code), 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'CUDA_VISIBLE_DEVICES': '0'},
           'selection_note': 'LOSO on 8804 official validation clips excluding all 1500 ASLC300 queries. Final LOSO on all validation is target-validation-selected, not an independent test claim.'}
    atomic_json(root/'job.json', job)
    unit = f'''[Unit]
Description=Pose-JEPA ASLC3000 durable training and evaluation
StartLimitIntervalSec=3600
StartLimitBurst=5

[Service]
Type=simple
WorkingDirectory={workspace}
ExecStart={job['python']} {frozen_script} supervise --job-root {root}
Restart=on-failure
RestartSec=30
KillMode=mixed
TimeoutStopSec=300
Environment=PYTHONPATH={code}
Environment=PYTHONUNBUFFERED=1
StandardOutput=append:{root}/service.log
StandardError=append:{root}/service.log

[Install]
WantedBy=default.target
'''
    unit_path = Path.home()/'.config/systemd/user/pose-jepa-aslc3000.service'
    require(not unit_path.exists(), 'Service name already exists; refusing to overwrite')
    unit_path.parent.mkdir(parents=True, exist_ok=True); unit_path.write_text(unit)
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'enable', unit_path.name], check=True)
    return {'job_root': str(root), 'service': unit_path.name, 'split_counts': {k:len(v) for k,v in splits.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['setup', 'prepare', 'supervise', 'start', 'resume', 'stop', 'status'])
    parser.add_argument('--job-root', default=str(PACKAGE/'runtime/aslc3000_job'))
    parser.add_argument('--dataset-root'); parser.add_argument('--baseline-root'); parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(); root = Path(args.job_root).resolve(); unit = 'pose-jepa-aslc3000.service'
    if args.command == 'setup': result = setup(args)
    elif args.command == 'prepare': result = prepare(json.loads((root/'job.json').read_text()), root)
    elif args.command == 'supervise': raise SystemExit(supervisor(root))
    elif args.command in ('start', 'resume'):
        require((root/'job.json').exists(), 'Run setup before starting the job')
        (root/'pause.request').unlink(missing_ok=True)
        subprocess.run(['systemctl', '--user', 'reset-failed', unit], check=True)
        subprocess.run(['systemctl', '--user', 'start', unit], check=True)
        result = {'requested': args.command, 'service': unit}
    elif args.command == 'stop':
        (root/'pause.request').touch()
        subprocess.run(['systemctl', '--user', 'stop', unit], check=True)
        result = {'requested': 'stop', 'checkpoint_policy': 'SIGTERM saves at optimizer boundary; previous checkpoint retained'}
    else:
        result = {'service': subprocess.run(['systemctl', '--user', 'show', unit, '--property=ActiveState,SubState,MainPID,NRestarts,Result'], capture_output=True, text=True).stdout}
        for name in ('supervisor', 'progress'):
            path = root/f'{name}.json'
            if path.exists(): result[name] = json.loads(path.read_text())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
