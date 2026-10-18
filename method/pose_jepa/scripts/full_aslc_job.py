"""Full ASL-Citizen: detached systemd training, validated cache reuse and resume."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import csv
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
if (PACKAGE/'src').is_dir(): sys.path.insert(0, str(PACKAGE/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aslc3000_job import prepare_one, file_identity
from signrepr.data.schema import load_pose, sha256_file, write_manifest
from signrepr.data.normalization import canonicalize_sample
from signrepr.training.job_support import atomic_json

UNIT = 'pose-jepa-full-aslc.service'
COUNTS = {'train':40154, 'val':10304, 'test':32941}


def require(value, message):
    if not value: raise ValueError(message)


def build_inputs(dataset, expected_counts=COUNTS):
    records, splits, signers = {}, {}, {}
    for split in ('train', 'val', 'test'):
        with (dataset/'splits'/f'{split}.csv').open() as file: rows = list(csv.DictReader(file))
        require(len(rows) == expected_counts[split], f'Unexpected {split} membership')
        splits[split] = []; signers[split] = set()
        for row in rows:
            sid = Path(row['Video file']).stem
            require(sid not in records and row['Participant ID'] and row['Gloss'], f'Duplicate/missing metadata: {sid}')
            splits[split].append(sid); signers[split].add(row['Participant ID'])
            metadata = {'sample_id':sid, 'source_video_id':sid, 'split':split, 'corpus':'asl_citizen', 'language':'ASL',
                        'signer_id':row['Participant ID'], 'gloss_id':row['Gloss'], 'gloss_namespace':'ASL:asl_citizen',
                        'video_path':str(dataset/'videos'/row['Video file']), 'coordinate_system':'viewport_normalized',
                        'joint_layout':'coco_wholebody_133', 'extractor_profile':'existing_ASLC_133_joint_cache_weights_unverified',
                        'extractor_weights_sha256':None, 'score_profile':'simcc_positive_rational_v1',
                        'timing_source':'decoded_source_frame_pts', 'synthetic':False}
            source = dataset/'keypoints'/f'{sid}.pkl'
            require(source.is_file() and Path(metadata['video_path']).is_file(), f'Missing source: {sid}')
            records[sid] = {'source':str(source), 'metadata':metadata}
    for a,b in [('train','val'), ('train','test'), ('val','test')]:
        require(not signers[a] & signers[b], f'Official signer overlap: {a}/{b}')
    return records, splits, signers


def cache_record_valid(receipt, identity):
    require(receipt['input_identity'] == identity, 'Prepared source metadata changed')
    row = receipt['row']
    require(sha256_file(row['pose_path']) == row['pose_sha256'], 'Prepared pose checksum mismatch')
    load_pose(row['pose_path'])
    require(sha256_file(row['raw_detector_scores_path']) == row['raw_detector_scores_sha256'], 'Raw score checksum mismatch')
    return row


def verified_signer_aliases(records, reuse_root):
    aliases = {}
    for path in (reuse_root/'receipts').glob('*.json'):
        old = json.loads(path.read_text()); sid = old['row']['sample_id']
        if sid not in records: continue
        old_id = str(old['input_identity']['metadata']['signer_id'])
        official = records[sid]['metadata']['signer_id']
        require(old_id not in aliases or aliases[old_id] == official, 'Ambiguous cached signer alias')
        aliases[old_id] = official
    return aliases


def prepare_clip(task):
    root = Path(task['root']); record = task['record']; meta = record['metadata']
    identity = {'pose':file_identity(Path(record['source'])), 'video':file_identity(Path(meta['video_path'])), 'metadata':meta}
    token = hashlib.sha256(meta['sample_id'].encode()).hexdigest()
    receipt_path = root/'receipts'/f'{token}.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text()); row = cache_record_valid(receipt, identity)
        return row, receipt.get('cache_reused', False), receipt['normalization_status']
    old_path = Path(task['reuse_root'])/'receipts'/f'{token}.json'
    reused = False
    if old_path.exists():
        old = json.loads(old_path.read_text()); old_identity = old['input_identity']
        require(old_identity['pose'] == identity['pose'] and old_identity['video'] == identity['video'], 'Reuse source changed')
        for key in ('sample_id','split','coordinate_system','joint_layout','score_profile','timing_source','signer_roi'):
            require(old_identity['metadata'].get(key) == meta.get(key), f'Incompatible reuse profile: {key}')
        old_signer = str(old_identity['metadata']['signer_id'])
        require(task.get('signer_aliases',{}).get(old_signer,old_signer) == str(meta['signer_id']), 'Reuse signer mismatch')
        row = cache_record_valid(old, old_identity)
        require(sha256_file(record['source']) == row['source_cache_sha256'], 'Reuse original pose source hash changed')
        # Reuse pose bytes only; authoritative labels always come from official CSV.
        row = {**row, **meta}; reused = True
    else:
        row = prepare_one({'root':str(root), **record})
    status = canonicalize_sample(load_pose(row['pose_path']), row)['status']
    atomic_json(receipt_path, {'row':row, 'input_identity':identity, 'cache_reused':reused, 'normalization_status':status})
    return row, reused, status


def audit_rows(rows_by_split):
    overlaps = {}
    for a,b in [('train','val'), ('train','test'), ('val','test')]:
        if a not in rows_by_split or b not in rows_by_split: continue
        left, right = rows_by_split[a], rows_by_split[b]
        require(not {r['sample_id'] for r in left} & {r['sample_id'] for r in right}, 'Split ID overlap')
        require(not {r['signer_id'] for r in left} & {r['signer_id'] for r in right}, 'Split signer overlap')
        for key in ('video_sha256','pose_sha256'):
            common = {r[key] for r in left} & {r[key] for r in right}
            overlaps[f'{a}_{b}_{key}'] = len(common)
            require(not common, f'Exact cross-split content duplicate: {a}/{b}/{key}')
    return overlaps


def prepare(job, job_root, include_test=False):
    root = Path(job['data_root']); root.mkdir(parents=True, exist_ok=True)
    (root/'receipts').mkdir(exist_ok=True)
    names = ['test'] if include_test else ['train','val']
    records = [r for r in json.loads(Path(job['input_index']).read_text()) if r['metadata']['split'] in names]
    stopped = {'value':False}
    def stop(number, frame): stopped['value'] = True
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig, stop)
    rows = {}; reused = 0; failed = []; started = time.time(); iterator = iter(records)
    with ProcessPoolExecutor(job['prepare_workers'], mp_context=mp.get_context('spawn')) as pool:
        pending = {}
        def schedule():
            while not stopped['value'] and not (job_root/'pause.request').exists() and len(pending) < 2*job['prepare_workers']:
                try: record = next(iterator)
                except StopIteration: break
                task = {'root':str(root), 'record':record, 'reuse_root':job['reuse_data_root'],
                        'signer_aliases':job['verified_cache_signer_aliases']}
                pending[pool.submit(prepare_clip, task)] = record['metadata']['sample_id']
        schedule()
        while pending:
            done,_ = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
            for future in done:
                sid = pending.pop(future); row, cached, status = future.result()
                rows[sid] = row; reused += int(cached)
                if status != 'ok': failed.append(sid)
            require(shutil.disk_usage(job_root).free > 6*2**30, 'Less than 6GiB free disk; cache checkpoint retained')
            progress = {'phase':'prepare_test' if include_test else 'prepare_train_val', 'status':'running',
                        'pid':os.getpid(), 'completed':len(rows), 'total':len(records), 'reused_existing':reused,
                        'normalization_failed':len(failed), 'elapsed_seconds':time.time()-started, 'updated_unix':time.time()}
            atomic_json(job_root/'progress.json', progress)
            (job_root/'progress.txt').write_text(f"Pose-JEPA full ASL-Citizen | {progress['phase']}\nClips: {len(rows)}/{len(records)}\nReused: {reused}\nUpdated: {time.ctime()}\n")
            schedule()
    if stopped['value'] or (job_root/'pause.request').exists(): return {'status':'paused', 'completed':len(rows)}
    require(len(rows) == len(records), 'Incomplete cache preparation')
    by_split = {name:[rows[sid] for sid in job['splits'][name]] for name in names}
    if include_test:
        from signrepr.data.schema import read_manifest
        by_split.update({name:read_manifest(root/f'{name}.jsonl') for name in ('train','val')})
    overlap = audit_rows(by_split)
    for name in names: write_manifest(root/f'{name}.jsonl', by_split[name])
    result = {'status':'complete', 'samples':len(rows), 'frames':sum(r['num_frames'] for r in rows.values()),
              'reused_existing':reused, 'normalization_failed_ids':sorted(failed), 'exact_content_overlap':overlap,
              'split_counts':{name:len(by_split[name]) for name in names},
              'manifests_sha256':{name:sha256_file(root/f'{name}.jsonl') for name in names},
              'timing':'native video PTS', 'score_profile':'q=s/(1+s), uncalibrated reliability',
              'input_index_sha256':sha256_file(job['input_index']), 'finished_unix':time.time()}
    atomic_json(root/('test.complete.json' if include_test else 'trainval.complete.json'), result)
    return result


def verify_job(root, job):
    lock = json.loads((root/'job_lock.json').read_text())
    from signrepr.training.provenance import source_hash
    require(source_hash() == lock['code_sha256'], 'Pinned training source changed')
    for name,digest in lock['ops_sha256'].items():
        require(sha256_file(root/'ops'/name) == digest, f'Pinned job script changed: {name}')
    for name,digest in lock['files_sha256'].items():
        require(sha256_file(root/name) == digest, f'Pinned job file changed: {name}')
    for split,digest in job['split_csv_sha256'].items():
        require(sha256_file(Path(job['dataset_root'])/'splits'/f'{split}.csv') == digest, 'Official split CSV changed')


def available_gpu_mib(job):
    gpu = job['environment'].get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    output = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free',
                                      '--format=csv,noheader,nounits'], text=True, timeout=15)
    return int(output.strip().splitlines()[0])


def wait_for_gpu(root, job, phase, stopped, cooldown_seconds=0, probe=None, sleep=None):
    """Wait without CUDA allocation or systemd failure retries; obey pause/SIGTERM."""
    probe = probe or (lambda: available_gpu_mib(job)); sleep = sleep or time.sleep
    minimum = 6144; started = time.monotonic()
    while not stopped():
        error = None
        try: free = probe()
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            free, error = None, str(exc)
        cooling = time.monotonic()-started < cooldown_seconds
        if free is not None and free >= minimum and not cooling: return True
        checkpoint = root/'train/checkpoint.json'
        step = json.loads(checkpoint.read_text())['step'] if checkpoint.exists() else 0
        state = {'phase':phase, 'status':'waiting_for_gpu', 'pid':os.getpid(), 'step':step,
                 'max_steps':job['max_steps'], 'free_vram_mib':free, 'required_free_vram_mib':minimum,
                 'gpu_probe_error':error, 'updated_unix':time.time(), 'retry_poll_seconds':30}
        atomic_json(root/'progress.json',state)
        atomic_json(root/'supervisor.json',{'status':'waiting_for_gpu','phase':phase,'pid':os.getpid(),
                                          'updated_unix':state['updated_unix']})
        (root/'progress.txt').write_text(f"Pose-JEPA full ASL-Citizen | waiting_for_gpu\nPhase: {phase}\n"
            f"Checkpoint: {step}/{job['max_steps']}\nFree VRAM: {free} MiB; required: {minimum} MiB\nUpdated: {time.ctime()}\n")
        for _ in range(30):
            if stopped(): return False
            sleep(1)
    return False


def attempt_was_cuda_oom(logpath, offset):
    with logpath.open('rb') as log:
        log.seek(0, os.SEEK_END); end = log.tell()
        log.seek(max(offset,end-65536)); text = log.read().decode('utf-8',errors='replace')
    return 'torch.OutOfMemoryError: CUDA out of memory' in text or 'torch.cuda.OutOfMemoryError: CUDA out of memory' in text


def supervisor(root):
    job = json.loads((root/'job.json').read_text()); verify_job(root, job)
    lock = (root/'supervisor.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None; stopped = {'value':False}
    def stop(number, frame):
        stopped['value'] = True
        if child and child.poll() is None: child.send_signal(signal.SIGTERM)
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig, stop)
    env = os.environ.copy(); env.update(job['environment'])
    script = str(root/'ops/full_aslc_job.py'); python = job['python']
    stages = [
        ('prepare_train_val', [python,script,'prepare','--job-root',str(root)], root/'data/trainval.complete.json'),
        ('train', [python,'-m','signrepr.training.managed','--config',job['train_config'],'--job-root',str(root),'--device','cuda','--cpu-threads','4'], root/'train/complete.json'),
        ('prepare_test', [python,script,'prepare-test','--job-root',str(root)], root/'data/test.complete.json'),
        ('evaluate', [python,str(root/'ops/evaluate_full_aslc.py'),'--job-root',str(root)], root/'evaluation/complete.json')]
    for phase, command, marker in stages:
        if stopped['value'] or (root/'pause.request').exists(): return 0
        if marker.exists(): continue
        logpath = root/'logs'/f'{phase}.log'; logpath.parent.mkdir(exist_ok=True)
        cooldown = 0
        while True:
            verify_job(root, job)
            if phase in ('train','evaluate'):
                ready = wait_for_gpu(root,job,phase,lambda:stopped['value'] or (root/'pause.request').exists(),cooldown)
                if not ready:
                    atomic_json(root/'supervisor.json',{'status':'paused','phase':phase}); return 0
            with logpath.open('a') as log:
                log.write(f'\n--- attempt {time.ctime()} ---\n'); log.flush()
                offset = logpath.stat().st_size
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                atomic_json(root/'supervisor.json', {'status':'running','phase':phase,'pid':os.getpid(), 'child_pid':child.pid,
                                                   'started_unix':time.time(), 'log':str(logpath)})
                if stopped['value']: child.terminate()
                code = child.wait()
            if stopped['value'] or (root/'pause.request').exists():
                atomic_json(root/'supervisor.json', {'status':'paused','phase':phase,'exit_code':code}); return 0
            if code and phase in ('train','evaluate') and attempt_was_cuda_oom(logpath,offset):
                # A resource race after the free-memory probe must not exhaust
                # systemd's finite retries. Child exited; reload committed state.
                with (root/'resource_events.jsonl').open('a') as events:
                    events.write(json.dumps({'phase':phase,'event':'cuda_oom_wait_and_resume','unix':time.time(),
                                             'exit_code':code,'log':str(logpath),'attempt_offset':offset})+'\n')
                cooldown = 120
                continue
            if code or not marker.exists():
                atomic_json(root/'supervisor.json', {'status':'failed','phase':phase,'exit_code':code}); return code or 1
            break
    atomic_json(root/'supervisor.json', {'status':'complete','finished_unix':time.time()}); return 0


def setup(args):
    import yaml
    root = Path(args.job_root).resolve(); dataset = Path(args.dataset_root).resolve()
    require(not root.exists(), 'Job root already exists; use status/resume, never overwrite a run')
    unit_path = Path.home()/'.config/systemd/user'/UNIT
    require(not unit_path.exists(), 'Full Pose-JEPA service already exists')
    require(shutil.disk_usage(root.parent).free > 25*2**30, 'Setup needs 25GiB free disk headroom')
    records,splits,signers = build_inputs(dataset)
    root.mkdir(parents=True); (root/'ops').mkdir()
    shutil.copytree(PACKAGE/'src/signrepr',root/'code/signrepr',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['full_aslc_job.py','aslc3000_job.py','evaluate_full_aslc.py']:
        shutil.copyfile(PACKAGE/'scripts'/name,root/'ops'/name)
    atomic_json(root/'inputs.json',list(records.values()))
    config = yaml.safe_load((PACKAGE/'configs/ssl_pose_jepa.yaml').read_text())
    max_steps = math.ceil(len(splits['train'])*100/32)
    config['run'].update(name='pose_jepa_full_aslc_hybrid_100epochs',output_dir=str(root/'train'),
                         protocol='in_domain_ssl', model_selection_scope='target_validation')
    config['data'].update(train_manifest=str(root/'data/train.jsonl'),validation_manifest=str(root/'data/val.jsonl'),
                          downstream_validation_protocol=None,sampling='isolated',num_workers=4)
    config['optim'].update(per_rank_batch=32,gradient_accumulation_steps=1,max_steps=max_steps,warmup_steps=5000,
                           cuda_memory_fraction=0.15)
    config['evaluation'].update(checkpoint_interval_steps=100,frozen_validation_interval_steps=5000,
                                early_stopping_patience=5,selection_protocol='dictionary')
    (root/'train.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    workspace = PACKAGE.parents[1]
    job = {'job_root':str(root),'workspace':str(workspace),'dataset_root':str(dataset),'python':str(Path(sys.executable).absolute()),
           'data_root':str(root/'data'),'reuse_data_root':str(Path(args.reuse_data_root).resolve()),
           'verified_cache_signer_aliases':verified_signer_aliases(records,Path(args.reuse_data_root).resolve()),
           'input_index':str(root/'inputs.json'),'train_config':str(root/'train.yaml'),'prepare_workers':4,
           'splits':splits,'split_csv_sha256':{s:sha256_file(dataset/'splits'/f'{s}.csv') for s in splits},
           'max_steps':max_steps,'epochs_target':100,'service':UNIT,
           'environment':{'PYTHONPATH':str(root/'code'),'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4',
                          'MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','CUDA_VISIBLE_DEVICES':'0'},
           'selection_note':'Full official train gallery -> official validation queries. Official test is only encoded/evaluated after SSL checkpoint selection. Frozen linear probe selects its head on validation, then evaluates test.'}
    atomic_json(root/'job.json',job)
    env = os.environ.copy(); env.update(job['environment'])
    digest = subprocess.check_output([job['python'],'-c','from signrepr.training.provenance import source_hash; print(source_hash())'],env=env,text=True).strip()
    atomic_json(root/'job_lock.json', {'code_sha256':digest,'ops_sha256':{p.name:sha256_file(p) for p in (root/'ops').glob('*.py')},
                                      'files_sha256':{name:sha256_file(root/name) for name in ['job.json','inputs.json','train.yaml']}})
    atomic_json(root/'setup.json', {'split_counts':{s:len(v) for s,v in splits.items()},'signer_counts':{s:len(v) for s,v in signers.items()},
                                   'classes':len({r['metadata']['gloss_id'] for r in records.values()}),'max_steps':max_steps,
                                   'warmup_steps':5000,'initialization':'scratch; cached poses reused only','created_unix':time.time()})
    unit = f'''[Unit]
Description=Pose-JEPA full ASL-Citizen training and held-out evaluation
StartLimitIntervalSec=3600
StartLimitBurst=5

[Service]
Type=simple
WorkingDirectory={workspace}
ExecStart={job['python']} {root}/ops/full_aslc_job.py supervise --job-root {root}
Restart=on-failure
RestartSec=30
KillMode=mixed
TimeoutStopSec=300
Nice=5
CPUQuota=800%
MemoryHigh=12G
MemoryMax=16G
Environment=PYTHONPATH={root}/code
Environment=PYTHONUNBUFFERED=1
StandardOutput=append:{root}/service.log
StandardError=append:{root}/service.log

[Install]
WantedBy=default.target
'''
    unit_path.parent.mkdir(parents=True,exist_ok=True); unit_path.write_text(unit)
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','enable',UNIT],check=True)
    return {'job_root':str(root),'service':UNIT,'max_steps':max_steps,'split_counts':COUNTS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['setup','prepare','prepare-test','supervise','start','resume','stop','status'])
    parser.add_argument('--job-root',default=str(PACKAGE/'runtime/full_aslc_job'))
    parser.add_argument('--dataset-root',default='/home/dongvk/datasets/ASL_Citizen')
    parser.add_argument('--reuse-data-root',default=str(PACKAGE/'runtime/aslc3000_job/data'))
    args = parser.parse_args(); root = Path(args.job_root).resolve()
    if args.command == 'setup': result = setup(args)
    elif args.command in ('prepare','prepare-test'):
        result = prepare(json.loads((root/'job.json').read_text()),root,args.command == 'prepare-test')
    elif args.command == 'supervise': raise SystemExit(supervisor(root))
    elif args.command in ('start','resume'):
        require((root/'job.json').exists(),'Setup the full job first')
        (root/'pause.request').unlink(missing_ok=True)
        subprocess.run(['systemctl','--user','reset-failed',UNIT],check=True)
        subprocess.run(['systemctl','--user','start',UNIT],check=True)
        result = {'requested':args.command,'service':UNIT}
    elif args.command == 'stop':
        require((root/'job.json').exists(),'Full job is not set up')
        (root/'pause.request').touch(); subprocess.run(['systemctl','--user','stop',UNIT],check=True)
        result = {'requested':'stop','policy':'SIGTERM saves current optimizer boundary; cache receipts persist'}
    else:
        result = {'service':subprocess.check_output(['systemctl','--user','show',UNIT,'--property=ActiveState,SubState,MainPID,NRestarts,Result'],text=True)}
        for name in ('setup','supervisor','progress'):
            if (root/f'{name}.json').exists(): result[name] = json.loads((root/f'{name}.json').read_text())
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__': main()
