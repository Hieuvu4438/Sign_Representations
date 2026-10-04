"""Read-only bounded inventory; never imports project methods or loads weights."""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[3]
ROOTS = [
    '/home/dongvk/datasets/ASL_Citizen', '/home/dongvk/datasets/WLASL2000',
    '/home/dongvk/datasets/MS-ASL', '/home/shared_data/sign_language/How2Sign',
    '/home/dongvk/datasets/How2Sign', '/home/shared_data/sign_language/CSLDaily',
    '/home/dongvk/datasets/CSL_Daily_Sentence_Crop',
    '/home/dongvk/datasets/phoenix14T', '/home/dongvk/datasets/youtubeASL',
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def command(args):
    try:
        p = subprocess.run(args, text=True, capture_output=True, timeout=15)
        return {'argv': args, 'exit_code': p.returncode,
                'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'argv': args, 'error': str(e)}


def audit_citizen(root):
    rows, summary, missing = {}, {}, {}
    # Each filename is checked directly; no recursive corpus scan or decoding.
    for split in ['train', 'val', 'test']:
        path = root / 'splits' / f'{split}.csv'
        with path.open(newline='') as f:
            reader = csv.DictReader(f)
            required = {'Participant ID', 'Video file', 'Gloss', 'ASL-LEX Code'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f'{path}: unsupported columns {reader.fieldnames}')
            rows[split] = list(reader)
        ids = [r['Video file'] for r in rows[split]]
        if any(not x or Path(x).name != x for x in ids):
            raise ValueError('Empty or non-basename video ID')
        missing[split] = [x for x in ids if not (root / 'videos' / x).is_file()]
        support = {}
        for r in rows[split]:
            support[r['Gloss']] = support.get(r['Gloss'], 0) + 1
        summary[split] = {
            'csv_path': str(path), 'csv_sha256': sha256(path), 'rows': len(ids),
            'unique_video_ids': len(set(ids)), 'duplicate_video_rows': len(ids)-len(set(ids)),
            'participants': len({r['Participant ID'] for r in rows[split]}),
            'empty_participant_rows': sum(not r['Participant ID'] for r in rows[split]),
            'empty_gloss_rows': sum(not r['Gloss'] for r in rows[split]),
            'gloss_classes': len(support), 'min_class_support': min(support.values()),
            'max_class_support': max(support.values()),
            'available_video_files': len(ids)-len(missing[split]),
            'missing_video_files': len(missing[split]),
        }
    overlaps = {}
    for a, b in itertools.combinations(rows, 2):
        overlaps[a+'__'+b] = {
            'video_ids': len({r['Video file'] for r in rows[a]} & {r['Video file'] for r in rows[b]}),
            'participant_ids': len({r['Participant ID'] for r in rows[a]} & {r['Participant ID'] for r in rows[b]}),
        }
    train_labels = {r['Gloss'] for r in rows['train']}
    return {'splits': summary, 'overlap_counts': overlaps,
            'non_train_glosses': {s: sorted({r['Gloss'] for r in rows[s]}-train_labels)
                                 for s in ['val', 'test']},
            'not_audited': ['video decoding', 'near duplicate content', 'session IDs',
                            'participant identity correctness', 'pretraining overlap',
                            'official release equality', 'license local provenance']}, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--citizen-root', type=Path, default=Path(ROOTS[0]))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    status = {'run_id': args.run_id, 'status': 'running', 'pid': os.getpid(),
              'host': platform.node(), 'start_utc': datetime.now(timezone.utc).isoformat(),
              'script_sha256': sha256(__file__)}
    def write(name, value):
        temp = args.output_dir / (name+'.tmp')
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
        temp.replace(args.output_dir / name)
    write('status.json', status)
    try:
        disk = shutil.disk_usage(REPO)
        inventory = {
            **status, 'python': platform.python_version(),
            'root': str(REPO), 'script_sha256': sha256(__file__),
            'config': {'citizen_root': str(args.citizen_root), 'dataset_roots': ROOTS},
            'git': {str(p): command(['git', '-C', str(p), 'rev-parse', 'HEAD'])
                    for p in [REPO, REPO/'third_party/SignRep', REPO/'third_party/SHuBERT']},
            'rules_hashes': {str(p.relative_to(REPO)): sha256(p)
                             for p in sorted((REPO/'docs/proposal2').rglob('*.md'))},
            'allocation_env': {k: os.environ[k] for k in
                               ['CUDA_VISIBLE_DEVICES', 'SLURM_JOB_ID', 'SLURM_JOB_GPUS', 'PBS_JOBID']
                               if k in os.environ},
            'gpu': command(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used',
                            '--format=csv,noheader']),
            'gpu_allocation_status': 'unknown',
            'disk': {'total_bytes': disk.total, 'free_bytes': disk.free,
                     'reserve_fraction': 0.20, 'reserve_gate_passed': disk.free/disk.total >= .20},
            'roots': {}, 'checkpoints': {},
        }
        inventory['config_sha256'] = hashlib.sha256(json.dumps(inventory['config'], sort_keys=True).encode()).hexdigest()
        for name in ROOTS:
            p = Path(name)
            try:
                inventory['roots'][name] = {'exists': p.exists(),
                    'entry_names': sorted(x.name for x in p.iterdir()) if p.is_dir() else []}
            except OSError as e:
                inventory['roots'][name] = {'error': str(e)}
        for name in ['checkpoints/signrep/ckpt.pt', 'checkpoints/shubert_author_hf/models/checkpoint_836_400000.pt']:
            p = REPO / name
            inventory['checkpoints'][name] = {'path': str(p), 'exists': p.is_file()}
            if p.is_file():
                inventory['checkpoints'][name].update(bytes=p.stat().st_size, sha256=sha256(p),
                    loading_coverage='unknown', inference_parity='not_run', upstream_mapping='unknown')
        inventory['asl_citizen'], missing = audit_citizen(args.citizen_root)
        write('missing_video_ids.json', missing)
        write('inventory.json', inventory)
        status.update(status='completed', exit_code=0, wall_seconds=time.monotonic()-start,
                      end_utc=datetime.now(timezone.utc).isoformat(), GPU_hours=0,
                      output_sha256=sha256(args.output_dir/'inventory.json'))
        write('status.json', status)
        print(json.dumps(status))
    except Exception as e:
        status.update(status='failed', exit_code=1, error=repr(e), wall_seconds=time.monotonic()-start)
        write('status.json', status)
        raise


if __name__ == '__main__':
    main()
