"""Train-derived cohort and nested low-label supports; never reads test."""
import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import sys
import time
import traceback


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def select_cohort(train, dev, classes, class_seed, support_seeds, budgets):
    def index(rows):
        result = defaultdict(list)
        seen = set()
        for row in rows:
            if not all(row[k] for k in ('Video file', 'Gloss', 'Participant ID')) or row['Video file'] in seen:
                raise ValueError('Empty metadata or duplicate sample ID')
            seen.add(row['Video file'])
            result[row['Gloss']].append(row)
        return result
    if not budgets or any(type(b) is not int or b <= 0 for b in budgets):
        raise ValueError('Invalid label budget')
    train_index, dev_index = index(train), index(dev)
    selected = sorted(random.Random(class_seed).sample(sorted(train_index), classes))
    if any(len(train_index[c]) < max(budgets) or c not in dev_index for c in selected):
        raise ValueError('Selected class lacks required support; no difficulty-based reselection')
    train_ids, dev_ids = {r['Video file'] for r in train}, {r['Video file'] for r in dev}
    if train_ids & dev_ids or {r['Participant ID'] for r in train} & {r['Participant ID'] for r in dev}:
        raise ValueError('Train/dev ID or participant leakage')
    supports = {}
    union_ids = set()
    for seed in support_seeds:
        rng = random.Random(seed)
        ordered = {}
        for label in selected:
            candidates = sorted(r['Video file'] for r in train_index[label])
            ordered[label] = rng.sample(candidates, max(budgets))
            union_ids.update(ordered[label])
        supports[str(seed)] = {str(b): {label: ordered[label][:b] for label in selected} for b in budgets}
    def record(row, split):
        return {'sample_id': row['Video file'], 'gloss': row['Gloss'], 'class_id': selected.index(row['Gloss']),
                'participant_id': row['Participant ID'], 'split': split}
    samples = [record(r,'train') for r in train if r['Video file'] in union_ids]
    samples += [record(r,'val') for r in dev if r['Gloss'] in selected]
    return {'classes': selected, 'class_map_origin': 'train only; selected classes sorted',
            'supports': supports, 'samples': sorted(samples, key=lambda r:(r['split'],r['sample_id']))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.seed != config['class_selection_seed']:
        raise ValueError('CLI/config selection seed mismatch')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {'run_id': args.run_id, 'status': 'running', 'GPU_hours': 0,
              'start_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(), 'host': platform.node(),
              'command': [sys.executable]+sys.argv, 'script_sha256': sha_file(Path(__file__)),
              'config_sha256': sha_file(args.config)}

    def save(name, value):
        temporary = args.output_dir/(name+'.tmp')
        temporary.write_text(json.dumps(value, indent=2)+'\n')
        temporary.replace(args.output_dir/name)

    def timeout_handler(*_):
        raise TimeoutError('Cohort audit time budget exhausted')

    save('status.json', status)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(config['timeout_seconds'])
    try:
        shutil.copyfile(__file__, args.output_dir/'build_cohort.py')
        shutil.copyfile(args.config, args.output_dir/'config.json')
        root = Path(config['dataset_root'])
        metadata = {}
        for split in ('train','val'):
            path = root/'splits'/f'{split}.csv'
            with path.open(newline='') as source:
                metadata[split] = list(csv.DictReader(source))
        cohort = select_cohort(metadata['train'], metadata['val'], config['classes'], args.seed,
                               config['support_seeds'], config['label_budgets'])
        cohort.update(run_id=args.run_id, protocol=config['protocol'], seed=args.seed,
                      config_sha256=sha_file(args.config),
                      split_hashes={s:sha_file(root/'splits'/f'{s}.csv') for s in ('train','val')},
                      test_access='none', source_read_bytes=0)
        for sample in cohort['samples']:
            path = root/'videos'/sample['sample_id']
            sample['source_bytes'] = path.stat().st_size
            sample['input_path'] = str(path)
        estimate = sum(s['source_bytes'] for s in cohort['samples'])
        if estimate > config['max_source_read_bytes']:
            raise ValueError(f'Source hash estimate {estimate} exceeds registered read budget')
        by_hash = defaultdict(list)
        for sample in cohort['samples']:
            sample['source_sha256'] = sha_file(Path(sample['input_path']))
            by_hash[sample['source_sha256']].append(sample)
            cohort['source_read_bytes'] += sample['source_bytes']
        duplicates = [group for group in by_hash.values() if len(group) > 1]
        cohort['exact_duplicate_groups'] = [[{'id':s['sample_id'],'split':s['split']} for s in group] for group in duplicates]
        cohort['cross_split_exact_duplicates'] = any(len({s['split'] for s in group}) > 1 for group in duplicates)
        cohort['near_duplicate_audit'] = 'not_run'
        cohort['counts'] = {s:sum(row['split']==s for row in cohort['samples']) for s in ('train','val')}
        cohort['passed'] = not cohort['cross_split_exact_duplicates']
        save('cohort.json', cohort)
        if not cohort['passed']:
            raise ValueError('Cross-split exact duplicate: measurement blocked')
        status.update(status='completed', exit_code=0, output_sha256=sha_file(args.output_dir/'cohort.json'))
    except Exception as error:
        status.update(status='failed', exit_code=1, error=repr(error))
        (args.output_dir/'error.txt').write_text(traceback.format_exc())
    finally:
        signal.alarm(0)
        status.update(end_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic()-started)
        save('status.json', status)
    print(json.dumps(status), flush=True)
    if status['status'] != 'completed':
        sys.exit(1)


if __name__ == '__main__':
    main()
