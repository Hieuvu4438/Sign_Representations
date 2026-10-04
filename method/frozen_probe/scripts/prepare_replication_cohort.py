"""Train-selected disjoint vocabulary/one-example inventory, no test CSV or pixels."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import file_sha, sample_map
from run_context import RunContext
from scripts.build_cohort import select_cohort


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        original_path = REPO / config['original_cohort']
        if file_sha(original_path) != config['original_cohort_sha256']:
            raise ValueError('Changed original vocabulary exclusion')
        original = json.loads(original_path.read_text())
        excluded = set(original['classes'])
        original_ids = {s['sample_id'] for s in original['samples']}
        original_hashes = {s['source_sha256'] for s in original['samples']}
        metadata = {}
        source_root = Path(config['dataset_root'])
        for split in ('train', 'val'):
            path = source_root / 'splits' / (split + '.csv')
            if file_sha(path) != config['split_sha256'][split]:
                raise ValueError('Official metadata changed')
            with path.open(newline='') as handle:
                metadata[split] = list(csv.DictReader(handle))
        # Exclude only the previously frozen class vocabulary; never use new test outcomes.
        filtered_train = [row for row in metadata['train'] if row['Gloss'] not in excluded]
        cohort = select_cohort(filtered_train, metadata['val'], config['classes'], args.seed,
                               config['support_seeds'], config['label_budgets'])
        if set(cohort['classes']) & excluded or config['label_budgets'] != [1]:
            raise ValueError('Replication vocabulary/budget changed')
        total_bytes = 0
        for sample in cohort['samples']:
            if Path(sample['sample_id']).name != sample['sample_id'] or sample['sample_id'] in original_ids:
                raise ValueError('Nonlocal/overlapping source ID')
            path = source_root / 'videos' / sample['sample_id']
            sample['source_bytes'] = path.stat().st_size
            sample['input_path'] = str(path)
            total_bytes += sample['source_bytes']
        if total_bytes > config['max_source_read_bytes']:
            raise ValueError('Source digest read allowance exceeded; no resampling')
        seen = set()
        for sample in cohort['samples']:
            digest = file_sha(Path(sample['input_path']))
            if digest in seen or digest in original_hashes:
                raise ValueError('Exact byte duplicate; no exclusions or reselection')
            sample['source_sha256'] = digest
            seen.add(digest)
        sample_map(cohort)  # validates source/class/split/participant contracts.
        counts = {split: sum(s['split'] == split for s in cohort['samples']) for split in ('train', 'val')}
        cohort.update(run_id=args.run_id, protocol=config['protocol'], seed=args.seed,
            config_sha256=file_sha(args.config), split_hashes=config['split_sha256'], counts=counts,
            class_exclusion_origin='immutable original COHORT001 class list only',
            original_cohort_sha256=config['original_cohort_sha256'], source_read_bytes=total_bytes,
            passed=True, test_access='none: no test CSV, pixels, features, predictions or new task scores read',
            regime='prospective replication feasibility inventory; no model or quality/cost measurement',
            overlap_scope='exact IDs/bytes disjoint from original train/dev cohort; sessions/pretraining unknown; same corpus/participant populations')
        run.save('cohort.json', cohort)
        run.save('inventory.json', dict(classes=len(cohort['classes']), counts=counts,
            participants={split: sorted({s['participant_id'] for s in cohort['samples'] if s['split'] == split})
                          for split in ('train', 'val')}, support_seeds=config['support_seeds'],
            labels_per_class=1, source_read_bytes=total_bytes, vocabulary_disjoint=True,
            outcome_access='No new model score computed; P003 original results already known and disclosed',
            next_gate='native source/PTS audit and train-gallery/test-query cost estimate before any GPU or new selected test access'))
        run.status['output_hashes'] = {name: file_sha(run.output / name) for name in ('cohort.json', 'inventory.json')}


if __name__ == '__main__':
    main()
