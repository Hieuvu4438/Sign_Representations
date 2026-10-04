"""Registered post-outcome factorial from existing pooled vectors, no GPU inference."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import CacheReader, file_sha
from confirmatory import RECIPES, SEEDS, build_galleries, check_freeze, search
from factorial import pair_key, summarize_pairs
from prediction_contract import verify_pooled_arrays
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'gate', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import numpy as np
        gate = json.loads(args.gate.read_text())
        if (gate['verdict'] != 'REGISTERED_POST_OUTCOME_EXPLORATORY'
                or gate['config_sha256'] != file_sha(args.config)
                or config['methods'] != list(RECIPES) or config['support_seeds'] != list(SEEDS)
                or config['new_policy_trials'] != 0 or config['GPU_hours'] != 0):
            raise ValueError('Unregistered factorial or changed original policy set')
        for name, digest in gate['file_hashes'].items():
            if file_sha(REPO / name) != digest:
                raise ValueError('Exploratory source binding changed')
        check_freeze(REPO / gate['scientific_gate'], REPO / 'method/frozen_probe/configs/CONFIRM001.json')
        directories = {}
        for role in ('prediction', 'analysis', 'verification'):
            directory = REPO / config[role + '_run']
            status = json.loads((directory / 'status.json').read_text())
            if status['status'] != 'completed' or file_sha(directory / 'status.json') != config[role + '_status_sha256']:
                raise ValueError('Changed/incomplete factorial input')
            for name, digest in status['output_hashes'].items():
                if file_sha(directory / name) != digest:
                    raise ValueError('Changed original output')
            directories[role] = directory
        prediction = directories['prediction']
        verification = json.loads((directories['verification'] / 'verification.json').read_text())
        if (not verification['passed'] or not verification['all15_exact_rankings_regenerated']
                or verification['input_status_hashes']['prediction'] != config['prediction_status_sha256']):
            raise ValueError('Complete verified original predictions required')
        source_config = json.loads((prediction / 'config.json').read_text())
        cohort = json.loads((REPO / source_config['cohort_manifest']).read_text())
        audit = json.loads((REPO / source_config['video_audit']).read_text())
        schema = json.loads((prediction / 'schema.json').read_text())
        manifest = json.loads((prediction / 'pooled_manifest.json').read_text())
        payload = Path(manifest['path'])
        if file_sha(payload) != manifest['sha256'] or manifest['query_ids'] != schema['query_ids']:
            raise ValueError('Changed pooled query identity')
        with np.load(payload, allow_pickle=False) as archive:
            pooled = {name: archive[name] for name in archive.files}
        verify_pooled_arrays(pooled, len(schema['query_ids']), 768)
        reader = CacheReader(source_config['cache_manifest'], source_config['cache_identity'], cohort, audit)
        galleries, metadata = build_galleries(reader, cohort)
        rankings = {}
        for q in config['methods']:
            for g in config['methods']:
                by_seed = {str(seed): [] for seed in SEEDS}
                for vector in pooled[q]:
                    for seed, row in search(vector, galleries[g], len(schema['class_ids'])).items():
                        by_seed[seed].append(row)
                for seed in SEEDS:
                    rankings[pair_key(q, g, seed)] = np.array(by_seed[str(seed)], dtype=np.int16)
        with np.load(prediction / 'rankings.npz', allow_pickle=False) as original:
            for method in config['methods']:
                for seed in SEEDS:
                    if not np.array_equal(rankings[pair_key(method, method, seed)], original[f'{method}_s{seed}']):
                        raise ValueError('Original diagonal exact-ranking parity failed')
        summary = summarize_pairs(schema, rankings, config['methods'], SEEDS)
        original_summary = json.loads((directories['analysis'] / 'analysis.json').read_text())
        for pair in summary['per_pair']:
            if pair['query_method'] == pair['gallery_method']:
                for metric, value in pair['mean_metrics'].items():
                    if not np.isclose(value, original_summary['mean_metrics'][pair['query_method']][metric], atol=1e-14, rtol=0):
                        raise ValueError('Original diagonal metric parity failed')
        summary.update(gate_sha256=file_sha(args.gate), input_status_hashes={role: config[role+'_status_sha256']
            for role in directories}, diagonal_exact_rank_parity=True, diagonal_metric_parity=True,
            gallery_unique_sources=metadata['unique_source_count'], pooled_query_sha256=file_sha(payload),
            latency='No mixed-pair timing measured; original actual query-cost measurements remain P003 only')
        np.savez(run.output / 'rankings.npz', **rankings)
        run.save('summary.json', summary)
        rows = []
        for pair in summary['per_pair']:
            for seed, measured in pair['per_seed'].items():
                rows.append(dict(query_method=pair['query_method'], gallery_method=pair['gallery_method'],
                                 support_seed=seed, N=summary['N_scored'], **measured))
        with (run.output / 'metrics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        means = {(p['query_method'], p['gallery_method']): p['mean_metrics']['dcg'] for p in summary['per_pair']}
        lines = ['# Post-outcome query/gallery factorial (descriptive)', '',
                 'All1217queries/100classes/11participants; mean DCG across the same three fixed eight-per-class train galleries.',
                 'Rows=query recipe; columns=gallery recipe. All12original activity-method diagonal rankings match exactly.', '',
                 '| Query / gallery | ' + ' | '.join(config['methods']) + ' |',
                 '| --- | ' + ' | '.join(['---:'] * len(config['methods'])) + ' |']
        for q in config['methods']:
            lines.append('| ' + q + ' | ' + ' | '.join(f'{means[q,g]:.9f}' for g in config['methods']) + ' |')
        lines += ['', '| Sparse recipe | Query-only DCG delta | Gallery-only DCG delta | Joint delta | Interaction |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for method, row in summary['contrasts'].items():
            values = row['mean']['dcg']
            lines.append('| ' + method + ' | ' + ' | '.join(f'{values[name]:.9f}' for name in
                         ('query_only', 'gallery_only', 'joint', 'interaction')) + ' |')
        lines += ['', 'Post-outcome exploratory; no new policy selection, primary metric or causal/linguistic claim.',
                  'All seed metrics, participant denominators/deltas and Top1 contrasts retained in summary/CSV.',
                  'These mixed query/gallery pairs have no measured mixed-pair query latency.']
        (run.output / 'table.md').write_text('\n'.join(lines) + '\n')
        run.status['output_hashes'] = {name: file_sha(run.output / name) for name in config['outputs']}


if __name__ == '__main__':
    main()
