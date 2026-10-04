"""Fixed P004 dev quality and train-only thresholds; no CUDA/test access."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import CacheReader, file_sha
from confirmatory import METHODS, SEEDS, build_galleries, search, train_thresholds
from replication import check_gate, quality_report, validate_cohort
from run_context import RunContext
from window_budget import pooled_vector, selected_indices


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'output-dir', 'launch-gate', 'cache-run'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume:
        raise ValueError('No partial quality resume; register a new run')
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import numpy as np
        check_gate(args.launch_gate, args.config, REPO)
        if (config['methods'] != list(METHODS) or config['support_seeds'] != list(SEEDS)
                or config['gallery_budget'] != 1 or config['dcg_loss_margin'] != .02):
            raise ValueError('Registered fixed pilot changed')
        status = json.loads((args.cache_run / 'status.json').read_text())
        report = json.loads((args.cache_run / 'report.json').read_text())
        if (status['status'] != 'completed' or not report['passed']
                or status['output_sha256'] != file_sha(args.cache_run / 'report.json')
                or report['registration_sha256'] != file_sha(args.launch_gate)):
            raise ValueError('Incomplete/unbound native cache run')
        cohort = json.loads((REPO / config['cohort_manifest']).read_text())
        audit = json.loads((REPO / config['video_audit']).read_text())
        indexed, videos = validate_cohort(cohort, audit, config['expected'], config['cache_identity']['contract'])
        cache = Path(report['cache_manifest'])
        if file_sha(cache) != report['cache_manifest_sha256']:
            raise ValueError('Native cache changed')
        reader = CacheReader(cache, config['cache_identity'], cohort, audit)
        galleries, gallery_metadata = build_galleries(reader, cohort, budget=1)
        gallery_metadata['origin'] = 'validated P004 native cache; fixed train-only one-example galleries'
        ids = sorted(k for k, s in indexed.items() if s['split'] == 'val')
        schema = dict(query_ids=ids, labels=[indexed[k]['class_id'] for k in ids],
                      participant_ids=[indexed[k]['participant_id'] for k in ids],
                      class_ids=list(range(len(cohort['classes']))), split='val',
                      support_seeds=list(SEEDS), gallery_budget=1)
        rankings = {f'{m}_s{s}': [] for m in METHODS for s in SEEDS}
        selections, counts = {}, {m: 0 for m in METHODS}
        for key in ids:
            features, activity, centers = reader.load(key)
            selections[key] = {}
            for method in METHODS:
                indices = selected_indices(method, len(centers), indexed[key]['source_sha256'])
                selections[key][method] = indices
                counts[method] += len(indices)
                vector = pooled_vector(features, activity, method, indices)
                for seed, rank in search(vector, galleries[method], len(cohort['classes'])).items():
                    rankings[f'{method}_s{seed}'].append(rank)
        rankings = {name: np.asarray(rows, dtype=np.int16) for name, rows in rankings.items()}
        result = quality_report(schema, rankings)
        result.update(run_id=args.run_id, test_access='none', query_windows=counts,
                      gallery=gallery_metadata, cache_manifest_sha256=file_sha(cache),
                      registration_sha256=file_sha(args.launch_gate), numpy=np.__version__)
        result['window_count_screen'] = bool(counts['uniform8'] * 2 <= counts['full_activity'])
        result['continue_to_test_preparation'] = bool(result['quality_screen'] and result['window_count_screen'])
        run.save('quality.json', result)
        run.save('schema.json', schema)
        run.save('selections.json', selections)
        run.save('train_thresholds.json', train_thresholds(reader, cohort, audit))
        np.savez_compressed(run.output / 'rankings.npz', **rankings)
        run.status['output_hashes'] = {name: file_sha(run.output / name) for name in (
            'quality.json', 'schema.json', 'selections.json', 'train_thresholds.json', 'rankings.npz')}


if __name__ == '__main__':
    main()
