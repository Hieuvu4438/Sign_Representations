"""Score only complete, frozen P003 held-out predictions; rebuild all evidence."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import file_sha
from confirmatory import METHODS, analyze, check_freeze
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'freeze', 'prediction-run', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    check_freeze(args.freeze, args.config)
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import numpy as np
        status = json.loads((args.prediction_run / 'status.json').read_text())
        if status['status'] != 'completed' or status['protocol'] != config['protocol']:
            raise ValueError('Scoring requires complete frozen held-out predictions')
        for name, digest in status['output_hashes'].items():
            if file_sha(args.prediction_run / name) != digest:
                raise ValueError('Held-out prediction artifact changed')
        schema = json.loads((args.prediction_run / 'schema.json').read_text())
        timing = json.loads((args.prediction_run / 'timing.json').read_text())
        diagnostics = json.loads((args.prediction_run / 'diagnostics.json').read_text())
        thresholds = json.loads((REPO / config['train_thresholds']).read_text())
        with np.load(args.prediction_run / 'rankings.npz', allow_pickle=False) as archive:
            rankings = {name: archive[name] for name in archive.files}
        result = analyze(schema, rankings, timing['observations'], diagnostics, thresholds)
        result.update(prediction_run=status['run_id'], prediction_status_sha256=file_sha(args.prediction_run / 'status.json'),
                      freeze_sha256=file_sha(args.freeze), N_expected=len(schema['query_ids']), N_scored=len(schema['query_ids']),
                      thresholds=thresholds, timing_boundary=timing['boundary'], timing_excluded=timing['excluded'])
        run.save('analysis.json', result)
        with (run.output / 'metrics.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['method', 'DCG_fraction', 'MRR_fraction', 'Top1_fraction', 'query_N', 'p50_seconds', 'p95_seconds', 'paired_cost_ratio', 'window_reduction'])
            for method in METHODS:
                metric = result['mean_metrics'][method]
                cost = result['costs'].get(method, {})
                writer.writerow([method, metric['dcg'], metric['mrr'], metric['top1'], result['N_scored'],
                    cost.get('p50_seconds'), cost.get('p95_seconds'), cost.get('median_paired_query_ratio'), cost.get('window_reduction_factor')])
        rows = ['| Method | DCG | Top1 (%) | Query p50 (s) | Query p95 (s) | Windows reduction |',
                '| --- | ---: | ---: | ---: | ---: | ---: |']
        for method in METHODS:
            metric = result['mean_metrics'][method]
            cost = result['costs'].get(method)
            values = [f"{cost[k]:.6f}" for k in ('p50_seconds', 'p95_seconds', 'window_reduction_factor')] if cost else ['—', '—', '—']
            rows.append(f"| {method} | {metric['dcg']:.9f} | {metric['top1']*100:.4f} | {' | '.join(values)} |")
        rows += ['', f"N={result['N_scored']}, mean over all three fixed train galleries; DCG/MRR fraction, Top1 percent.",
            'Full mean shares the dense encoder pass for quality; no separate latency claim.',
            'One independent measured pipeline per query/recipe, cyclic counterbalance, shared eligible GPU, uncontrolled OS cache.',
            'Gallery extraction/setup excluded from query latency; conditional group uncertainty, no population equivalence or full-benchmark claim.']
        (run.output / 'table.md').write_text('\n'.join(rows) + '\n')
        run.status['output_hashes'] = {name: file_sha(run.output / name) for name in ('analysis.json', 'metrics.csv', 'table.md')}


if __name__ == '__main__':
    main()
