"""Analyze a completed prediction run using its immutable complete rankings."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import file_sha
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--prediction-run', required=True, type=Path)
    args = parser.parse_args()
    status = json.loads((args.prediction_run / 'status.json').read_text())
    config = json.loads(args.config.read_text())
    if (status['status'] != 'completed' or status['protocol'] != config['protocol']
            or config['protocol'] not in ('P001_v3', 'P001_v4') or config['seed'] != 1701):
        raise ValueError('Prediction run/protocol not completed')
    for name, expected in status['output_hashes'].items():
        if file_sha(args.prediction_run / name) != expected: raise ValueError('Prediction evidence changed')
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import numpy as np
        from analysis import analyze
        schema = json.loads((args.prediction_run / 'prediction_schema.json').read_text())
        results = json.loads((args.prediction_run / 'results.json').read_text())
        if len(schema['query_ids']) != 378 or len(schema['class_ids']) != 100 or results['test_access'] != 'none':
            raise ValueError('Candidate cohort differs')
        cohort_path = REPO / config['cohort_manifest']
        if file_sha(cohort_path) != config['cohort_sha256']:
            raise ValueError('Analysis cohort identity changed')
        cohort = json.loads(cohort_path.read_text())
        dev = sorted((s for s in cohort['samples'] if s['split'] == 'val'), key=lambda s: s['sample_id'])
        if (schema['query_ids'] != [s['sample_id'] for s in dev]
                or schema['labels'] != [s['class_id'] for s in dev]
                or schema['participant_ids'] != [s['participant_id'] for s in dev]):
            raise ValueError('Prediction schema/reference mapping differs')
        with np.load(args.prediction_run / 'rankings.npz', allow_pickle=False) as arrays:
            report = analyze(schema, results['results'], {k: arrays[k] for k in arrays.files})
        report.update(run_id=args.run_id, prediction_run=status['run_id'],
                      prediction_status_sha256=file_sha(args.prediction_run / 'status.json'), numpy=np.__version__)
        run.save('analysis.json', report)
        run.status['output_sha256'] = file_sha(run.output / 'analysis.json')


if __name__ == '__main__': main()
