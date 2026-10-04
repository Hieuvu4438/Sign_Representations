"""Rebuild train-only registration artifacts and prove byte parity; no new fit rule."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import CacheReader, file_sha
from confirmatory import train_thresholds
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import numpy as np
        original = REPO / config['original_config']
        if file_sha(original) != config['original_config_sha256']:
            raise ValueError('Original train-only registration config changed')
        recipe = json.loads(original.read_text())
        cohort = json.loads((REPO / recipe['cohort_manifest']).read_text())
        audit = json.loads((REPO / recipe['video_audit']).read_text())
        reader = CacheReader(recipe['cache_manifest'], recipe['cache_identity'], cohort, audit)
        thresholds = train_thresholds(reader, cohort, audit)
        timing_path = REPO / config['timing_report']
        if file_sha(timing_path) != config['timing_report_sha256']:
            raise ValueError('Representative timing input changed')
        timing = json.loads(timing_path.read_text())
        frames = {s['sample_id']: s['frames'] for s in audit['samples']}
        rows = timing['observations']
        rates = dict(encoder_seconds_per_window=float(np.quantile(
            [r['encoder_wrapper_seconds']/r['encoded_windows'] for r in rows], .9)),
            decode_preprocess_seconds_per_frame=float(np.quantile(
                [(r['decode_seconds']+r['preprocess_seconds'])/frames[r['sample_id']] for r in rows], .9)),
            pool_search_seconds_per_query_recipe=float(np.quantile(
                [r['pool_seconds']+r['three_gallery_search_seconds'] for r in rows], .9)),
            safety_factor=1.5, setup_seconds=60, source_run=timing['run_id'], source_sha256=file_sha(timing_path))
        run.save('train_thresholds.json', thresholds)
        run.save('representative_rates.json', rates)
        for name, digest in config['expected_output_hashes'].items():
            if file_sha(run.output / name) != digest:
                raise ValueError('Rebuilt train registration does not exactly match the pre-test artifact')
        run.save('verification.json', dict(passed=True, byte_parity=True, fit_split='train only',
            original_run='THR001_20261004_P003v1_s1701', new_scientific_thresholds=False, test_access='none',
            expected_hashes=config['expected_output_hashes'],
            reason='THR001 historical Python stdin argv omitted its driver; this versioned command closes replay provenance'))
        run.status['output_hashes'] = {name: file_sha(run.output / name) for name in
            ('train_thresholds.json', 'representative_rates.json', 'verification.json')}


if __name__ == '__main__':
    main()
