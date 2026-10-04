"""Verify complete held-out artifacts and regenerate rankings, without task scoring."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import CacheReader, file_sha
from confirmatory import METHODS, SEEDS, build_galleries, search
from prediction_contract import verify_pooled_arrays, verify_source_recipes
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'output-dir'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import numpy as np
        prediction = REPO / config['prediction_run']
        prepared = REPO / config['prepared_run']
        statuses = {}
        for role, path in (('prediction', prediction), ('prepared', prepared)):
            status = json.loads((path/'status.json').read_text())
            if status['status'] != 'completed' or status['protocol'] != 'P004_v1':
                raise ValueError('Complete P004 execution required before verification')
            for name, digest in status['output_hashes'].items():
                if file_sha(path/name) != digest:
                    raise ValueError('Bound source/output artifact changed')
            statuses[role] = file_sha(path/'status.json')
        scientific_config = json.loads((prediction/'config.json').read_text())
        if file_sha(prediction/'config.json') != json.loads((prediction/'status.json').read_text())['config_sha256']:
            raise ValueError('Prediction config changed')
        cohort = json.loads((REPO/scientific_config['cohort_manifest']).read_text())
        old_audit = json.loads((REPO/scientific_config['video_audit']).read_text())
        test = json.loads((prepared/'test_manifest.json').read_text())
        audit = json.loads((prepared/'video_audit.json').read_text())
        schema = json.loads((prediction/'schema.json').read_text())
        timing = json.loads((prediction/'timing.json').read_text())
        counts = verify_source_recipes(cohort, test, audit, schema, timing['observations'])
        pooled_manifest = json.loads((prediction/'pooled_manifest.json').read_text())
        payload = Path(pooled_manifest['path'])
        if (pooled_manifest['query_ids'] != schema['query_ids'] or pooled_manifest['methods'] != list(METHODS)
                or payload.stat().st_size != pooled_manifest['bytes'] or file_sha(payload) != pooled_manifest['sha256']):
            raise ValueError('Pooled payload identity/completeness differs')
        with np.load(payload, allow_pickle=False) as archive:
            pooled = {name: archive[name] for name in archive.files}
        verify_pooled_arrays(pooled, len(schema['query_ids']), 768)
        reader = CacheReader(scientific_config['cache_manifest'], scientific_config['cache_identity'], cohort, old_audit)
        galleries, metadata = build_galleries(reader, cohort, budget=1)
        with np.load(prediction/'rankings.npz', allow_pickle=False) as archive:
            rankings = {name: archive[name] for name in archive.files}
        if set(rankings) != {f'{m}_s{s}' for m in METHODS for s in SEEDS}:
            raise ValueError('Incomplete held-out rank matrices')
        for method in METHODS:
            expected = {str(s): [] for s in SEEDS}
            for vector in pooled[method]:
                for seed, rank in search(vector, galleries[method], 100).items():
                    expected[seed].append(rank)
            for seed, rows in expected.items():
                if not np.array_equal(rankings[f'{method}_s{seed}'], np.asarray(rows, dtype=np.int16)):
                    raise ValueError('Saved complete rankings differ from independently regenerated exact gallery search')
        run.save('verification.json', dict(passed=True, source_recipe_complete=True,
            pooled_shape_finite_L2_pass=True, all15_exact_rankings_regenerated=True,
            query_N=len(schema['query_ids']), actual_window_counts=counts, pooled_payload_sha256=file_sha(payload),
            input_status_hashes=statuses, gallery_unique_sources=metadata['unique_source_count'],
            scope='post-execution integrity only; no ground-truth accuracy/DCG/MRR computed', test_task_scores_read=False))
        run.status['output_hashes'] = {'verification.json':file_sha(run.output/'verification.json')}


if __name__ == '__main__':
    main()
