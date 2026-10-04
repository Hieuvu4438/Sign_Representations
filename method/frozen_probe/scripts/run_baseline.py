"""Registered train/dev baseline CLI; refuses a missing launch gate."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from cache import CacheReader, file_sha, object_sha
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--cache-manifest', required=True, type=Path)
    parser.add_argument('--launch-gate', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume:
        raise ValueError('Terminal prediction runs are immutable; register a new run instead')
    config = json.loads(args.config.read_text())
    gate = json.loads(args.launch_gate.read_text())
    if (gate.get('verdict') != 'VALIDATED' or gate.get('protocol') != config['protocol']
            or gate.get('baseline_config_sha256') != file_sha(args.config)
            or gate.get('cache_manifest_sha256') != file_sha(args.cache_manifest)):
        raise ValueError('Protocol/cache launch gate is missing or mismatched')
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import numpy as np
        from baseline import pool_cache, run_controls
        cohort_path = REPO / config['cohort_manifest']
        audit_path = REPO / config['video_audit']
        if file_sha(cohort_path) != config['cohort_sha256'] or file_sha(audit_path) != config['video_audit_sha256']:
            raise ValueError('Locked manifests changed')
        cohort = json.loads(cohort_path.read_text())
        audit = json.loads(audit_path.read_text())
        if (len(cohort['classes']) != 100 or len(cohort['samples']) != 1691
                or config['support_seeds'] != [1701, 1702, 1703] or config['label_budgets'] != [1, 2, 4, 8]
                or config['alpha'] != 1.0 or config['projection_seed'] != 1337
                or config['seed'] != 1337 or config['cache_identity']['contract']['output_dim'] != 768):
            raise ValueError('Registered candidate configuration differs')
        identity = config['cache_identity']
        if identity['cohort_sha256'] != file_sha(cohort_path) or identity['audit_sha256'] != file_sha(audit_path):
            raise ValueError('Cache identity is not bound to locked manifests')
        reader = CacheReader(args.cache_manifest, identity, cohort, audit)
        ids, matrices, projection, metadata = pool_cache(reader, cohort, config['projection_seed'])
        query_ids, results, rankings = run_controls(cohort, ids, matrices, config['support_seeds'],
                                                    config['label_budgets'], config['alpha'])
        if len(results) != 108 or len(query_ids) != 378:
            raise ValueError('Incomplete prediction/control coverage')
        np.savez_compressed(run.output / 'rankings.npz', **rankings)
        run.save('prediction_schema.json', dict(query_ids=query_ids, class_ids=list(range(100)),
                 participant_ids=[reader.samples[key]['participant_id'] for key in query_ids],
                 labels=[reader.samples[key]['class_id'] for key in query_ids],
                 dtype='int16', shape=[378, 100], ranking_keys=list(rankings),
                 ties='descending_score_then_ascending_class_id'))
        run.save('pooling_metadata.json', metadata)
        run.save('results.json', dict(run_id=args.run_id, scope=f"adapted {config['protocol']} train/dev failure pilot",
                 results=results, numpy=np.__version__, projection_sha256=__import__('hashlib').sha256(
                     projection.astype('<f8').tobytes()).hexdigest(),
                 pooled_matrix_sha256={name: __import__('hashlib').sha256(x.astype('<f8').tobytes()).hexdigest()
                                       for name, x in matrices.items()}, pooled_sample_ids=ids,
                 cache_identity_sha256=object_sha(identity), cache_manifest_sha256=file_sha(args.cache_manifest),
                 launch_gate_sha256=file_sha(args.launch_gate), test_access='none', HPO_trials=0,
                 uncertainty='not yet analyzed; raw complete rankings retained'))
        # Preserve the small cache manifest even if tmpfs payload is later rebuilt.
        run.save('cache_manifest.json', reader.manifest)
        run.status['output_hashes'] = {name: file_sha(run.output / name) for name in (
            'rankings.npz', 'prediction_schema.json', 'pooling_metadata.json', 'results.json', 'cache_manifest.json')}


if __name__ == '__main__': main()
