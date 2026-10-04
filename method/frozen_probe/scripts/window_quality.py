"""Hash-gated H002 cached quality simulation; no CUDA or test access."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import CacheReader, file_sha
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'output-dir', 'cache-manifest', 'launch-gate'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume:
        raise ValueError('Register a new run; no partial quality resume')
    config = json.loads(args.config.read_text())
    gate = json.loads(args.launch_gate.read_text())
    if (gate['verdict'] != 'VALIDATED' or gate['protocol'] != config['protocol']
            or gate['quality_config_sha256'] != file_sha(args.config)
            or gate['cache_manifest_sha256'] != file_sha(args.cache_manifest)
            or gate['policy_code_sha256'] != file_sha(ROOT / 'window_budget.py')):
        raise ValueError('H002 protocol/cache/policy gate mismatch')
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import numpy as np
        from window_budget import POLICIES, pool_policies, lookup, quality_analysis
        if config['policies'] != list(POLICIES) or config['support_seeds'] != [1701, 1702, 1703] or config['gallery_budget'] != 8:
            raise ValueError('Registered policy/gallery search changed')
        paths = {k: REPO / config[k] for k in ('cohort_manifest', 'video_audit', 'reference_prediction_run')}
        if (file_sha(paths['cohort_manifest']) != config['cache_identity']['cohort_sha256']
                or file_sha(paths['video_audit']) != config['cache_identity']['audit_sha256']):
            raise ValueError('Locked source manifests changed')
        cohort = json.loads(paths['cohort_manifest'].read_text())
        audit = json.loads(paths['video_audit'].read_text())
        reader = CacheReader(args.cache_manifest, config['cache_identity'], cohort, audit)
        ids, matrices, selections, counts = pool_policies(reader, cohort)
        query_ids, results, rankings = lookup(cohort, ids, matrices, config['support_seeds'], 8)
        reference = paths['reference_prediction_run']
        status = json.loads((reference / 'status.json').read_text())
        schema = json.loads((reference / 'prediction_schema.json').read_text())
        if (status['status'] != 'completed' or len(query_ids) != 378 or len(cohort['classes']) != 100
                or len(ids) != 1691 or len(results) != 24 or schema['query_ids'] != query_ids
                or counts['full_activity'] != 61339):
            raise ValueError('Incomplete fixed cohort/reference')
        for name, digest in status['output_hashes'].items():
            if file_sha(reference / name) != digest:
                raise ValueError('Original baseline output changed')
        with np.load(reference / 'rankings.npz', allow_pickle=False) as previous:
            for seed in config['support_seeds']:
                for method, old in (('full_mean', 'mean'), ('full_activity', 'activity')):
                    if not np.array_equal(rankings[f'{method}_s{seed}'], previous[f'{old}_nearest_s{seed}_n8']):
                        raise ValueError('Dense class-lookup reference parity failed')
        report = quality_analysis(query_ids, results, rankings, cohort, counts, config['support_seeds'])
        report.update(run_id=args.run_id, policies_tested=6, test_access='none',
                      cache_manifest_sha256=file_sha(args.cache_manifest), numpy=np.__version__,
                      dense_reference_rankings_exact_parity=True)
        run.save('quality.json', report)
        run.save('metrics.json', results)
        run.save('selections.json', selections)
        run.save('schema.json', dict(query_ids=query_ids, class_ids=list(range(100)),
                 labels=[reader.samples[k]['class_id'] for k in query_ids],
                 participant_ids=[reader.samples[k]['participant_id'] for k in query_ids],
                 support_seeds=config['support_seeds'], gallery_budget=8))
        np.savez_compressed(run.output / 'rankings.npz', **rankings)
        run.status['output_hashes'] = {name: file_sha(run.output / name) for name in (
            'quality.json', 'metrics.json', 'selections.json', 'schema.json', 'rankings.npz')}


if __name__ == '__main__':
    main()
