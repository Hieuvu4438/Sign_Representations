"""H004_v0 train/dev cached completion bounds; no GPU or model inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import CacheReader, file_sha, sample_map
from completion_bounds import directional_bounds, source_order, simulate_prefix
from run_context import RunContext
from resources import scratch_preflight


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import numpy as np
        import torch
        from confirmatory import build_galleries, search
        from window_budget import pooled_vector
        from completion_pairwise import pair_lower_bounds, simulate_paired_prefix
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if torch.cuda.is_initialized():
            raise ValueError('CPU-only diagnostic unexpectedly initialized CUDA')
        for name, expected in config['input_hashes'].items():
            if file_sha(REPO / name) != expected:
                raise ValueError(f'Changed registered input: {name}')
        resource = scratch_preflight('/dev/shm', 1 << 20)
        prior = json.loads((REPO / config['prior_quality_config']).read_text())
        report = json.loads((REPO / config['cache_report']).read_text())
        if not report['passed'] or file_sha(report['cache_manifest']) != report['cache_manifest_sha256']:
            raise ValueError('Unavailable or changed native cache')
        cohort = json.loads((REPO / prior['cohort_manifest']).read_text())
        audit = json.loads((REPO / prior['video_audit']).read_text())
        indexed = sample_map(cohort)
        reader = CacheReader(report['cache_manifest'], prior['cache_identity'], cohort, audit)
        galleries, metadata = build_galleries(reader, cohort, budget=1)
        galleries = galleries['full_activity']
        if sorted(galleries) != ['1701', '1702', '1703']:
            raise ValueError('Gallery seeds differ')
        galleries = {int(seed): values for seed, values in galleries.items()}
        state = torch.load(REPO / config['checkpoint'], map_location='cpu', weights_only=True)['model']
        w, gamma, beta = [state[k].detach().numpy().astype(np.float64)
                          for k in ('fc.1.weight', 'fc.0.weight', 'fc.0.bias')]
        if w.shape != (768, 768) or gamma.shape != (768,) or beta.shape != (768,):
            raise ValueError('Native projection shape differs')
        del state
        envelope = {seed: directional_bounds(matrix, w, gamma, beta)
                    for seed, (matrix, _) in galleries.items()}
        mode = config.get('completion_mode', 'component')
        if mode not in ('component', 'paired'):
            raise ValueError('Unknown completion formula')
        paired = {seed: pair_lower_bounds(matrix, w, gamma, beta)
                  for seed, (matrix, _) in galleries.items()} if mode == 'paired' else {}
        ids = sorted(k for k, row in indexed.items() if row['split'] == 'val')
        if len(ids) != 383 or len(cohort['classes']) != 100:
            raise ValueError('Registered cohort coverage differs')
        records = []
        worst_excess = 0.0
        for key in ids:
            x, a, _ = reader.load(key)
            weights = np.clip(a.astype(np.float64), 0., 1.).max(axis=1)
            order = source_order(indexed[key]['source_sha256'], len(x), args.seed)
            dense = pooled_vector(x, a, 'full_activity', list(range(len(x))))
            reference = search(dense, galleries, 100)
            for seed, (matrix, class_rows) in galleries.items():
                lo, hi = envelope[seed]
                scalar = x.astype(np.float64) @ matrix.T
                excess = float(max(0., np.max(lo[None, :] - scalar), np.max(scalar - hi[None, :])))
                worst_excess = max(worst_excess, excess)
                if excess > 0:
                    raise ValueError('Saved features violate ideal architecture envelope')
                if mode == 'paired':
                    # Every complete saved pair direction is a separate verifier.
                    for anchor in range(len(matrix)):
                        if np.any(scalar[:, anchor, None]-scalar < paired[seed][anchor, None, :]):
                            raise ValueError('Saved features violate pairwise envelope')
                    stop = simulate_paired_prefix(x, weights, matrix, class_rows, lo, hi,
                                                  order, paired[seed])
                else:
                    stop = simulate_prefix(x, weights, matrix, class_rows, lo, hi, order)
                if stop['early']:
                    stopped = stop['class_id']
                else:
                    stopped = reference[seed][0]
                if stopped != reference[seed][0]:
                    raise ValueError('Early-stop top1 differs from native dense reference')
                records.append(dict(sample_id=key, gallery_seed=seed,
                                    participant_id=indexed[key]['participant_id'], windows=len(x),
                                    dense_class=reference[seed][0], stopped_class=stopped,
                                    agreement=True, half_budget=stop['used_windows'] <= len(x)//2,
                                    full_activity_weight_sum=float(weights.sum()), **stop))
        if len(records) != 1149:
            raise ValueError('Incomplete pairs')
        summary = {}
        for seed in (1701, 1702, 1703):
            rows = [row for row in records if row['gallery_seed'] == seed]
            used, total = sum(r['used_windows'] for r in rows), sum(r['windows'] for r in rows)
            summary[str(seed)] = dict(pairs=len(rows), early=sum(r['early'] for r in rows),
                                     half_budget=sum(r['half_budget'] for r in rows),
                                     used_windows=used, dense_windows=total,
                                     window_ratio=used/total)
        passed = sum(row['half_budget'] for row in records) >= .5 * len(records)
        run.save('pairs.json', records)
        run.save('completion_feasibility.json', dict(
            run_id=args.run_id, completion_mode=mode, passed_feasibility=bool(passed), pairs=len(records),
            envelope_violations=0, worst_envelope_excess=worst_excess,
            all_top1_reference_agreement=True, by_gallery=summary,
            bound_radius_ranges={str(seed): [float(np.min((hi-lo)/2)), float(np.max((hi-lo)/2))]
                                 for seed, (lo, hi) in envelope.items()},
            source_hashes=config['input_hashes'], cache_manifest_sha256=file_sha(reader.path),
            resource=resource, gallery=metadata, GPU_hours=0, test_access='none',
            no_accuracy_or_DCG_measurement=True, no_new_encoder_calls=True,
            guarantee_scope='ideal-real arithmetic only; floating-point certification unproved',
            timing_scope='cached simulation CPU wall only; no latency/speedup claim'))
        run.status['output_hashes'] = {name: file_sha(run.output/name) for name in
                                       ('pairs.json', 'completion_feasibility.json')}
        if torch.cuda.is_initialized():
            raise ValueError('CUDA initialized during CPU-only diagnostic')


if __name__ == '__main__':
    main()
