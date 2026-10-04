"""Bounded train/dev duplicate screen and conditional cost projection; no new test."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from budget import inspect_budget
from cache import file_sha
from run_context import RunContext
from scripts.screen_duplicates import compare_pair
from window_budget import selected_indices


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        loaded = {}
        for name, item in config['inputs'].items():
            path = REPO / item['path']
            if file_sha(path) != item['sha256']:
                raise ValueError('Changed feasibility input')
            loaded[name] = json.loads(path.read_text())
        cohort, audit, old, rates, original = [loaded[k] for k in
                                               ('cohort', 'audit', 'original_audit', 'rates', 'original_analysis')]
        if not audit['passed'] or len(audit['samples']) != len(cohort['samples']) or audit['cohort_sha256'] != config['inputs']['cohort']['sha256']:
            raise ValueError('Incomplete new native source audit')
        indexed = {s['sample_id']: s for s in cohort['samples']}
        if len(indexed) != len(cohort['samples']) or set(indexed) != {s['sample_id'] for s in audit['samples']}:
            raise ValueError('Audit source identity mismatch')
        for source in audit['samples']:
            if any(source[k] != indexed[source['sample_id']][k] for k in ('split', 'participant_id', 'class_id', 'source_sha256')):
                raise ValueError('Audit split/label/source mismatch')
        samples = sorted(audit['samples'], key=lambda s: s['sample_id'])
        tested = dict(cross_split=0, within_split=0)
        eligible = dict(cross_split=0, within_split=0)
        candidates = []
        for i, a in enumerate(samples):
            for b in samples[i+1:]:
                kind = 'cross_split' if a['split'] != b['split'] else 'within_split'
                tested[kind] += 1
                result = compare_pair(a, b)
                if result is None:
                    continue
                eligible[kind] += 1
                if result.pop('candidate'):
                    candidates.append(dict(pair_type=kind,
                        a={k: a[k] for k in ('sample_id', 'split', 'participant_id', 'class_id')},
                        b={k: b[k] for k in ('sample_id', 'split', 'participant_id', 'class_id')}, **result))
        screening = dict(pairs_tested=tested, eligible_pairs=eligible, candidates=candidates,
            cross_split_candidates=sum(c['pair_type'] == 'cross_split' for c in candidates),
            automatic_exclusions=0, thresholds='original compare_pair: frame_ratio>=.8, >=3 informative positions(std>=5), mean Hamming<=6/max<=12',
            scope='bounded coarse fingerprint screen, not exhaustive uniqueness/session/pretraining disjointness')
        run.save('screen.json', screening)
        old_dev = [s for s in old['samples'] if s['split'] == 'val']
        new_dev = [s for s in samples if s['split'] == 'val']
        train = [s for s in samples if s['split'] == 'train']

        def totals(sources):
            return dict(N=len(sources), frames=sum(s['frames'] for s in sources),
                        dense=sum(s['windows'] for s in sources),
                        sparse=sum(len(selected_indices('uniform8', s['windows'], s['source_sha256'])) for s in sources))

        old_counts, new_counts, train_counts = map(totals, (old_dev, new_dev, train))
        old_test_sources = loaded['original_test_audit']['samples']
        if len(old_test_sources) != original['N_scored'] or any(s['split'] != 'test' for s in old_test_sources):
            raise ValueError('Incomplete original measured-test frame denominator')
        old_test = dict(N=original['N_scored'], frames=sum(s['frames'] for s in old_test_sources),
                        dense=original['costs']['full_activity']['encoded_windows'], sparse=original['costs']['uniform8']['encoded_windows'])
        projected_test = {key: float(new_counts[key] * old_test[key] / old_counts[key]) for key in old_counts}

        def cost(windows, frames, pools):
            components = windows*rates['encoder_seconds_per_window'] + frames*rates['decode_preprocess_seconds_per_frame'] + pools*rates['pool_search_seconds_per_query_recipe']
            return components*rates['safety_factor'] + rates['setup_seconds']

        gallery = cost(train_counts['dense'], train_counts['frames'], train_counts['N']*4)
        query = cost(projected_test['dense']+3*projected_test['sparse'], projected_test['frames']*4, projected_test['N']*4)
        full_cache = cost(sum(s['windows'] for s in samples), sum(s['frames'] for s in samples), 0)
        budget = inspect_budget([REPO / r for r in config['budget_runtime_roots']], 'H002', 14400, 7200)
        result = dict(train_only_gallery=train_counts, old_dev=old_counts, new_dev=new_counts,
            original_measured_test=old_test, projected_new_test=projected_test,
            projection_rule='new-dev aggregate × original-measured-test/original-dev aggregate per field; fractional planning quantities, no new test access',
            gallery_estimated_GPU_seconds=gallery, query_estimated_GPU_seconds=query,
            total_projected_GPU_seconds=gallery+query, budget=budget,
            projected_fits_remaining=bool(gallery+query < budget['remaining_seconds']),
            dev_first_full_native_cache_estimated_GPU_seconds=full_cache,
            dev_first_total_projected_GPU_seconds=full_cache+query,
            dev_first_projected_fits_remaining=bool(full_cache+query < budget['remaining_seconds']),
            dev_first_full_native_payload_bytes=sum(s['windows'] for s in samples)*(768*4+2*4+8),
            train_only_feature_payload_bytes=train_counts['dense']*(768*4+2*4+8),
            cross_split_candidates=screening['cross_split_candidates'],
            verdict='FEASIBILITY_ONLY_NOT_A_LAUNCH_GATE',
            limitations='Projection from known original outcomes/resources; does not inspect fresh test counts or guarantee timing. Same corpus/checkpoint/participant populations. Train-only gallery cache contract/runner, software parity and P004 scientific freeze remain required before fresh selected test pixels/scores; actual post-freeze counts/cost/source gate required before GPU.')
        run.save('feasibility.json', result)
        run.status['output_hashes'] = {name: file_sha(run.output / name) for name in ('screen.json', 'feasibility.json')}


if __name__ == '__main__':
    main()
