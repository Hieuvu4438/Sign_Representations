"""Read-only cycle closure; bounded exclusive provenance inventory output."""
import csv
from datetime import datetime, timezone
import json
from math import fsum, isclose
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounded_artifacts import inspect_file_plan, verify_store
from cache import file_sha


def read(path): return json.loads(Path(path).read_text())


def near(a, b): assert isclose(float(a), float(b), abs_tol=1e-12, rel_tol=0), (a, b)


def main():
    started = time.monotonic(); records = Path('method/research_lab/records')
    keys = ['HEADTEST001', 'HEADPOOL001', 'HEADPAIRTEST001', 'HEADPAIRPOOL001', 'HEADPAIRCOST001',
            'HEADPILOT001', 'HEADPILOT002', 'HEADPILOT003', 'HEADANALYSIS001', 'HEADREPORT001']
    roots = {}; runs = {}; bindings = {}
    for key in keys:
        registration = read(records / f'experiments/{key}_registration.json')
        root = roots[key] = Path(registration['output_dir']); config = read(root / 'config.json'); status = read(root / 'status.json')
        failed = key == 'HEADPOOL001'
        assert status['status'] == ('failed' if failed else 'completed') and status['exit_code'] == int(failed)
        assert status['run_id'] == registration['run_id']
        assert file_sha(registration['config']) == file_sha(root / 'config.json') == registration['config_sha256'] == status['config_sha256']
        assert registration['registered_utc'] < status['start_utc'] < status['end_utc']
        assert status['wall_seconds'] <= registration['CPU_limit_seconds']
        gpu = status['GPU_hours'] * 3600
        assert gpu <= registration.get('GPU_seconds', 0)
        if key not in ('HEADPAIRCOST001', 'HEADPILOT001', 'HEADPILOT002', 'HEADPILOT003'): assert gpu == 0
        for path, digest in config['input_hashes'].items():
            assert path not in bindings or bindings[path] == digest, path
            bindings[path] = digest
        caps = {**config['artifact_file_caps'], 'artifact_manifest.json': config['artifact_manifest_cap']}
        plan = inspect_file_plan(caps, config['artifact_ceiling_bytes'], ('status.json', 'error.txt'))
        assert plan['passed'] and plan['maximum_logical_bytes'] == registration['maximum_planned_logical_bytes']
        assert config['artifact_ceiling_bytes'] == registration['durable_bytes_ceiling']
        sizes = {}
        for path in root.rglob('*'):
            if path.is_file():
                relative = str(path.relative_to(root)); assert not path.is_symlink()
                assert relative in caps and path.stat().st_size <= caps[relative]
                sizes[relative] = path.stat().st_size
        actual = sum(sizes.values()); assert actual <= config['artifact_ceiling_bytes']
        for name, digest in status['output_hashes'].items(): assert file_sha(root / name) == digest
        if failed:
            assert status['error'] == "ValueError('Signer has fewer than two query classes: P38')"
            assert sorted(status['missing_artifacts']) == ['fit_vectors.npy', 'pool_summary.json', 'source_roster.json', 'training_plan.npy']
            assert not (root / 'artifact_manifest.json').exists()
            store = dict(logical_bytes=actual, complete_manifest=False, failed_partial_caps_passed=True)
        else:
            store = verify_store(root); assert store['logical_bytes'] == actual
            assert read(root / 'artifact_manifest.json')['plan']['file_caps'] == caps
            for name, spec in config.get('array_specs', {}).items():
                x = np.load(root / name, allow_pickle=False)
                assert x.shape == tuple(spec['shape']) and x.dtype.str == spec['dtype'] and np.isfinite(x).all()
        runs[key] = dict(run_id=status['run_id'], status=status['status'], pid=status['pid'],
            start_utc=status['start_utc'], end_utc=status['end_utc'], wall_seconds=status['wall_seconds'],
            GPU_seconds=gpu, config_sha256=registration['config_sha256'], status_sha256=file_sha(root / 'status.json'), store=store)
    for path, digest in bindings.items(): assert file_sha(path) == digest, path
    analysis = roots['HEADANALYSIS001']; proof = read(analysis / 'head_analysis.json')
    assert proof['passed'] and proof['verification_passed'] and not proof['scientific_screen_passed']
    assert (proof['head_metric_rows_checked'], proof['existing_metric_rows_checked'], len(proof['curve_segments']), proof['curve_points_used']) == (324, 216, 162, 76409)
    rows = list(csv.DictReader((analysis / 'per_seed.csv').open())); means = list(csv.DictReader((analysis / 'mean.csv').open()))
    assert len(rows) == 540 and len(means) == 108
    for mean in means:
        cells = [r for r in rows if tuple(r[k] for k in ('part', 'size', 'method', 'policy')) == tuple(mean[k] for k in ('part', 'size', 'method', 'policy'))]
        expected = 9 if mean['method'] in ('residual_linear', 'FRDisc_frozen_MLP') else 3
        assert len(cells) == int(mean['cells']) == expected
        for metric in ('FAR', 'CCR', 'known_acceptance', 'Top1', 'Top5', 'DCG_FP64', 'MRR', 'adapted_AUC'):
            near(mean[metric], fsum(float(r[metric]) for r in cells) / len(cells))
    for name, metric in [('MS_dev', 'CCR'), ('MS_dev', 'FAR'), ('base_dev', 'CCR')]:
        selected = [m for m in means if m['part'] == name and m['size'] == '50' and m['policy'] == 'FROZEN50']
        mlp = next(m for m in selected if m['method'] == 'FRDisc_frozen_MLP')
        raw = next(m for m in selected if m['method'] == 'maxcosine')
        near(float(mlp[metric]) - float(raw[metric]), proof['primary'][name + '_' + metric]['mean_delta'])
    deltas = np.load(analysis / 'bootstrap_head_deltas.npy', allow_pickle=False)
    valid = np.load(analysis / 'bootstrap_head_valid.npy', allow_pickle=False)
    assert valid.all() and deltas.shape == valid.shape == (2000, 3)
    for i, name in enumerate(proof['primary_order']):
        assert proof['primary'][name]['interval95'] == np.quantile(deltas[:, i], [.025, .975]).tolist()
        assert proof['primary'][name]['invalid_draws'] == 0
    assert [k for k, v in proof['gates'].items() if not v] == ['MS_FAR_conditional_upper_at_most_2pp', 'MS_FAR_mean_increase_at_most_2pp']
    report = read(roots['HEADREPORT001'] / 'report_summary.json')
    assert report['passed'] and report['primary'] == proof['primary']
    assert (report['figure_curves'], report['mean_operating_points'], report['numerical_case_buckets']) == (60, 12, 10)
    # Reproduce the exact fixed-order numerical selection and bucket counts.
    meta = read(roots['REJECTEVAL001'] / 'rejection_metadata.json') if 'REJECTEVAL001' in roots else read(Path(read(analysis / 'config.json')['measured_runtime']) / 'rejection_metadata.json')
    measured = Path(read(analysis / 'config.json')['measured_runtime'])
    frozen = np.load(measured / 'scores.npy', allow_pickle=False); ranks = np.load(measured / 'known_rank.npy', allow_pickle=False)
    pred = np.load(measured / 'predictions.npy', allow_pickle=False); old = read(measured / 'rejection_eval.json')
    logits = np.load(roots['HEADPILOT001'] / 'head_scores.npy', allow_pickle=False)
    hd = read(roots['HEADPILOT001'] / 'head_evaluation.json'); cases = read(roots['HEADREPORT001'] / 'numerical_disagreements.json')['cases']
    rng = np.random.default_rng(1701); cursor = 0
    for name in ('base_dev', 'MS_dev'):
        part = next(p for p in meta['parts'] if p['name'] == name); sl = slice(part['start'], part['stop'])
        known = ranks[0, 2, sl] > 0; correct = pred[0, 2, sl] == meta['labels'][sl]
        rt = old['calibration']['1701:50:maxcosine']['threshold']; ht = hd['calibration']['1701:50:FRDisc_frozen_MLP']['threshold']
        ra = frozen[0, 2, 0, sl] >= rt; ha = logits[0, 2, 1, sl] >= ht
        masks = [known & correct & ~ra & ha, known & correct & ra & ~ha, ~known & ~ra & ha,
                 ~known & ra & ~ha, known & ~correct & ha]
        for mask in masks:
            case = cases[cursor]; cursor += 1; eligible = np.flatnonzero(mask) + part['start']
            chosen = sorted(rng.choice(eligible, min(2, len(eligible)), replace=False).tolist())
            assert case['part'] == name and case['eligible_count'] == len(eligible)
            assert [r['sample_id'] for r in case['samples']] == [meta['query_ids'][i] for i in chosen]
            for sample, i in zip(case['samples'], chosen):
                assert sample['predicted_class'] == pred[0, 2, i] and sample['known_rank'] == ranks[0, 2, i]
                assert sample['true_class'] == meta['labels'][i] and sample['official_signer'] == meta['signers'][i]
                assert sample['raw_score'] == frozen[0, 2, 0, i] and sample['MLP_logit'] == logits[0, 2, 1, i]
                assert sample['raw_accepted'] == bool(frozen[0, 2, 0, i] >= rt) and sample['MLP_accepted'] == bool(logits[0, 2, 1, i] >= ht)
    capture = read(records / 'inventory/FRDISC_MODEL_SOURCE_CAPTURE_r1.json')
    for path, entry in capture['files'].items():
        assert Path(path).stat().st_size == entry['bytes'] and file_sha(path) == entry['sha256']
    assert sum(e['bytes'] for e in capture['files'].values()) == capture['source_bytes']
    sys.path.insert(0, 'method/stream_reliability'); from gpu_accounting import inspect_effective_budget
    policy = read(roots['HEADPILOT001'] / 'config.json'); budget = inspect_effective_budget(policy)
    census = []
    for root in map(Path, policy['runtime_roots']):
        for path in root.glob('*/status.json'):
            s = read(path); census.append(dict(path=str(path), status=s['status']))
    assert all(r['status'] not in ('running', 'queued', 'pending') for r in census)
    result = dict(passed=True, verified_utc=datetime.now(timezone.utc).isoformat(), source_sha256=file_sha(__file__),
        run_count=len(keys), current_bound_inputs_verified=len(bindings), runs=runs,
        total_runtime_logical_bytes=sum(v['store']['logical_bytes'] for v in runs.values()),
        summed_captured_wall_seconds=fsum(v['wall_seconds'] for v in runs.values()),
        GPU_seconds=fsum(v['GPU_seconds'] for v in runs.values()), primary=proof['primary'], gates=proof['gates'],
        scientific_screen_passed=False, per_seed_rows=540, mean_rows=108, curve_segments=162,
        curve_points=76409, figure_curves=60, figure_operating_points=12, numerical_case_buckets=10,
        bootstrap_draws=2000, invalid_bootstrap_draws=0, terminal_status_count=len(census),
        live_owned_jobs=0, program_used_seconds=budget['project_used_seconds'], H007_used_seconds=budget['hypothesis_used_seconds'],
        program_limit_seconds=budget['project_limit_seconds'], H007_limit_seconds=budget['hypothesis_limit_seconds'],
        public_capture_inventory_sha256=file_sha(records / 'inventory/FRDISC_MODEL_SOURCE_CAPTURE_r1.json'),
        captured_public_source_bytes=capture['source_bytes'], closure_wall_seconds=time.monotonic() - started,
        scope='Same-agent current-file/cap/hash and export recount; numerical audit separate from failed scientific screen. No new fitting, CPU/GPU full pipeline, raw-video replay, scientific external review or novelty claim.')
    payload = (json.dumps(result, indent=2) + '\n').encode(); assert len(payload) <= 262144
    target = records / 'inventory/H007_HEAD_CYCLE_BINDING_VERIFICATION_r1.json'
    with target.open('xb') as f: f.write(payload)
    print(json.dumps({k: v for k, v in result.items() if k not in ('runs', 'primary')}, indent=2))


if __name__ == '__main__': main()
