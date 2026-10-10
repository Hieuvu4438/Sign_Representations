"""Close immutable H007 stores and independently check descriptive exports."""
import csv
from datetime import datetime, timezone
import json
from math import fsum, isclose
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounded_artifacts import inspect_file_plan, verify_store
from cache import file_sha


def read(path):
    return json.loads(Path(path).read_text())


def near(left, right):
    assert isclose(float(left), float(right), rel_tol=0, abs_tol=1e-12), (left, right)


def main():
    records = Path('method/research_lab/records')
    keys = ('REJECTSCORETEST001', 'REJECTANALTEST001', 'REJECTEVAL001',
            'REJECTVERIFY001', 'REJECTVERIFY002', 'REJECTREPORT001')
    runs, roots, bindings = {}, {}, {}
    for key in keys:
        reg = read(records / f'experiments/{key}_registration.json')
        root = roots[key] = Path(reg['output_dir'])
        config, status = read(root / 'config.json'), read(root / 'status.json')
        assert status['run_id'] == reg['run_id']
        assert file_sha(reg['config']) == file_sha(root / 'config.json') == reg['config_sha256'] == status['config_sha256']
        failed = key == 'REJECTVERIFY001'
        assert status['status'] == ('failed' if failed else 'completed')
        assert status['exit_code'] == int(failed)
        assert reg['registered_utc'] < status['start_utc'] < status['end_utc']
        assert status['wall_seconds'] <= reg['CPU_limit_seconds']
        assert status['GPU_hours'] == 0
        for path, digest in config['input_hashes'].items():
            assert path not in bindings or bindings[path] == digest
            bindings[path] = digest
        caps = {**config['artifact_file_caps'], 'artifact_manifest.json': config['artifact_manifest_cap']}
        plan = inspect_file_plan(caps, config['artifact_ceiling_bytes'], ('status.json', 'error.txt'))
        assert plan['passed'] and plan['maximum_logical_bytes'] == reg['maximum_planned_logical_bytes']
        assert config['artifact_ceiling_bytes'] == reg['durable_bytes_ceiling']
        actual = {}
        for path in root.rglob('*'):
            if path.is_file():
                assert not path.is_symlink()
                relative = str(path.relative_to(root))
                assert relative in caps and path.stat().st_size <= caps[relative]
                actual[relative] = path.stat().st_size
        assert sum(actual.values()) <= config['artifact_ceiling_bytes']
        for relative, digest in status['output_hashes'].items():
            assert file_sha(root / relative) == digest
        if failed:
            assert status['error'] == 'AssertionError((0.7785012722015381, 0.7785012623304999))'
            assert status['missing_artifacts'] == ['rejection_verification.json']
            assert not (root / 'artifact_manifest.json').exists()
            store = dict(logical_bytes=sum(actual.values()), complete_manifest=False,
                         failed_partial_store_caps_passed=True, missing_artifacts=status['missing_artifacts'])
        else:
            store = verify_store(root)
            assert store['logical_bytes'] == sum(actual.values())
            assert read(root / 'artifact_manifest.json')['plan']['file_caps'] == caps
        runs[key] = dict(run_id=status['run_id'], pid=status['pid'], status=status['status'],
                         wall_seconds=status['wall_seconds'], GPU_seconds=0, store=store,
                         config_sha256=reg['config_sha256'], status_sha256=file_sha(root / 'status.json'))
    for path, digest in bindings.items():
        assert file_sha(path) == digest, path

    measure = roots['REJECTEVAL001']
    verification = roots['REJECTVERIFY002']
    export = roots['REJECTREPORT001']
    report = read(measure / 'rejection_eval.json')
    meta = read(measure / 'rejection_metadata.json')
    proof = read(verification / 'rejection_verification.json')
    erratum = read(verification / 'dcg_precision_erratum.json')
    summary = read(export / 'report_summary.json')
    assert proof['passed'] and report['primary'] == proof['primary'] == summary['primary']
    assert summary['DCG_precision_version'] == erratum['version'] == 'P007_v1.secondary_DCG_FP64_r1'
    assert (proof['base_repooled_clips'], proof['exact_prediction_and_knownrank_rows'],
            proof['metric_rows_verified'], proof['primary_bootstrap_replicates']) == (805, 12708, 216, 2000)
    assert (summary['figure_curve_count'], summary['operating_point_count']) == (24, 8)
    correction = {(r['part'], r['seed'], r['size'], r['method'], r['policy']): r for r in erratum['rows']}
    with (export / 'per_seed.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    with (export / 'mean.csv').open(newline='') as handle:
        means = list(csv.DictReader(handle))
    assert len(rows) == len(report['metric_rows']) == 216 and len(means) == 48
    for row, measured in zip(rows, report['metric_rows']):
        key = (row['part'], int(row['seed']), int(row['size']), row['score'], row['policy'])
        assert key == (measured['part'], measured['seed'], measured['size'], measured['method'], measured['policy'])
        for field in ('n_known', 'n_unknown', 'threshold', 'FAR', 'CCR', 'known_acceptance'):
            near(row[field], measured['point'][field])
        for exported, original in (('closed_known_Top1', 'Top1'), ('known_Top5', 'Top5'), ('known_MRR', 'MRR')):
            near(row[exported], measured['known_retrieval'][original])
        near(row['known_DCG_FP64_corrected'], correction[key]['corrected_FP64_DCG'])
        near(row['known_DCG_FP32_original'], correction[key]['reported_FP32_DCG'])
        near(row['adapted_joint_AUC'], measured['adapted_grouped_trapezoid_AUC'])
        for field in ('common_known_CCR', 'common_known_acceptance', 'persistent_unknown_FAR'):
            near(row[field], measured[field])
    mean_keys = set()
    for mean in means:
        key = (mean['part'], mean['size'], mean['score'], mean['policy'])
        assert key not in mean_keys and mean['aggregation'] == 'equal_fixed_three_seed_mean'
        mean_keys.add(key)
        chosen = [r for r in rows if (r['part'], r['size'], r['score'], r['policy']) == key]
        assert len(chosen) == 3 and sorted(int(r['seed']) for r in chosen) == [1701, 1702, 1703]
        for field in list(rows[0])[5:]:
            near(mean[field], fsum(float(r[field]) for r in chosen) / 3)

    # Check the posthoc buckets, not a human interpretation of the videos.
    scores = np.load(measure / 'scores.npy', allow_pickle=False)
    pred = np.load(measure / 'predictions.npy', allow_pickle=False)
    ranks = np.load(measure / 'known_rank.npy', allow_pickle=False)
    observed_cases = read(export / 'numerical_cases.json')
    assert observed_cases['posthoc'] and observed_cases['seed'] == 1701
    rng = np.random.default_rng(1701)
    case_index = 0
    bucket_order = ('known_correct_accepted', 'known_correct_rejected', 'known_wrong_accepted',
                    'unknown_accepted', 'unknown_rejected')
    for part in sorted((p for p in meta['parts'] if p['name'] in ('base_dev', 'MS_dev')), key=lambda p: p['name']):
        for si, seed in enumerate(meta['support_seeds']):
            buckets = {name: [] for name in bucket_order}
            threshold = report['calibration'][f'{seed}:50:maxcosine']['threshold']
            for i in range(part['start'], part['stop']):
                known, correct = ranks[si, 2, i] > 0, pred[si, 2, i] == meta['labels'][i]
                accepted = scores[si, 2, 0, i] >= threshold
                if known and correct:
                    bucket = 'known_correct_accepted' if accepted else 'known_correct_rejected'
                elif known and accepted:
                    bucket = 'known_wrong_accepted'
                elif not known:
                    bucket = 'unknown_accepted' if accepted else 'unknown_rejected'
                else:
                    continue  # known-wrong-rejected was explicitly omitted.
                buckets[bucket].append(i)
            for bucket, pool in buckets.items():
                indices = sorted(rng.choice(pool, size=min(2, len(pool)), replace=False).tolist()) if pool else []
                case = observed_cases['cases'][case_index]
                case_index += 1
                assert (case['part'], case['seed'], case['bucket'], case['eligible_count']) == (part['name'], seed, bucket, len(pool))
                assert len(case['sampled']) == len(indices)
                for sample, i in zip(case['sampled'], indices):
                    assert (sample['sample_id'], sample['class_id'], sample['official_signer'], sample['predicted_class'], sample['known_rank']) == (
                        meta['query_ids'][i], meta['labels'][i], meta['signers'][i], int(pred[si, 2, i]), int(ranks[si, 2, i]))
                    for mi, method in enumerate(read(export / 'config.json')['score_names']):
                        near(sample['scores'][method], scores[si, 2, mi, i])
                        assert sample['accepted'][method] == bool(scores[si, 2, mi, i] >= report['calibration'][f'{seed}:50:{method}']['threshold'])
    assert case_index == len(observed_cases['cases']) == 30
    sys.path.insert(0, 'method/stream_reliability')
    from gpu_accounting import inspect_effective_budget
    budget = inspect_effective_budget(read('method/frozen_probe/configs/MSFULLFEAT001.json'))
    statuses = [read(path) for direction in ('frozen_probe', 'research_lab', 'stream_reliability')
                for path in Path('method', direction, 'runtime').glob('*/status.json')]
    assert all(s['status'] in ('completed', 'failed', 'interrupted') for s in statuses)
    captured = {p.name: dict(bytes=p.stat().st_size, sha256=file_sha(p))
                for p in Path('method/frozen_probe/references/nac_971f6767').iterdir() if p.is_file()}
    assert sum(e['bytes'] for e in captured.values()) == 5648
    assert captured['utils.py']['sha256'] == '57ec4b2077012b548a18a6084430f6f867032fc42098d04799938372d40d288e'
    assert captured['LICENSE']['sha256'] == '97a39b95ccf0e0a78daf426c240233eff04ca3b01bc4a9f8fa438019c58a5d0c'
    output = dict(verified_utc=datetime.now(timezone.utc).isoformat(), passed=True, runs=runs,
                  current_input_bindings_checked=len(bindings), effective_budget=budget,
                  status_count=len(statuses), live_owned_statuses=0,
                  total_captured_wall_seconds=sum(r['wall_seconds'] for r in runs.values()),
                  total_charged_GPU_seconds=0,
                  total_runtime_logical_bytes=sum(r['store']['logical_bytes'] for r in runs.values()),
                  public_source_capture=captured, capture_total_bytes=5648,
                  report_per_seed_rows=216, report_mean_rows=48, posthoc_buckets_checked=30,
                  primary=report['primary'], secondary_DCG_precision_version=erratum['version'],
                  source_sha256=file_sha(__file__),
                  scope='Same-agent current binding/store/export closure; five complete stores and failed partial verifier retained. No raw-video replay, external review, new quality run or novel method.')
    with (records / 'inventory/H007_QUALITY_CYCLE_BINDING_VERIFICATION_r1.json').open('x') as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps({k: v for k, v in output.items() if k not in ('runs', 'effective_budget', 'public_source_capture')}, indent=2))


if __name__ == '__main__':
    main()
