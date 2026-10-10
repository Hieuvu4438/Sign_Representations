"""Verify completed and failed synthetic stores/current bindings without reruns."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounded_artifacts import inspect_file_plan, verify_store
from cache import file_sha


def read(path):
    return json.loads(Path(path).read_text())


def main():
    records = Path('method/research_lab/records')
    runs, roots, bindings = {}, {}, {}
    for key in ('OSCRREFTEST001', 'OSCRREFTEST002', 'OSCRADAPTTEST001'):
        reg = read(records / f'experiments/{key}_registration.json')
        root = Path(reg['output_dir'])
        roots[key] = root
        config, status = read(root / 'config.json'), read(root / 'status.json')
        assert status['run_id'] == reg['run_id']
        assert file_sha(reg['config']) == file_sha(root / 'config.json') == reg['config_sha256'] == status['config_sha256']
        assert status['status'] == ('failed' if key == 'OSCRREFTEST001' else 'completed')
        assert status['exit_code'] == (1 if key == 'OSCRREFTEST001' else 0)
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
        if key == 'OSCRREFTEST001':
            assert status['error'] == "KeyError('__import__')"
            assert not (root / 'reference_tests.json').exists()
            assert not (root / 'artifact_manifest.json').exists()
            store = dict(logical_bytes=sum(actual.values()), complete_manifest=False,
                         failed_partial_store_caps_passed=True,
                         missing_artifacts=status['missing_artifacts'])
        else:
            store = verify_store(root)
            assert store['logical_bytes'] == sum(actual.values())
            manifest = read(root / 'artifact_manifest.json')
            assert manifest['plan']['file_caps'] == caps
        runs[key] = dict(run_id=status['run_id'], pid=status['pid'], status=status['status'],
                         wall_seconds=status['wall_seconds'], GPU_seconds=status['GPU_hours'] * 3600,
                         store=store, config_sha256=reg['config_sha256'],
                         status_sha256=file_sha(root / 'status.json'))
    for path, digest in bindings.items():
        assert file_sha(path) == digest, path
    reference = read(roots['OSCRREFTEST002'] / 'reference_tests.json')
    assert reference['fixture_audit_passed'] and not reference['reference_validated_for_H007']
    assert [x['observed_reference_area'] for x in reference['fixtures']] == [.25, 1., .25]
    assert [x['manual_joint_area'] for x in reference['fixtures']] == [0., 1., 0.]
    adapted = read(roots['OSCRADAPTTEST001'] / 'adapted_tests.json')
    assert adapted['passed'] and len(adapted['groups']) == 6 and len(adapted['rational_cases']) == 30
    assert [x['observed']['grouped_trapezoid_AUC'] for x in adapted['manual_fixtures']] == [0., 1., 0., .5, .5]
    for fixture in adapted['manual_fixtures']:
        last = fixture['observed']['points'][-1]
        assert last['FAR'] == 1 and last['CCR'] == last['closed_known_accuracy']
    sys.path.insert(0, 'method/stream_reliability')
    from gpu_accounting import inspect_effective_budget
    budget = inspect_effective_budget(read('method/frozen_probe/configs/MSFULLFEAT001.json'))
    statuses = [read(path) for direction in ('frozen_probe', 'research_lab', 'stream_reliability')
                for path in Path('method', direction, 'runtime').glob('*/status.json')]
    assert all(s['status'] in ('completed', 'failed', 'interrupted') for s in statuses)
    source_capture = Path('method/frozen_probe/references/fsosar_c8115c09')
    captured = {p.name: dict(bytes=p.stat().st_size, sha256=file_sha(p)) for p in source_capture.iterdir() if p.is_file()}
    assert sum(e['bytes'] for e in captured.values()) == 35043
    assert captured['utils.py']['sha256'] == '55b79e5e9ca57956b499427dec97b187b464faad7fb4d382e0c3263542738b6b'
    assert captured['LICENSE']['sha256'] == 'd3522fc7814fc79c74f64c68dc9cb591b062b5e90a5bddd50e444e5b3f727711'
    output = dict(verified_utc=datetime.now(timezone.utc).isoformat(), passed=True, runs=runs,
                  current_input_bindings_checked=len(bindings), effective_budget=budget,
                  status_count=len(statuses), live_owned_statuses=0,
                  total_captured_wall_seconds=sum(x['wall_seconds'] for x in runs.values()),
                  total_charged_GPU_seconds=sum(x['GPU_seconds'] for x in runs.values()),
                  total_runtime_logical_bytes=sum(x['store']['logical_bytes'] for x in runs.values()),
                  public_source_capture=captured, capture_total_bytes=35043,
                  pinned_reference_rejected=True, adapted_specification_fixture_groups=6,
                  independent_rational_cases=30, source_sha256=file_sha(__file__),
                  scope='Same-agent current store/binding/resource closure; failed partial store retained, two completed manifests verified; no scientific quality or rerun.')
    with (records / 'inventory/H007_EVALUATOR_CYCLE_VERIFICATION_r1.json').open('x') as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps({k: v for k, v in output.items() if k not in ('runs', 'effective_budget', 'public_source_capture')}, indent=2))


if __name__ == '__main__':
    main()
