"""Close terminal metadata stores and current bindings without loading features."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounded_artifacts import verify_store
from cache import file_sha


def read(path):
    return json.loads(Path(path).read_text())


def main():
    records = Path('method/research_lab/records')
    runs, bindings, roots = {}, {}, {}
    for key in ('OSMETATEST001', 'OSMETA001', 'OSMETAVERIFY001'):
        reg = read(records / f'experiments/{key}_registration.json')
        root = Path(reg['output_dir'])
        roots[key] = root
        config, status = read(root / 'config.json'), read(root / 'status.json')
        assert status['status'] == 'completed' and status['exit_code'] == 0
        assert status['run_id'] == reg['run_id']
        assert file_sha(reg['config']) == file_sha(root / 'config.json') == reg['config_sha256'] == status['config_sha256']
        for path, digest in config['input_hashes'].items():
            assert path not in bindings or bindings[path] == digest
            bindings[path] = digest
        store = verify_store(root)
        manifest = read(root / 'artifact_manifest.json')
        assert manifest['plan']['file_caps'] == {**config['artifact_file_caps'], 'artifact_manifest.json': config['artifact_manifest_cap']}
        assert manifest['plan']['ceiling_bytes'] == reg['durable_bytes_ceiling'] == config['artifact_ceiling_bytes']
        for path, digest in status['output_hashes'].items():
            assert file_sha(root / path) == digest
        runs[key] = dict(run_id=status['run_id'], pid=status['pid'], status=status['status'],
                         wall_seconds=status['wall_seconds'], GPU_seconds=status['GPU_hours'] * 3600,
                         store=store, config_sha256=reg['config_sha256'],
                         status_sha256=file_sha(root / 'status.json'),
                         manifest_sha256=file_sha(root / 'artifact_manifest.json'))
    for path, digest in bindings.items():
        assert file_sha(path) == digest, path
    tests = read(roots['OSMETATEST001'] / 'metadata_tests.json')
    proof = read(roots['OSMETAVERIFY001'] / 'metadata_verification.json')
    census = read(roots['OSMETA001'] / 'metadata_census.json')
    assert tests['passed'] and proof['passed'] and proof['membership_cells_checked'] == 27
    assert sum(proof['source_rows'].values()) == 3389
    base, target = census['roles']['ASLCitizen_base'], census['roles']['ASLCitizen_target']
    assert base['calibration']['clips'] == 286 and base['calibration']['signers'] == 8
    assert base['gallery_pool']['clips'] == 1027 and base['gallery_pool']['signers'] == 24
    assert base['calibration']['missing_classes'] == base['gallery_pool']['missing_classes'] == []
    assert target['gallery_pool']['missing_classes'] == [83]
    assert len(target['calibration']['missing_classes']) == 53
    assert all(census[k] == 0 for k in ('feature_arrays_loaded', 'quality_scores_computed', 'thresholds_fitted', 'new_encoder_calls'))
    assert census['test_access'] == 'none'
    sys.path.insert(0, 'method/stream_reliability')
    from gpu_accounting import inspect_effective_budget
    budget = inspect_effective_budget(read('method/frozen_probe/configs/MSFULLFEAT001.json'))
    output = dict(verified_utc=datetime.now(timezone.utc).isoformat(), passed=True, runs=runs,
                  current_input_bindings_checked=len(bindings), effective_budget=budget,
                  total_captured_wall_seconds=sum(x['wall_seconds'] for x in runs.values()),
                  total_charged_GPU_seconds=sum(x['GPU_seconds'] for x in runs.values()),
                  total_logical_bytes=sum(x['store']['logical_bytes'] for x in runs.values()),
                  source_rows_verified=proof['source_rows'], membership_cells_checked=27,
                  base_calibration_feasible=True, target_fixed_split_feasible=False,
                  source_sha256=file_sha(__file__),
                  scope='Same-agent store/binding closure plus prior independent metadata recount; no numeric feature reads, quality scores or external review.')
    with (records / 'inventory/H007_METADATA_CYCLE_VERIFICATION_r1.json').open('x') as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps({k: v for k, v in output.items() if k not in ('runs', 'effective_budget')}, indent=2))


if __name__ == '__main__':
    main()
