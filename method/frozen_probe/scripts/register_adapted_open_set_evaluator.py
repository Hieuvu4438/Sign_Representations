"""Freeze the explicitly adapted evaluator fixtures and all source dependencies."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from register_axis_stage import dependencies, put, sha


def main():
    root, records = Path('method/frozen_probe'), Path('method/research_lab/records')
    entry = root / 'scripts/test_adapted_open_set_evaluator.py'
    sources = set(dependencies(entry)) | set(dependencies(root / 'tests/test_open_set_evaluation.py'))
    sources |= {root / 'scripts/register_adapted_open_set_evaluator.py', root / 'scripts/register_axis_stage.py'}
    snapshots = {'source_snapshot/' + str(path.relative_to(root)): str(path) for path in sorted(sources)}
    inputs = {path: sha(path) for path in snapshots.values()}
    for path in (records / 'decisions/D078.md', records / 'protocols/OSCRADAPTTEST001_joint_metrics_v1.md'):
        inputs[str(path)] = sha(path)
    registration = json.loads((records / 'experiments/OSCRREFTEST002_registration.json').read_text())
    previous = Path(registration['output_dir'])
    assert json.loads((previous / 'status.json').read_text())['status'] == 'completed'
    reference = json.loads((previous / 'reference_tests.json').read_text())
    assert reference['fixture_audit_passed'] and not reference['reference_validated_for_H007']
    for path in previous.rglob('*'):
        if path.is_file(): inputs[str(path)] = sha(path)
    caps = {target: Path(source).stat().st_size for target, source in snapshots.items()}
    caps.update({'config.json': 65536, 'environment.json': 32768, 'status.json': 65536,
                 'error.txt': 65536, 'combined_log.txt': 32768, 'adapted_tests.json': 262144})
    config = dict(protocol='OSCRADAPTTEST001_joint_metrics_v1', seed=1701, timeout_seconds=60,
                  GPU_allocation_seconds=0, HPO_trials=0, source_snapshots=snapshots, input_hashes=inputs,
                  artifact_file_caps=caps, artifact_manifest_cap=65536, artifact_ceiling_bytes=2 << 20,
                  float_atol=1e-12, software_only=True, scientific_quality_access='none')
    sys.path.insert(0, str(root))
    from bounded_artifacts import inspect_file_plan
    plan = inspect_file_plan({**caps, 'artifact_manifest.json': config['artifact_manifest_cap']}, config['artifact_ceiling_bytes'], ('status.json', 'error.txt'))
    assert plan['passed']
    name = 'OSCRADAPTTEST001'
    path = root / f'configs/{name}.json'
    put(path, config)
    digest = sha(path)
    run_id = f'{name}_{datetime.now(timezone.utc):%Y%m%d}_{digest[:8]}_s1701'
    output = root / 'runtime' / run_id
    command = ['/home/haipd/miniconda3/bin/python', str(entry), '--config', str(path), '--run-id', run_id, '--output-dir', str(output), '--seed', '1701']
    reg = dict(registered_utc=datetime.now(timezone.utc).isoformat(), run_id=run_id, config=str(path),
               config_sha256=digest, output_dir=str(output), CPU_limit_seconds=60, GPU_seconds=0,
               durable_bytes_ceiling=config['artifact_ceiling_bytes'], maximum_planned_logical_bytes=plan['maximum_logical_bytes'],
               input_hashes=inputs, command=command)
    put(records / f'experiments/{name}_registration.json', reg)
    with (records / f'experiments/{name}.md').open('x') as handle:
        handle.write(f'# {name} — registered adapted evaluator software check\n\n{run_id}; outcome pending. No sign-task quality/test access.\n')
    print(json.dumps({k: v for k, v in reg.items() if k != 'input_hashes'}, indent=2))


if __name__ == '__main__':
    main()
