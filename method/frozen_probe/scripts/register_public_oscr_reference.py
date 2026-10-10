"""Exclusive prelaunch registration for the pinned-source/manual OSCR audit."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from register_axis_stage import dependencies, put, sha


def main():
    root, records = Path('method/frozen_probe'), Path('method/research_lab/records')
    entry = root / 'scripts/test_public_oscr_reference.py'
    sources = set(dependencies(entry)) | {root / 'scripts/register_public_oscr_reference.py', root / 'scripts/register_axis_stage.py'}
    sources |= {root / 'references/fsosar_c8115c09/utils.py', root / 'references/fsosar_c8115c09/LICENSE'}
    snapshots = {'source_snapshot/' + str(path.relative_to(root)): str(path) for path in sorted(sources)}
    inputs = {path: sha(path) for path in snapshots.values()}
    for path in (records / 'decisions/D076.md', records / 'protocols/OSCRREFTEST001_manual_oracles_v1.md'):
        inputs[str(path)] = sha(path)
    caps = {target: Path(source).stat().st_size for target, source in snapshots.items()}
    caps.update({'config.json': 65536, 'environment.json': 32768, 'status.json': 65536,
                 'error.txt': 65536, 'combined_log.txt': 32768, 'reference_tests.json': 32768})
    config = dict(protocol='OSCRREFTEST001_manual_oracles_v1', seed=1701, timeout_seconds=60,
                  GPU_allocation_seconds=0, HPO_trials=0, source_snapshots=snapshots, input_hashes=inputs,
                  artifact_file_caps=caps, artifact_manifest_cap=65536, artifact_ceiling_bytes=2 << 20,
                  reference_commit='c8115c09c54d5480bf0adf972f66988bb2274c2b',
                  reference_source=str(root / 'references/fsosar_c8115c09/utils.py'),
                  reference_license='BSD-3-Clause, complete license retained', float_atol=1e-12,
                  fixtures=[dict(id='all_known_wrong', targets=[1, 1, -1, -1], scores=[.8, .8, .2, .2], manual_joint_area=0., predicted_reference_area=.25),
                            dict(id='perfect_separated', targets=[0, 0, -1, -1], scores=[.8, .8, .2, .2], manual_joint_area=1., predicted_reference_area=1.),
                            dict(id='wrong_tied_scores', targets=[1, 1, -1, -1], scores=[.9, .9, .9, .9], manual_joint_area=0., predicted_reference_area=.25)])
    sys.path.insert(0, str(root))
    from bounded_artifacts import inspect_file_plan
    plan = inspect_file_plan({**caps, 'artifact_manifest.json': config['artifact_manifest_cap']}, config['artifact_ceiling_bytes'], ('status.json', 'error.txt'))
    assert plan['passed']
    name = 'OSCRREFTEST001'
    path = root / f'configs/{name}.json'
    put(path, config)
    digest = sha(path)
    run_id = f'{name}_{datetime.now(timezone.utc):%Y%m%d}_{digest[:8]}_s1701'
    output = root / 'runtime' / run_id
    command = ['/home/haipd/miniconda3/bin/python', str(entry), '--config', str(path), '--run-id', run_id, '--output-dir', str(output), '--seed', '1701']
    registration = dict(registered_utc=datetime.now(timezone.utc).isoformat(), run_id=run_id,
                        config=str(path), config_sha256=digest, output_dir=str(output),
                        CPU_limit_seconds=60, GPU_seconds=0, durable_bytes_ceiling=config['artifact_ceiling_bytes'],
                        maximum_planned_logical_bytes=plan['maximum_logical_bytes'], input_hashes=inputs, command=command)
    put(records / f'experiments/{name}_registration.json', registration)
    with (records / f'experiments/{name}.md').open('x') as handle:
        handle.write(f'# {name} — registered synthetic reference audit\n\n{run_id}; outcome pending. No task quality/test access.\n')
    print(json.dumps({k: v for k, v in registration.items() if k != 'input_hashes'}, indent=2))


if __name__ == '__main__':
    main()
