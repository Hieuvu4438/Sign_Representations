"""Exclusive registration of fixed baseline/calibration manual/reference gate."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from register_axis_stage import dependencies, put, sha


def main():
    root, r = Path('method/frozen_probe'), Path('method/research_lab/records')
    entry = root / 'scripts/test_frozen_rejection_scores.py'
    sources = set(dependencies(entry)) | set(dependencies(root / 'tests/test_frozen_rejection.py'))
    sources |= {root / 'scripts/register_rejection_score_tests.py', root / 'scripts/register_axis_stage.py',
                root / 'references/nac_971f6767/utils.py', root / 'references/nac_971f6767/LICENSE'}
    snapshots = {'source_snapshot/' + str(p.relative_to(root)): str(p) for p in sorted(sources)}
    inputs = {p: sha(p) for p in snapshots.values()}
    for p in (r / 'decisions/D080.md', r / 'protocols/REJECTSCORETEST001_manual_oracles_v1.md'): inputs[str(p)] = sha(p)
    prior = json.loads((r / 'experiments/OSCRADAPTTEST001_registration.json').read_text())
    for p in (Path(prior['output_dir']) / 'status.json', Path(prior['output_dir']) / 'adapted_tests.json'): inputs[str(p)] = sha(p)
    caps = {n: Path(p).stat().st_size for n, p in snapshots.items()}
    caps.update({'config.json': 65536, 'environment.json': 32768, 'status.json': 65536,
                 'error.txt': 65536, 'combined_log.txt': 32768, 'rejection_score_tests.json': 65536})
    c = dict(protocol='REJECTSCORETEST001_manual_oracles_v1', seed=1701, timeout_seconds=60, GPU_allocation_seconds=0,
             HPO_trials=0, source_snapshots=snapshots, input_hashes=inputs, artifact_file_caps=caps,
             artifact_manifest_cap=65536, artifact_ceiling_bytes=2 << 20,
             reference_source=str(root / 'references/nac_971f6767/utils.py'), float_atol=1e-12)
    sys.path.insert(0, str(root))
    from bounded_artifacts import inspect_file_plan
    plan = inspect_file_plan({**caps, 'artifact_manifest.json': c['artifact_manifest_cap']}, c['artifact_ceiling_bytes'], ('status.json', 'error.txt')); assert plan['passed']
    name = 'REJECTSCORETEST001'; path = root / f'configs/{name}.json'; put(path, c); digest = sha(path)
    rid = f'{name}_{datetime.now(timezone.utc):%Y%m%d}_{digest[:8]}_s1701'; output = root / 'runtime' / rid
    command = ['/home/haipd/miniconda3/bin/python', str(entry), '--config', str(path), '--run-id', rid, '--output-dir', str(output), '--seed', '1701']
    reg = dict(registered_utc=datetime.now(timezone.utc).isoformat(), run_id=rid, config=str(path), config_sha256=digest,
               output_dir=str(output), CPU_limit_seconds=60, GPU_seconds=0, durable_bytes_ceiling=c['artifact_ceiling_bytes'],
               maximum_planned_logical_bytes=plan['maximum_logical_bytes'], input_hashes=inputs, command=command)
    put(r / f'experiments/{name}_registration.json', reg)
    with (r / f'experiments/{name}.md').open('x') as f: f.write(f'# {name} — fixed scoring/calibration gate\n\n{rid}; outcome pending.\n')
    print(json.dumps({k:v for k,v in reg.items() if k != 'input_hashes'}, indent=2))


if __name__ == '__main__': main()
