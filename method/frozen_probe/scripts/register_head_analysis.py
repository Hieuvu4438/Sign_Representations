"""Register separate CPU audit after all three preregistered fits finish."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from register_axis_stage import dependencies, put, sha


def read(path): return json.loads(Path(path).read_text())


def main():
    root = Path('method/frozen_probe'); r = Path('method/research_lab/records')
    entry = root / 'scripts/analyze_rejection_head_pilots.py'
    sources = set(dependencies(entry)) | {root / 'scripts/register_head_analysis.py', root / 'scripts/register_axis_stage.py'}
    snapshots = {'source_snapshot/' + str(p.relative_to(root)): str(p) for p in sorted(sources)}
    inputs = {p: sha(p) for p in snapshots.values()}
    parents = {}
    for key in ('HEADPILOT001', 'HEADPILOT002', 'HEADPILOT003', 'HEADPAIRPOOL001', 'REJECTEVAL001', 'REJECTVERIFY002'):
        reg = read(r / f'experiments/{key}_registration.json'); directory = Path(reg['output_dir'])
        assert read(directory / 'status.json')['status'] == 'completed'
        parents[key] = directory; inputs.update(read(directory / 'config.json')['input_hashes'])
        for p in directory.rglob('*'):
            if p.is_file(): inputs[str(p)] = sha(p)
    for p in (r / 'decisions/D089.md', r / 'protocols/HEADANALYSIS001_saved_head_audit.md'):
        inputs[str(p)] = sha(p)
    caps = {n: Path(p).stat().st_size for n, p in snapshots.items()}
    caps.update({'config.json': 2 << 20, 'environment.json': 65536, 'status.json': 262144,
        'error.txt': 65536, 'combined_log.txt': 65536, 'per_seed.csv': 1 << 20,
        'mean.csv': 262144, 'head_analysis.json': 1 << 20})
    sys.path.insert(0, str(root))
    from bounded_artifacts import array_spec, inspect_file_plan
    specs = {'head_count_curves.npy': array_spec((77000, 4), '<f8'),
        'bootstrap_head_deltas.npy': array_spec((2000, 3), '<f8'),
        'bootstrap_head_valid.npy': array_spec((2000, 3), '|b1')}
    caps.update({n: s['file_bytes'] for n, s in specs.items()})
    c = dict(protocol='HEADANALYSIS001_saved_head_audit', H_id='H007', seed=1701,
        timeout_seconds=300, GPU_allocation_seconds=0, HPO_trials=0, source_snapshots=snapshots,
        input_hashes=inputs, artifact_file_caps=caps, artifact_manifest_cap=262144,
        artifact_ceiling_bytes=10 << 20, array_specs=specs,
        head_runtimes=[str(parents[key]) for key in ('HEADPILOT001', 'HEADPILOT002', 'HEADPILOT003')],
        pool_runtime=str(parents['HEADPAIRPOOL001']), measured_runtime=str(parents['REJECTEVAL001']))
    plan = inspect_file_plan({**caps, 'artifact_manifest.json': c['artifact_manifest_cap']}, c['artifact_ceiling_bytes'], ('status.json', 'error.txt'))
    assert plan['passed']
    name = 'HEADANALYSIS001'; config = root / f'configs/{name}.json'; put(config, c); digest = sha(config)
    rid = f'{name}_{datetime.now(timezone.utc):%Y%m%d}_{digest[:8]}_s1701'; output = root / 'runtime' / rid
    command = ['/home/haipd/miniconda3/bin/python', str(entry), '--config', str(config), '--run-id', rid, '--output-dir', str(output), '--seed', '1701']
    reg = dict(registered_utc=datetime.now(timezone.utc).isoformat(), run_id=rid, config=str(config),
        config_sha256=digest, output_dir=str(output), CPU_limit_seconds=300, GPU_seconds=0,
        durable_bytes_ceiling=c['artifact_ceiling_bytes'], maximum_planned_logical_bytes=plan['maximum_logical_bytes'], input_hashes=inputs, command=command)
    put(r / f'experiments/{name}_registration.json', reg)
    with (r / f'experiments/{name}.md').open('x') as f: f.write(f'# {name} — saved-weight and joint-screen audit\n\n{rid}; outcome pending.\n')
    print(json.dumps({k: v for k, v in reg.items() if k != 'input_hashes'}, indent=2))


if __name__ == '__main__': main()
