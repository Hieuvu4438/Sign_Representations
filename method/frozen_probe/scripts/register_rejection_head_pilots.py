"""Register all three initialization seeds before any head-quality result."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from register_axis_stage import dependencies, put, sha


def read(path): return json.loads(Path(path).read_text())


def main():
    root, r = Path('method/frozen_probe'), Path('method/research_lab/records')
    entry = root / 'scripts/train_rejection_head_pilot.py'
    sources = set(dependencies(entry)) | set(dependencies(Path('method/stream_reliability/gpu_accounting.py'))) | {
        root / 'scripts/register_rejection_head_pilots.py', root / 'scripts/register_axis_stage.py'}
    snapshots = {'source_snapshot/' + (str(p.relative_to(root)) if p.is_relative_to(root) else 'accounting/' + p.name): str(p) for p in sorted(sources)}
    inputs = {p: sha(p) for p in snapshots.values()}
    for p in (r / 'decisions/D088.md', r / 'hypotheses/H007_v3.md', r / 'protocols/P007_v3_paired_dictionary_controls.md',
              r / 'protocols/P007_v2_supervised_rejection_controls.md', r / 'hypotheses/H007_v2.md'):
        inputs[str(p)] = sha(p)
    parents = {}
    for key, proof in [('HEADTEST001', 'head_tests.json'), ('HEADPAIRTEST001', 'pair_tests.json'),
                       ('HEADPAIRPOOL001', 'pool_summary.json'), ('HEADPAIRCOST001', 'head_cost.json'),
                       ('REJECTVERIFY002', 'rejection_verification.json')]:
        reg = read(r / f'experiments/{key}_registration.json'); directory = Path(reg['output_dir'])
        assert read(directory / 'status.json')['status'] == 'completed' and read(directory / proof)['passed']
        inputs.update(read(directory / 'config.json')['input_hashes']); parents[key] = directory
        for p in directory.rglob('*'):
            if p.is_file(): inputs[str(p)] = sha(p)
    measured = Path(read(r / 'experiments/REJECTEVAL001_registration.json')['output_dir'])
    for p in measured.rglob('*'):
        if p.is_file(): inputs[str(p)] = sha(p)
    for p in (r / 'literature/sources/S48_METHOD_r1.md', r / 'literature/NOVELTY_20261005_H007_r1.md'):
        inputs[str(p)] = sha(p)
    for p in (root / 'references/fsosar_c8115c09_model').rglob('*'):
        if p.is_file(): inputs[str(p)] = sha(p)
    sys.path.insert(0, str(root)); from bounded_artifacts import array_spec, inspect_file_plan
    caps = {n: Path(p).stat().st_size for n, p in snapshots.items()}
    caps.update({'config.json': 1 << 20, 'environment.json': 65536, 'status.json': 262144,
                 'error.txt': 65536, 'combined_log.txt': 65536, 'head_evaluation.json': 1 << 20})
    specs = {'head_weights.npy': array_spec((2411650,), '<f4'), 'head_scores.npy': array_spec((3, 3, 2, 1412), '<f4'),
             'train_losses.npy': array_spec((1000, 2), '<f8'), 'training_counts.npy': array_spec((1000, 3), '<i2')}
    caps.update({n: s['file_bytes'] for n, s in specs.items()})
    policy = read(root / 'configs/HEADPAIRCOST001.json')
    plan = inspect_file_plan({**caps, 'artifact_manifest.json': 262144}, 20 << 20, ('status.json', 'error.txt')); assert plan['passed']
    for i, seed in enumerate((1701, 1702, 1703), start=1):
        name = f'HEADPILOT00{i}'
        c = dict(protocol='P007_v3_paired_dictionary_controls', H_id='H007', seed=seed,
            timeout_seconds=300, GPU_allocation_seconds=120, HPO_trials=0,
            source_snapshots=snapshots, input_hashes=inputs, artifact_file_caps=caps,
            artifact_manifest_cap=262144, artifact_ceiling_bytes=20 << 20, array_specs=specs,
            pool_runtime=str(parents['HEADPAIRPOOL001']), measured_runtime=str(measured))
        for key in ('program_ceiling_seconds', 'hypothesis_ceiling_seconds', 'runtime_roots', 'cost_errata', 'granted_uuid', 'min_free_mib', 'max_gpu_utilization'):
            c[key] = policy[key]
        config = root / f'configs/{name}.json'; put(config, c); digest = sha(config)
        rid = f'{name}_{datetime.now(timezone.utc):%Y%m%d}_{digest[:8]}_s{seed}'; output = root / 'runtime' / rid
        command = ['/home/haipd/miniconda3/bin/python', str(entry), '--config', str(config), '--run-id', rid, '--output-dir', str(output), '--seed', str(seed)]
        reg = dict(registered_utc=datetime.now(timezone.utc).isoformat(), run_id=rid, config=str(config), config_sha256=digest,
            output_dir=str(output), CPU_limit_seconds=300, GPU_seconds=120, durable_bytes_ceiling=20 << 20,
            maximum_planned_logical_bytes=plan['maximum_logical_bytes'], input_hashes=inputs, command=command)
        put(r / f'experiments/{name}_registration.json', reg)
        with (r / f'experiments/{name}.md').open('x') as handle: handle.write(f'# {name} — fixed head pilot\n\n{rid}; outcome pending.\n')
        print(json.dumps({k: v for k, v in reg.items() if k != 'input_hashes'}, indent=2))


if __name__ == '__main__': main()
