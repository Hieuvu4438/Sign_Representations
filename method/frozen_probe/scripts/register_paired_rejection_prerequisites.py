"""New immutable IDs for the explicit paired-dictionary protocol version."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from register_axis_stage import dependencies, put, sha


def read(path): return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=('tests', 'pool', 'cost'), required=True); a = p.parse_args()
    root, r = Path('method/frozen_probe'), Path('method/research_lab/records')
    name, entry_name = dict(tests=('HEADPAIRTEST001', 'test_paired_rejection_episodes.py'),
        pool=('HEADPAIRPOOL001', 'pool_paired_rejection_source.py'), cost=('HEADPAIRCOST001', 'measure_rejection_head_cost.py'))[a.stage]
    entry = root / 'scripts' / entry_name
    sources = set(dependencies(entry)) | {root / 'scripts/register_paired_rejection_prerequisites.py', root / 'scripts/register_axis_stage.py'}
    if a.stage == 'tests': sources |= set(dependencies(root / 'tests/test_paired_rejection_episodes.py'))
    if a.stage == 'cost': sources |= set(dependencies(Path('method/stream_reliability/gpu_accounting.py')))
    snapshots = {'source_snapshot/' + (str(p.relative_to(root)) if p.is_relative_to(root) else 'accounting/' + p.name): str(p) for p in sorted(sources)}
    inputs = {p: sha(p) for p in snapshots.values()}
    for p in (r / 'decisions/D087.md', r / 'hypotheses/H007_v3.md', r / 'protocols/P007_v3_paired_dictionary_controls.md',
              r / 'hypotheses/H007_v2.md', r / 'protocols/P007_v2_supervised_rejection_controls.md'):
        inputs[str(p)] = sha(p)
    # Original author architecture/loss gate is still required, not rerun.
    prior = read(r / 'experiments/HEADTEST001_registration.json'); directory = Path(prior['output_dir'])
    assert read(directory / 'status.json')['status'] == 'completed' and read(directory / 'head_tests.json')['passed']
    for p in directory.rglob('*'):
        if p.is_file(): inputs[str(p)] = sha(p)
    caps = {n: Path(p).stat().st_size for n, p in snapshots.items()}
    caps.update({'config.json': (128 if a.stage == 'tests' else 1024) << 10,
                 'environment.json': 65536, 'status.json': 262144, 'error.txt': 65536, 'combined_log.txt': 65536})
    c = dict(protocol='P007_v3_paired_dictionary_controls', stage=a.stage, H_id='H007', seed=1701,
             timeout_seconds=60 if a.stage == 'tests' else 300 if a.stage == 'pool' else 180,
             GPU_allocation_seconds=60 if a.stage == 'cost' else 0, HPO_trials=0,
             source_snapshots=snapshots, input_hashes=inputs, artifact_file_caps=caps,
             artifact_manifest_cap=262144, artifact_ceiling_bytes={'tests': 2, 'pool': 20, 'cost': 4}[a.stage] << 20)
    if a.stage == 'tests': caps['pair_tests.json'] = 65536
    else:
        prior = read(r / 'experiments/HEADPAIRTEST001_registration.json'); directory = Path(prior['output_dir'])
        assert read(directory / 'status.json')['status'] == 'completed' and read(directory / 'pair_tests.json')['passed']
        for p in directory.rglob('*'):
            if p.is_file(): inputs[str(p)] = sha(p)
        sys.path.insert(0, str(root)); from bounded_artifacts import array_spec
        if a.stage == 'pool':
            prior = read(r / 'experiments/OSMETA001_registration.json'); metadata = Path(prior['output_dir']); mc = read(metadata / 'config.json')
            inputs.update(mc['input_hashes'])
            for p in metadata.rglob('*'):
                if p.is_file(): inputs[str(p)] = sha(p)
            base = mc['asl_inputs']['ASLCitizen_base']; identity = read(base['identity_config'])
            inputs[identity['video_audit']] = sha(identity['video_audit'])
            c.update(metadata_runtime=str(metadata), base_inputs=base, training_steps=1000)
            c['array_specs'] = {'fit_vectors.npy': array_spec((1027, 768), '<f8'), 'training_plan.npy': array_spec((1000, 16, 52), '<i2')}
            caps.update({n: s['file_bytes'] for n, s in c['array_specs'].items()})
            caps.update({'source_roster.json': 1 << 20, 'pool_summary.json': 65536})
        else:
            prior = read(r / 'experiments/HEADPAIRPOOL001_registration.json'); directory = Path(prior['output_dir'])
            assert read(directory / 'status.json')['status'] == 'completed' and read(directory / 'pool_summary.json')['passed']
            inputs.update(read(directory / 'config.json')['input_hashes'])
            for p in directory.rglob('*'):
                if p.is_file(): inputs[str(p)] = sha(p)
            c.update(pool_runtime=str(directory), paired_episodes=True)
            policy = read(root / 'configs/MSFULLFEAT001.json')
            for key in ('program_ceiling_seconds', 'runtime_roots', 'cost_errata', 'granted_uuid', 'min_free_mib', 'max_gpu_utilization'): c[key] = policy[key]
            c['hypothesis_ceiling_seconds'] = 420
            for item in c['cost_errata']: inputs[item['path']] = item['sha256']
            c['array_specs'] = {'cost_losses.npy': array_spec((100, 2), '<f8')}
            caps.update({n: s['file_bytes'] for n, s in c['array_specs'].items()})
            caps['head_cost.json'] = 65536
    sys.path.insert(0, str(root)); from bounded_artifacts import inspect_file_plan
    plan = inspect_file_plan({**caps, 'artifact_manifest.json': c['artifact_manifest_cap']}, c['artifact_ceiling_bytes'], ('status.json', 'error.txt')); assert plan['passed']
    config = root / f'configs/{name}.json'; put(config, c); digest = sha(config)
    rid = f'{name}_{datetime.now(timezone.utc):%Y%m%d}_{digest[:8]}_s1701'; output = root / 'runtime' / rid
    command = ['/home/haipd/miniconda3/bin/python', str(entry), '--config', str(config), '--run-id', rid, '--output-dir', str(output), '--seed', '1701']
    reg = dict(registered_utc=datetime.now(timezone.utc).isoformat(), run_id=rid, config=str(config), config_sha256=digest,
               output_dir=str(output), CPU_limit_seconds=c['timeout_seconds'], GPU_seconds=c['GPU_allocation_seconds'],
               durable_bytes_ceiling=c['artifact_ceiling_bytes'], maximum_planned_logical_bytes=plan['maximum_logical_bytes'],
               input_hashes=inputs, command=command)
    put(r / f'experiments/{name}_registration.json', reg)
    with (r / f'experiments/{name}.md').open('x') as handle: handle.write(f'# {name} — paired head prerequisite\n\n{rid}; outcome pending.\n')
    print(json.dumps({k: v for k, v in reg.items() if k != 'input_hashes'}, indent=2))


if __name__ == '__main__': main()
