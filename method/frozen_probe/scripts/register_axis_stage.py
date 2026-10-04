"""Create exclusive prelaunch records for the fixed axis/spectrum stages."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('method/frozen_probe')
RECORDS = Path('method/research_lab/records')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def dependencies(entry):
    pending, seen = [entry], set()
    while pending:
        path = pending.pop()
        if path in seen: continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.module] if isinstance(node, ast.ImportFrom) and node.module else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            for name in names:
                candidate = ROOT / (name.split('.')[0] + '.py')
                if candidate.is_file(): pending.append(candidate)
    return sorted(seen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['tests', 'measure', 'verify'], required=True)
    args = parser.parse_args()
    name = {'tests': 'AXISTEST001', 'measure': 'AXISTRANSFER001', 'verify': 'AXISVERIFY001'}[args.stage]
    entry = ROOT / 'scripts' / ('verify_axis_spectrum.py' if args.stage == 'verify' else 'run_axis_spectrum.py')
    sources = dependencies(entry)
    if args.stage == 'tests': sources += [ROOT/'tests/test_axis_spectrum.py']
    snapshots = {'source_snapshot/'+str(p.relative_to(ROOT)):str(p) for p in sources}
    inputs = {str(p):sha(p) for p in sources}
    extras = {'tests': ['decisions/D038.md', 'protocols/AXISTEST001_manual_oracles.md'],
              'measure': ['decisions/D039.md', 'hypotheses/H006_v2.md', 'protocols/H006_v2_axis_spectrum.md',
                          'protocols/P001_v1_parity.json', 'literature/sources/S35.md', 'literature/sources/S37.md'],
              'verify': ['protocols/AXISVERIFY001_original_metric.md']}[args.stage]
    for relative in extras: inputs[str(RECORDS/relative)] = sha(RECORDS/relative)
    caps = {target:Path(source).stat().st_size for target,source in snapshots.items()}
    caps.update({'config.json':65536,'environment.json':32768,'status.json':65536,'error.txt':65536,'combined_log.txt':65536})
    c = dict(protocol={'tests':'AXISTEST001_manual_oracles','measure':'H006_v2_axis_spectrum','verify':'AXISVERIFY001_original_metric'}[args.stage],
             seed=1701,timeout_seconds=120,GPU_allocation_seconds=0,HPO_trials=0,source_snapshots=snapshots,input_hashes=inputs,
             artifact_file_caps=caps,artifact_manifest_cap=65536,artifact_ceiling_bytes=2097152)
    if args.stage == 'tests':
        c['software_only']=True; caps['axis_tests.json']=16384
    elif args.stage == 'measure':
        prior_reg=json.loads((RECORDS/'experiments/AXISTEST001_registration.json').read_text()); testroot=Path(prior_reg['output_dir'])
        if json.loads((testroot/'status.json').read_text())['status']!='completed' or not json.loads((testroot/'axis_tests.json').read_text())['passed']:
            raise ValueError('Axis tests not passed')
        for path in (testroot/'status.json',testroot/'axis_tests.json'):inputs[str(path)]=sha(path)
        reg=json.loads((RECORDS/'experiments/SHRINKTRANSFER001_registration.json').read_text()); parent=Path(reg['output_dir'])
        c.update(parent_runtime=str(parent),relative_gap_floor=1e-12,factor_atol=1e-10,reference_whitening_atol=1e-12,primary_gain_floor=.005,artifact_ceiling_bytes=20971520)
        for path in (parent/'status.json',parent/'config.json',parent/'fitted_shrinkage.npz',parent/'predictions.json',parent/'fit_support_metadata.json'):
            inputs[str(path)]=sha(path)
        import sys
        sys.path.insert(0,str(ROOT))
        from bounded_artifacts import array_spec
        shapes={'basis_R.npy':(768,768),'basis_T.npy':(768,768),'spectrum_R.npy':(768,),'spectrum_T.npy':(768,),
                'mean.npy':(768,),'target_vectors.npy':(661,768),'ranks.npy':(4596,100),'row_metrics.npy':(4596,6)}
        c['array_specs']={n:array_spec(s,'<i2' if n=='ranks.npy' else '<f8') for n,s in shapes.items()}
        caps.update({n:spec['file_bytes'] for n,spec in c['array_specs'].items()})
        caps.update({'shared_metadata.json':524288,'axis_spectrum.json':131072})
    else:
        measured=json.loads((RECORDS/'experiments/AXISTRANSFER001_registration.json').read_text())
        c['measured_runtime']=measured['output_dir']; caps['axis_verification.json']=131072
        root=Path(c['measured_runtime'])
        for path in sorted(root.rglob('*')):
            if path.is_file(): inputs[str(path)]=sha(path)
    import sys
    sys.path.insert(0,str(ROOT))
    from bounded_artifacts import inspect_file_plan
    plan=inspect_file_plan({**caps,'artifact_manifest.json':c['artifact_manifest_cap']},c['artifact_ceiling_bytes'],('status.json','error.txt'))
    if not plan['passed']:raise ValueError('Stage complete artifact bound exceeds ceiling')
    config=ROOT/'configs'/f'{name}.json';put(config,c);digest=sha(config)
    run_id=f'{name}_20261004_{digest[:8]}_s1701'
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),run_id=run_id,config=str(config),config_sha256=digest,
             output_dir=str(ROOT/'runtime'/run_id),GPU_hours=0,CPU_limit_seconds=120,durable_bytes_ceiling=c['artifact_ceiling_bytes'],
             maximum_planned_logical_bytes=plan['maximum_logical_bytes'],input_hashes=inputs)
    put(RECORDS/'experiments'/f'{name}_registration.json',reg)
    with (RECORDS/'experiments'/f'{name}.md').open('x') as stream:stream.write(f'# {name} — Registered fixed stage\n\n{run_id}; actual outcomes follow.\n')
    print(json.dumps({k:v for k,v in reg.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':main()
