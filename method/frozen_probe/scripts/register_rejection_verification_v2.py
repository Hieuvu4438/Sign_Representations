"""Exclusive stats-gate/quality/recount stages for the fixed H007_v1 diagnostic."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from register_axis_stage import dependencies, put, sha


def read(path): return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=('verify',), required=True); a = p.parse_args()
    root, r = Path('method/frozen_probe'), Path('method/research_lab/records')
    name = dict(tests='REJECTANALTEST001', measure='REJECTEVAL001', verify='REJECTVERIFY002')[a.stage]
    entry = root / 'scripts' / dict(tests='test_rejection_analysis.py', measure='run_rejection_diagnostic.py', verify='verify_rejection_diagnostic_v2.py')[a.stage]
    sources = set(dependencies(entry)) | {root / 'scripts/register_rejection_verification_v2.py', root / 'scripts/register_axis_stage.py'}
    if a.stage == 'tests': sources |= set(dependencies(root / 'tests/test_rejection_analysis.py'))
    snapshots = {'source_snapshot/' + str(x.relative_to(root)): str(x) for x in sorted(sources)}
    inputs = {x: sha(x) for x in snapshots.values()}
    for path in (r / 'decisions/D081.md', r / 'hypotheses/H007_v1.md', r / 'protocols/P007_v1_rejection_transfer.md'):
        inputs[str(path)] = sha(path)
    caps = {n: Path(x).stat().st_size for n, x in snapshots.items()}
    caps.update({'config.json': (128 if a.stage == 'tests' else 1024) << 10, 'environment.json': 65536,
                 'status.json': 262144, 'error.txt': 65536, 'combined_log.txt': 65536})
    c = dict(protocol='P007_v1_rejection_transfer', stage=a.stage, seed=1701, timeout_seconds=60 if a.stage == 'tests' else 300,
             GPU_allocation_seconds=0, HPO_trials=0, source_snapshots=snapshots, input_hashes=inputs,
             artifact_file_caps=caps, artifact_manifest_cap=262144, artifact_ceiling_bytes=(2 if a.stage == 'tests' else 20) << 20)
    if a.stage == 'tests':
        path = r / 'protocols/REJECTANALTEST001_manual_oracles_v1.md'; inputs[str(path)] = sha(path)
        caps['rejection_analysis_tests.json'] = 65536
    elif a.stage == 'measure':
        for key, report in [('REJECTANALTEST001', 'rejection_analysis_tests.json'), ('REJECTSCORETEST001', 'rejection_score_tests.json'), ('OSCRADAPTTEST001', 'adapted_tests.json'), ('OSMETAVERIFY001', 'metadata_verification.json')]:
            prior = read(r / f'experiments/{key}_registration.json'); directory = Path(prior['output_dir'])
            assert read(directory / 'status.json')['status'] == 'completed' and read(directory / report)['passed']
            for x in directory.rglob('*'):
                if x.is_file(): inputs[str(x)] = sha(x)
        reg = read(r / 'experiments/OSMETA001_registration.json'); metadata = Path(reg['output_dir']); mc = read(metadata / 'config.json')
        assert read(metadata / 'status.json')['status'] == 'completed'
        inputs.update(mc['input_hashes'])
        for x in metadata.rglob('*'):
            if x.is_file(): inputs[str(x)] = sha(x)
        tasks = read(metadata / 'candidate_partitions.json')
        base = tasks['ASLCitizen_base']; wanted = set(base['calibration_ids'] + base['dev_ids'])
        for v in base['seeds'].values(): wanted.update(v['galleries']['50']['support_ids'])
        assert len(wanted) == 805
        prior = read(mc['asl_inputs']['ASLCitizen_base']['identity_config'])
        assert sha(prior['video_audit']) == prior['video_audit_sha256']; inputs[prior['video_audit']] = sha(prior['video_audit'])
        total = len(base['calibration_ids']) + len(base['dev_ids']) + len(tasks['MSASL100']['dev_ids']); assert total == 1412
        c.update(metadata_runtime=str(metadata), base_inputs=mc['asl_inputs']['ASLCitizen_base'],
                 feature_runtime=mc['msasl_inputs']['runtime'], base_ids=sorted(wanted),
                 support_seeds=[1701,1702,1703], gallery_sizes=[10,25,50], score_names=['maxcosine','margin','maxsoftmax','NAC16'],
                 calibration_policies=['FROZEN50','MATCHED_SOURCE_SIZE'], target_fraction=[19,20],
                 nac_k=16, score_scale=1., bootstrap_replicates=2000, primary_drift_floor=.05,
                 workspace_plan_bytes=512 << 20, expected_total_queries=total)
        sys.path.insert(0, str(root)); from bounded_artifacts import array_spec
        shapes = {'base_vectors.npy': (805,768), 'cosine50.npy': (3,total,50), 'scores.npy': (3,3,4,total),
                  'predictions.npy': (3,3,total), 'known_rank.npy': (3,3,total),
                  'count_curves.npy': (4*9*(total+3),4), 'bootstrap_base.npy': (2000,5), 'bootstrap_MS.npy': (2000,19),
                  'bootstrap_primary.npy': (2000,), 'bootstrap_primary_valid.npy': (2000,)}
        c['array_specs'] = {n: array_spec(s, '|b1' if n.endswith('_valid.npy') else '<i2' if n in ('predictions.npy','known_rank.npy','bootstrap_base.npy','bootstrap_MS.npy') else '<f8') for n,s in shapes.items()}
        caps.update({n: spec['file_bytes'] for n,spec in c['array_specs'].items()})
        caps.update({'rejection_metadata.json': 2 << 20, 'rejection_eval.json': 1 << 20})
    else:
        reg = read(r / 'experiments/REJECTEVAL001_registration.json'); directory = Path(reg['output_dir'])
        assert read(directory / 'status.json')['status'] == 'completed'
        c['measured_runtime'] = str(directory)
        inputs.update(read(directory / 'config.json')['input_hashes'])
        for x in directory.rglob('*'):
            if x.is_file(): inputs[str(x)] = sha(x)
        caps['rejection_verification.json'] = 1 << 20
        old = read(r / 'experiments/REJECTVERIFY001_registration.json'); failed = Path(old['output_dir'])
        assert read(failed / 'status.json')['status'] == 'failed'
        for x in failed.rglob('*'):
            if x.is_file(): inputs[str(x)] = sha(x)
        for x in (r / 'decisions/D083.md', r / 'protocols/P007_v1_secondary_DCG_FP64_erratum_r1.md'):
            inputs[str(x)] = sha(x)
        c['protocol'] = 'P007_v1_rejection_transfer_secondary_DCG_FP64_r1'
        caps['dcg_precision_erratum.json'] = 131072

    sys.path.insert(0, str(root)); from bounded_artifacts import inspect_file_plan
    plan = inspect_file_plan({**caps, 'artifact_manifest.json': c['artifact_manifest_cap']}, c['artifact_ceiling_bytes'], ('status.json','error.txt')); assert plan['passed']
    config = root / f'configs/{name}.json'; put(config,c); digest=sha(config)
    rid=f'{name}_{datetime.now(timezone.utc):%Y%m%d}_{digest[:8]}_s1701'; output=root/'runtime'/rid
    command=['/home/haipd/miniconda3/bin/python',str(entry),'--config',str(config),'--run-id',rid,'--output-dir',str(output),'--seed','1701']
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),run_id=rid,config=str(config),config_sha256=digest,
             output_dir=str(output),CPU_limit_seconds=c['timeout_seconds'],GPU_seconds=0,durable_bytes_ceiling=c['artifact_ceiling_bytes'],
             maximum_planned_logical_bytes=plan['maximum_logical_bytes'],input_hashes=inputs,command=command)
    put(r/f'experiments/{name}_registration.json',reg)
    with (r/f'experiments/{name}.md').open('x') as f:f.write(f'# {name} — fixed H007 stage\n\n{rid}; outcome pending.\n')
    print(json.dumps({k:v for k,v in reg.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':main()
