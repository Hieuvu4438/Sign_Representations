"""Register inference-environment inputs or deterministic target dictionaries."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from register_axis_stage import dependencies,put,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('environment','gallery'),required=True);a=p.parse_args()
    root,r=Path('method/frozen_probe'),Path('method/research_lab/records')
    env=a.stage=='environment';name='MSESTENV001' if env else 'MSDICT001'
    protocol='MSESTENV001_native_input_parity' if env else 'MSDICT001_fixed_supports'
    entry=root/'scripts'/('check_msasl_inference_inputs.py' if env else 'freeze_msasl_dictionary.py')
    sources=dependencies(entry)+[root/'scripts/register_msdictionary_stage.py',root/'scripts/register_axis_stage.py']
    if env:sources.append(root/'tests/test_msasl_estimated_clock.py')
    snapshots={'source_snapshot/'+str(x.relative_to(root)):str(x) for x in sorted(set(sources))}
    inputs={x:sha(x) for x in snapshots.values()}
    for x in (r/'decisions/D067.md',r/'protocols'/(protocol+'.md'),r/'inventory/MSEST_COHORT_BINDING_VERIFICATION.json'):
        inputs[str(x)]=sha(x)
    assert json.loads((r/'inventory/MSEST_COHORT_BINDING_VERIFICATION.json').read_text())['passed']
    reg=json.loads((r/'experiments/MSESTCOVER001_registration.json').read_text());parent=Path(reg['output_dir'])
    assert json.loads((parent/'status.json').read_text())['status']=='completed'
    pc=json.loads((parent/'config.json').read_text());rows=json.loads((parent/'adapted_cohort_records.json').read_text())
    for x in parent.rglob('*'):
        if x.is_file():inputs[str(x)]=sha(x)
    candidate=Path(pc['candidate_records_path']);inputs[str(candidate)]=sha(candidate)
    caps={x:Path(source).stat().st_size for x,source in snapshots.items()}
    caps.update({'config.json':131072,'environment.json':32768,'status.json':65536,'error.txt':65536,'combined_log.txt':65536})
    c=dict(protocol=protocol,seed=1701,timeout_seconds=600 if env else 120,GPU_allocation_seconds=0,HPO_trials=0,
           source_snapshots=snapshots,input_hashes=inputs,artifact_file_caps=caps,artifact_manifest_cap=131072,
           artifact_ceiling_bytes=2<<20 if env else 4<<20,cohort_runtime=str(parent),candidate_records_path=str(candidate))
    if env:
        affected=[x['sample_id'] for x in rows if x['prior_raw_clock_status']=='incomplete']
        controls=[min(x['sample_id'] for x in rows if x['split']==s and x['status']=='eligible' and x['prior_raw_clock_status']!='incomplete') for s in ('train','val')]
        controls += [x['sample_id'] for x in rows if x['status']=='native_short']
        ids=affected+controls;assert len(ids)==len(set(ids))==43
        metadata=Path(pc['dataset_root']);clock=Path(pc['clock_runtime'])
        for x in clock.rglob('*'):
            if x.is_file():inputs[str(x)]=sha(x)
        for s in ('train','val'):inputs[str(metadata/f'MSASL_{s}.json')]=sha(metadata/f'MSASL_{s}.json')
        candidate_rows={x['sample_id']:x for x in json.loads(candidate.read_text())['candidates']};byid={x['sample_id']:x for x in rows}
        for id in ids:inputs[candidate_rows[id]['raw_path']]=byid[id]['source_sha256']
        inputs[pc['ffprobe']]=sha(pc['ffprobe'])
        c.update(sample_ids=ids,clock_runtime=str(clock),dataset_root=str(metadata),ffprobe=pc['ffprobe'])
        caps.update({'estimated_clock_tests.json':32768,'inference_input_parity.json':524288})
    else:
        test=json.loads((r/'experiments/DICTTEST001_registration.json').read_text());tr=Path(test['output_dir'])
        assert json.loads((tr/'status.json').read_text())['status']=='completed'
        for x in tr.rglob('*'):
            if x.is_file():inputs[str(x)]=sha(x)
        c.update(gallery_seeds=[1701,1702,1703]);caps.update({'dictionary_manifest.json':131072,'extraction_manifest.json':2<<20,'dictionary_summary.json':65536})
    sys.path.insert(0,str(root));from bounded_artifacts import inspect_file_plan
    plan=inspect_file_plan({**caps,'artifact_manifest.json':c['artifact_manifest_cap']},c['artifact_ceiling_bytes'],('status.json','error.txt'));assert plan['passed']
    cfg=root/'configs'/(name+'.json');put(cfg,c);h=sha(cfg);rid=f'{name}_20261004_{h[:8]}_s1701'
    python=str(Path('.venv-signrep/bin/python').absolute()) if env else '/home/haipd/miniconda3/bin/python'
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),run_id=rid,config=str(cfg),config_sha256=h,output_dir=str(root/'runtime'/rid),
             CPU_limit_seconds=c['timeout_seconds'],GPU_hours=0,durable_bytes_ceiling=c['artifact_ceiling_bytes'],
             maximum_planned_logical_bytes=plan['maximum_logical_bytes'],input_hashes=inputs,
             command=[python,str(entry),'--config',str(cfg),'--run-id',rid,'--output-dir',str(root/'runtime'/rid),'--seed','1701'])
    put(r/f'experiments/{name}_registration.json',reg)
    with (r/f'experiments/{name}.md').open('x') as f:f.write(f'# {name} — registered technical/dictionary stage\n\n{rid}; outcome pending.\n')
    print(json.dumps({k:v for k,v in reg.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':main()
