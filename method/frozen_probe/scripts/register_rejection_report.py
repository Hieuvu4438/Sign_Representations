"""Exclusive bounded derived export registration after complete verification."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from register_axis_stage import dependencies,put,sha


def main():
    root,r=Path('method/frozen_probe'),Path('method/research_lab/records');entry=root/'scripts/report_rejection_diagnostic.py'
    sources=set(dependencies(entry))|{root/'scripts/register_rejection_report.py',root/'scripts/register_axis_stage.py'}
    snapshots={'source_snapshot/'+str(p.relative_to(root)):str(p) for p in sorted(sources)};inputs={p:sha(p) for p in snapshots.values()}
    for p in (r/'decisions/D084.md',r/'protocols/REJECTREPORT001_descriptive_export.md'):inputs[str(p)]=sha(p)
    parents={}
    for key in ('REJECTEVAL001','REJECTVERIFY002'):
        reg=json.loads((r/f'experiments/{key}_registration.json').read_text());parent=Path(reg['output_dir']);parents[key]=parent
        assert json.loads((parent/'status.json').read_text())['status']=='completed'
        for p in parent.rglob('*'):
            if p.is_file():inputs[str(p)]=sha(p)
    assert json.loads((parents['REJECTVERIFY002']/'rejection_verification.json').read_text())['passed']
    caps={n:Path(p).stat().st_size for n,p in snapshots.items()}
    caps.update({'config.json':131072,'environment.json':65536,'status.json':131072,'error.txt':65536,'combined_log.txt':65536,
                 'per_seed.csv':262144,'mean.csv':65536,'numerical_cases.json':131072,'report_summary.json':65536,
                 'joint_curves.png':1048576,'joint_curves.pdf':1048576})
    c=dict(protocol='REJECTREPORT001_descriptive_export',seed=1701,timeout_seconds=60,GPU_allocation_seconds=0,HPO_trials=0,
           source_snapshots=snapshots,input_hashes=inputs,artifact_file_caps=caps,artifact_manifest_cap=131072,artifact_ceiling_bytes=4<<20,
           measured_runtime=str(parents['REJECTEVAL001']),verified_runtime=str(parents['REJECTVERIFY002']),score_names=['maxcosine','margin','maxsoftmax','NAC16'])
    sys.path.insert(0,str(root));from bounded_artifacts import inspect_file_plan
    plan=inspect_file_plan({**caps,'artifact_manifest.json':c['artifact_manifest_cap']},c['artifact_ceiling_bytes'],('status.json','error.txt'));assert plan['passed']
    name='REJECTREPORT001';config=root/f'configs/{name}.json';put(config,c);digest=sha(config);rid=f'{name}_{datetime.now(timezone.utc):%Y%m%d}_{digest[:8]}_s1701';out=root/'runtime'/rid
    command=['/home/haipd/miniconda3/bin/python',str(entry),'--config',str(config),'--run-id',rid,'--output-dir',str(out),'--seed','1701']
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),run_id=rid,config=str(config),config_sha256=digest,output_dir=str(out),
             CPU_limit_seconds=60,GPU_seconds=0,durable_bytes_ceiling=c['artifact_ceiling_bytes'],maximum_planned_logical_bytes=plan['maximum_logical_bytes'],input_hashes=inputs,command=command)
    put(r/f'experiments/{name}_registration.json',reg)
    with (r/f'experiments/{name}.md').open('x') as f:f.write(f'# {name} — derived evidence export\n\n{rid}; outcome pending.\n')
    print(json.dumps({k:v for k,v in reg.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':main()
