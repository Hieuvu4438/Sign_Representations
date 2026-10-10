"""Register bounded post-outcome reporting without changing scientific protocol."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from register_axis_stage import dependencies,put,sha


def main():
    root,r=Path('method/frozen_probe'),Path('method/research_lab/records');name='MSREPORT001';entry=root/'scripts/report_msasl_geometry.py'
    sources=dependencies(entry)+[root/'scripts/register_msasl_report.py',root/'scripts/register_axis_stage.py']
    snapshots={'source_snapshot/'+str(x.relative_to(root)):str(x) for x in sorted(set(sources))};inputs={x:sha(x) for x in snapshots.values()}
    for path in (r/'protocols/MSREPORT001_descriptive_export.md',r/'literature/NOVELTY_20261004_H006_r2.md'):inputs[str(path)]=sha(path)
    parents={}
    for key in ('MSEVAL001','MSEVALVERIFY001'):
        reg=json.loads((r/f'experiments/{key}_registration.json').read_text());parent=Path(reg['output_dir'])
        assert json.loads((parent/'status.json').read_text())['status']=='completed';parents[key]=str(parent)
        for path in parent.rglob('*'):
            if path.is_file():inputs[str(path)]=sha(path)
    caps={x:Path(source).stat().st_size for x,source in snapshots.items()}
    caps.update({'config.json':131072,'environment.json':32768,'status.json':131072,'error.txt':65536,'combined_log.txt':65536,
        'metrics.csv':32768,'descriptive_cases.json':65536,'geometry_comparison.png':524288,'geometry_comparison.pdf':262144,'report_summary.json':32768})
    c=dict(protocol='MSREPORT001_descriptive_export',seed=1701,GPU_allocation_seconds=0,timeout_seconds=60,HPO_trials=0,
        measured_runtime=parents['MSEVAL001'],verified_runtime=parents['MSEVALVERIFY001'],
        artifact_file_caps=caps,artifact_manifest_cap=131072,artifact_ceiling_bytes=4<<20,input_hashes=inputs,source_snapshots=snapshots)
    sys.path.insert(0,str(root));from bounded_artifacts import inspect_file_plan
    plan=inspect_file_plan({**caps,'artifact_manifest.json':c['artifact_manifest_cap']},c['artifact_ceiling_bytes'],('status.json','error.txt'));assert plan['passed']
    cfg=root/'configs'/(name+'.json');put(cfg,c);digest=sha(cfg);rid=f'{name}_20261005_{digest[:8]}_s1701'
    command=['/home/haipd/miniconda3/bin/python',str(entry),'--config',str(cfg),'--run-id',rid,'--output-dir',str(root/'runtime'/rid),'--seed','1701']
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),run_id=rid,config=str(cfg),config_sha256=digest,
        output_dir=str(root/'runtime'/rid),CPU_limit_seconds=60,GPU_seconds=0,durable_bytes_ceiling=4<<20,
        maximum_planned_logical_bytes=plan['maximum_logical_bytes'],input_hashes=inputs,command=command)
    put(r/f'experiments/{name}_registration.json',reg)
    with (r/f'experiments/{name}.md').open('x') as f:f.write(f'# {name} — registered descriptive export\n\n{rid}; outcome pending.\n')
    print(json.dumps({k:v for k,v in reg.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':main()
