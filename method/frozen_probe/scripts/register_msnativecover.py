"""Register full fixed candidate audit after completed native-cost reference gates."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

from register_axis_stage import dependencies,put,sha


def main():
    root,records=Path('method/frozen_probe'),Path('method/research_lab/records');name='MSNATIVECOVER001'
    prior=json.loads((records/'experiments/MSNATIVECOST002_registration.json').read_text());parent=Path(prior['output_dir'])
    assert json.loads((parent/'status.json').read_text())['status']=='completed'
    summary=json.loads((parent/'native_scan_summary.json').read_text())
    assert summary['no_incomplete_rows'] and summary['status_counts']=={'eligible':11}
    rows=json.loads((parent/'native_scan_records.json').read_text())
    assert sum(x.get('old_native_reference_equal') is True for x in rows)==4
    c=json.loads(Path(prior['config']).read_text())
    for key in ('selected_sample_ids','selection_rule','reference_control_samples','prior_native_runtime','prior_id_mapping'):
        c.pop(key,None)
    sources=dependencies(root/'scripts/audit_msasl_native_scan_v3.py')+[Path(__file__).resolve().relative_to(Path.cwd()),root/'scripts/register_axis_stage.py']
    snapshots={'source_snapshot/'+str(x.relative_to(root)):str(x) for x in sorted(set(sources))}
    inputs=c['input_hashes'];inputs.update({x:sha(x) for x in snapshots.values()})
    for p in parent.rglob('*'):
        if p.is_file():inputs[str(p)]=sha(p)
    for p in (records/'decisions/D057.md',records/'protocols/MSNATIVECOVER001_full_eligibility.md'):
        inputs[str(p)]=sha(p)
    caps={target:Path(x).stat().st_size for target,x in snapshots.items()}
    caps.update({'config.json':131072,'environment.json':32768,'status.json':65536,'error.txt':65536,'combined_log.txt':131072,
                 'source_parents.json':1<<20,'native_scan_records.json':12<<20,'native_scan_summary.json':131072})
    c.update(protocol='MSNATIVECOVER001_full_eligibility',timeout_seconds=7200,progress_every_rows=100,
             source_snapshots=snapshots,artifact_file_caps=caps,artifact_manifest_cap=262144,artifact_ceiling_bytes=20<<20)
    assert len(json.loads(Path(c['candidate_records_path']).read_text())['candidates'])==3425
    sys.path.insert(0,str(root));from bounded_artifacts import inspect_file_plan
    plan=inspect_file_plan({**caps,'artifact_manifest.json':c['artifact_manifest_cap']},c['artifact_ceiling_bytes'],('status.json','error.txt'));assert plan['passed']
    cfg=root/'configs'/f'{name}.json';put(cfg,c);h=sha(cfg);rid=f'{name}_20261004_{h[:8]}_s1701'
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),run_id=rid,config=str(cfg),config_sha256=h,
             output_dir=str(root/'runtime'/rid),CPU_limit_seconds=7200,GPU_hours=0,
             durable_bytes_ceiling=c['artifact_ceiling_bytes'],maximum_planned_logical_bytes=plan['maximum_logical_bytes'],input_hashes=inputs)
    put(records/'experiments'/f'{name}_registration.json',reg)
    with (records/'experiments'/f'{name}.md').open('x') as f:f.write(f'# {name} — Registered full3425native audit\n\n{rid}; outcome pending.\n')
    print(json.dumps({k:v for k,v in reg.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':main()
