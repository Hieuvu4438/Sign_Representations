"""Exclusive registered parser regression and fixed-subset cost retry1."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

from register_axis_stage import dependencies,put,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['tests','cost'],required=True);stage=p.parse_args().stage
    root,records=Path('method/frozen_probe'),Path('method/research_lab/records')
    name='MSNATIVESCANTEST002' if stage=='tests' else 'MSNATIVECOST002'
    old=json.loads((records/'experiments/MSNATIVECOST001_registration.json').read_text())
    c=json.loads(Path(old['config']).read_text())
    entry=root/'scripts/audit_msasl_native_scan_v3.py'
    sources=dependencies(entry)+[Path(__file__).resolve().relative_to(Path.cwd()),root/'scripts/register_axis_stage.py']
    if stage=='tests':sources.append(root/'tests/test_msasl_native_scan_v2.py')
    snapshots={'source_snapshot/'+str(x.relative_to(root)):str(x) for x in sorted(set(sources))}
    inputs=dict(c['input_hashes']);inputs.update({x:sha(x) for x in snapshots.values()})
    for x in (Path(old['config']),records/'experiments/MSNATIVECOST001_registration.json',
              Path(old['output_dir'])/'status.json',records/'inventory/MSNATIVECOST001_FFPROBE_DIAGNOSIS.json',records/'decisions/D056.md'):
        inputs[str(x)]=sha(x)
    caps={target:Path(x).stat().st_size for target,x in snapshots.items()}
    caps.update({'config.json':65536,'environment.json':32768,'status.json':65536,'error.txt':65536,'combined_log.txt':65536})
    c.update(protocol='MSNATIVESCAN001_eligibility_D056_native_tick_field',source_snapshots=snapshots,input_hashes=inputs,
             artifact_file_caps=caps,timeout_seconds=120 if stage=='tests' else 1800)
    if stage=='tests':
        c['software_only']=True;caps['native_scan_tests.json']=16384
    else:
        tr=json.loads((records/'experiments/MSNATIVESCANTEST002_registration.json').read_text());t=Path(tr['output_dir'])
        assert json.loads((t/'status.json').read_text())['status']=='completed'
        assert json.loads((t/'native_scan_tests.json').read_text())['passed']
        for x in t.rglob('*'):
            if x.is_file():inputs[str(x)]=sha(x)
        caps.update({'source_parents.json':65536,'native_scan_records.json':262144,'native_scan_summary.json':65536})
    sys.path.insert(0,str(root));from bounded_artifacts import inspect_file_plan
    plan=inspect_file_plan({**caps,'artifact_manifest.json':c['artifact_manifest_cap']},c['artifact_ceiling_bytes'],('status.json','error.txt'));assert plan['passed']
    cfg=root/'configs'/f'{name}.json';put(cfg,c);h=sha(cfg);rid=f'{name}_20261004_{h[:8]}_s1701'
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),run_id=rid,config=str(cfg),config_sha256=h,
             output_dir=str(root/'runtime'/rid),CPU_limit_seconds=c['timeout_seconds'],GPU_hours=0,
             durable_bytes_ceiling=c['artifact_ceiling_bytes'],maximum_planned_logical_bytes=plan['maximum_logical_bytes'],input_hashes=inputs)
    put(records/'experiments'/f'{name}_registration.json',reg)
    with (records/'experiments'/f'{name}.md').open('x') as f:f.write(f'# {name} — Registered D056 fixed stage\n\n{rid}; outcome pending.\n')
    print(json.dumps({k:v for k,v in reg.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':main()
