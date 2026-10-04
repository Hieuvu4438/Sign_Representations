"""Exclusive DICTTEST001 synthetic prerequisite registration."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

from register_axis_stage import dependencies,put,sha


def main():
    root,records=Path('method/frozen_probe'),Path('method/research_lab/records');name='DICTTEST001'
    entry=root/'scripts/test_dictionary_protocol.py';test=root/'tests/test_dictionary_protocol.py'
    sources=set(dependencies(entry)+dependencies(test)+[Path(__file__).resolve().relative_to(Path.cwd()),root/'scripts/register_axis_stage.py'])
    snapshots={'source_snapshot/'+str(x.relative_to(root)):str(x) for x in sorted(sources)}
    inputs={x:sha(x) for x in snapshots.values()}
    for p in (records/'decisions/D059.md',records/'protocols/DICTTEST001_micro_macro_sampling.md',
              records/'hypotheses/H006_v3_MSASL_transfer_DRAFT.md',records/'protocols/P001_v1_parity.json'):
        inputs[str(p)]=sha(p)
    caps={target:Path(x).stat().st_size for target,x in snapshots.items()}
    caps.update({'config.json':65536,'environment.json':32768,'status.json':65536,'error.txt':65536,
                 'combined_log.txt':65536,'dictionary_tests.json':16384})
    c=dict(protocol='DICTTEST001_micro_macro_sampling',seed=1701,timeout_seconds=120,GPU_allocation_seconds=0,
           HPO_trials=0,software_only=True,source_snapshots=snapshots,input_hashes=inputs,
           artifact_file_caps=caps,artifact_manifest_cap=65536,artifact_ceiling_bytes=2<<20)
    sys.path.insert(0,str(root));from bounded_artifacts import inspect_file_plan
    plan=inspect_file_plan({**caps,'artifact_manifest.json':c['artifact_manifest_cap']},c['artifact_ceiling_bytes'],('status.json','error.txt'));assert plan['passed']
    cfg=root/'configs'/f'{name}.json';put(cfg,c);h=sha(cfg);rid=f'{name}_20261004_{h[:8]}_s1701'
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),run_id=rid,config=str(cfg),config_sha256=h,
             output_dir=str(root/'runtime'/rid),CPU_limit_seconds=120,GPU_hours=0,
             durable_bytes_ceiling=c['artifact_ceiling_bytes'],maximum_planned_logical_bytes=plan['maximum_logical_bytes'],input_hashes=inputs)
    put(records/'experiments'/f'{name}_registration.json',reg)
    with (records/'experiments'/f'{name}.md').open('x') as f:f.write(f'# {name} — Registered seven-group synthetic protocol checks\n\n{rid}; outcome pending.\n')
    print(json.dumps({k:v for k,v in reg.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':main()
