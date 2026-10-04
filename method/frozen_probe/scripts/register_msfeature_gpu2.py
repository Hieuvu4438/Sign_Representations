"""Fresh GPU resource policy based on observed same-checkpoint memory usage."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

from register_axis_stage import dependencies,put,sha


def main():
    root,records=Path('method/frozen_probe'),Path('method/research_lab/records');name='MSFEATUREGPU002'
    parent=records/'experiments/MSFEATUREGPU001_registration.json';old=json.loads(parent.read_text());c=json.loads(Path(old['config']).read_text())
    assert not Path(old['output_dir']).exists()
    mempath=records/'experiments/GPU004_report.json';mem=json.loads(mempath.read_text())
    assert mem['input_hashes']['checkpoint']==c['input_hashes'][c['checkpoint']]
    assert mem['dtype']=='float32' and mem['batch_size']==2 and mem['peak_reserved_bytes']<1<<30
    entry=root/'scripts/audit_msasl_features_v2.py'
    sources=dependencies(entry)+[Path(__file__).resolve().relative_to(Path.cwd()),root/'scripts/register_axis_stage.py',root/'scripts/audit_signrep.py']
    for p in sorted(set(sources)):
        target='source_snapshot/'+str(p.relative_to(root));c['source_snapshots'][target]=str(p)
        c['input_hashes'][str(p)]=sha(p);c['artifact_file_caps'][target]=p.stat().st_size
    for p in (parent,Path(old['config']),mempath,records/'experiments/GPU004_status.json',
              records/'decisions/D060.md',records/'protocols/MSFEATURE002_bounded_vram.md'):
        c['input_hashes'][str(p)]=sha(p)
    c.update(protocol='MSFEATURE002_bounded_vram',min_free_mib=3072,max_gpu_utilization=100,torch_allocator_cap_mib=2048,
             contended_measurement=True)
    sys.path.insert(0,str(root));from bounded_artifacts import inspect_file_plan
    plan=inspect_file_plan({**c['artifact_file_caps'],'artifact_manifest.json':c['artifact_manifest_cap']},c['artifact_ceiling_bytes'],('status.json','error.txt'));assert plan['passed']
    cfg=root/'configs'/f'{name}.json';put(cfg,c);h=sha(cfg);rid=f'{name}_20261004_{h[:8]}_s1701'
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),run_id=rid,config=str(cfg),config_sha256=h,
             output_dir=str(root/'runtime'/rid),H_id='H006',CPU_limit_seconds=120,GPU_allocation_seconds=120,
             hypothesis_cumulative_ceiling_seconds=120,durable_bytes_ceiling=c['artifact_ceiling_bytes'],
             maximum_planned_logical_bytes=plan['maximum_logical_bytes'],input_hashes=c['input_hashes'])
    put(records/'experiments'/f'{name}_registration.json',reg)
    with (records/'experiments'/f'{name}.md').open('x') as f:f.write(f'# {name} — Registered 2GiB allocator numerical smoke\n\n{rid}; outcome pending.\n')
    print(json.dumps({k:v for k,v in reg.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':main()
