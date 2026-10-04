"""Prepare a GPU smoke only after completed CPU/context dependencies."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from register_axis_stage import sha,put


def main():
    ROOT=Path('method/frozen_probe');R=Path('method/research_lab/records');name='MSFEATUREGPU001'
    reg=json.loads((R/'experiments/MSFEATURECPU001_registration.json').read_text());parent=Path(reg['output_dir'])
    assert json.loads((parent/'status.json').read_text())['status']=='completed'
    report=json.loads((parent/'feature_smoke.json').read_text());assert report['strict_load'] and report['eval_mode']
    c=json.loads((parent/'config.json').read_text());c.update(device='gpu',H_id='H006',GPU_allocation_seconds=120,
        CPU_reference_runtime=str(parent),program_ceiling_seconds=14400,hypothesis_ceiling_seconds=120,
        runtime_roots=['method/frozen_probe/runtime','method/research_lab/runtime','method/stream_reliability/runtime'],
        cost_errata=[dict(path='method/stream_reliability/runtime/GLAUDIT001_20261004_H003_v0_0bf71e65_s1701/graphics_accounting.json',
                          sha256='6bb427a85a8a0a2dbca3b61f8082aa72f76818d14153615d11aca18a78f96e17')],
        granted_uuid='GPU-3775c606-49da-34fe-322c-756749074f5b',min_free_mib=8192,max_gpu_utilization=10)
    for p in parent.rglob('*'):
        if p.is_file():c['input_hashes'][str(p)]=sha(p)
    for p in (Path(__file__),R/'decisions/D049.md',R/'protocols/MSFEATURE001_native_reference.md'):
        c['input_hashes'][str(p)]=sha(p)
    src=Path('method/stream_reliability/gpu_accounting.py');target='source_snapshot/gpu_accounting.py'
    c['input_hashes'][str(src)]=sha(src);c['source_snapshots'][target]=str(src);c['artifact_file_caps'][target]=src.stat().st_size
    for e in c['cost_errata']:c['input_hashes'][e['path']]=e['sha256']
    sys.path.insert(0,str(ROOT));from bounded_artifacts import inspect_file_plan
    plan=inspect_file_plan({**c['artifact_file_caps'],'artifact_manifest.json':c['artifact_manifest_cap']},c['artifact_ceiling_bytes'],('status.json','error.txt'));assert plan['passed']
    config=ROOT/'configs'/f'{name}.json';put(config,c);h=sha(config);rid=f'{name}_20261004_{h[:8]}_s1701'
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),run_id=rid,config=str(config),config_sha256=h,output_dir=str(ROOT/'runtime'/rid),
        H_id='H006',CPU_limit_seconds=120,GPU_allocation_seconds=120,hypothesis_cumulative_ceiling_seconds=120,
        durable_bytes_ceiling=c['artifact_ceiling_bytes'],maximum_planned_logical_bytes=plan['maximum_logical_bytes'],input_hashes=c['input_hashes'])
    put(R/'experiments'/f'{name}_registration.json',reg)
    with (R/'experiments'/f'{name}.md').open('x') as f:f.write(f'# {name} — Registered, resource gated\n\n{rid}; not launched.\n')
    print(json.dumps({k:v for k,v in reg.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':main()
