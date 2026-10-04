"""Register all38 actual incomplete native rows for cause localization."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

from register_axis_stage import dependencies,put,sha


def main():
    root,records=Path('method/frozen_probe'),Path('method/research_lab/records');name='MSPTSDIAG001'
    old=json.loads((records/'experiments/MSNATIVECOVER001_registration.json').read_text());parent=Path(old['output_dir'])
    assert json.loads((parent/'status.json').read_text())['status']=='failed'
    a=json.loads((parent/'native_scan_records.json').read_text());failed=[x for x in a if x['status']=='incomplete']
    assert len(failed)==38 and all(x['split']=='train' and x['reason']=='native_pts_not_increasing' for x in failed)
    entry=root/'scripts/diagnose_msasl_pts.py'
    sources=dependencies(entry)+[Path(__file__).resolve().relative_to(Path.cwd()),root/'scripts/register_axis_stage.py']
    snapshots={'source_snapshot/'+str(p.relative_to(root)):str(p) for p in sorted(set(sources))}
    inputs={p:sha(p) for p in snapshots.values()}
    for p in parent.rglob('*'):
        if p.is_file():inputs[str(p)]=sha(p)
    train=Path('/home/dongvk/datasets/MS-ASL/MSASL_train.json');raw=Path('/home/dongvk/datasets/MS-ASL/video')
    for p in (train,records/'decisions/D061.md',records/'protocols/MSPTSDIAG001_failure_localization.md'):
        inputs[str(p)]=sha(p)
    identities=json.loads((parent/'source_parents.json').read_text())
    for x in failed:inputs[str(raw/(x['parent_id']+'.mp4'))]=identities[x['parent_id']]['sha256']
    caps={target:Path(p).stat().st_size for target,p in snapshots.items()}
    caps.update({'config.json':131072,'environment.json':32768,'status.json':65536,'error.txt':65536,'combined_log.txt':65536,
                 'pts_diagnosis_records.json':12<<20,'pts_diagnosis_summary.json':65536})
    c=dict(protocol='MSPTSDIAG001_failure_localization',seed=1701,timeout_seconds=300,GPU_allocation_seconds=0,HPO_trials=0,
           source_snapshots=snapshots,input_hashes=inputs,artifact_file_caps=caps,artifact_manifest_cap=131072,artifact_ceiling_bytes=20<<20,
           failed_runtime=str(parent),train_metadata=str(train),raw_root=str(raw),failed_sample_ids=[x['sample_id'] for x in failed],
           max_decoded_frames=4096,reference_timeout_seconds=30,reference_byte_limit=1<<20)
    sys.path.insert(0,str(root));from bounded_artifacts import inspect_file_plan
    plan=inspect_file_plan({**caps,'artifact_manifest.json':c['artifact_manifest_cap']},c['artifact_ceiling_bytes'],('status.json','error.txt'));assert plan['passed']
    cfg=root/'configs'/f'{name}.json';put(cfg,c);h=sha(cfg);rid=f'{name}_20261004_{h[:8]}_s1701'
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),run_id=rid,config=str(cfg),config_sha256=h,
             output_dir=str(root/'runtime'/rid),CPU_limit_seconds=300,GPU_hours=0,durable_bytes_ceiling=c['artifact_ceiling_bytes'],
             maximum_planned_logical_bytes=plan['maximum_logical_bytes'],input_hashes=inputs)
    put(records/'experiments'/f'{name}_registration.json',reg)
    with (records/'experiments'/f'{name}.md').open('x') as f:f.write(f'# {name} — Registered38row diagnostic\n\n{rid}; outcome pending.\n')
    print(json.dumps({k:v for k,v in reg.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':main()
