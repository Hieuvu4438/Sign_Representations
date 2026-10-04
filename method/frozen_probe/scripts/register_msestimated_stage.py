"""Prelaunch fullclock protocol registration with unchanged prior evidence bound."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from register_axis_stage import dependencies,put,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('tests','cohort'),required=True);a=p.parse_args()
    root,records=Path('method/frozen_probe'),Path('method/research_lab/records')
    name='MSESTTEST001' if a.stage=='tests' else 'MSESTCOVER001'
    protocol='MSESTTEST001_temporal_pixel_oracles' if a.stage=='tests' else 'MSASL_ESTCLK_v1'
    entry=root/'scripts/audit_msasl_estimated_cohort.py'
    sources=dependencies(entry)+[root/'scripts/register_msestimated_stage.py',root/'scripts/register_axis_stage.py']
    if a.stage=='tests':sources.append(root/'tests/test_msasl_estimated_clock.py')
    snapshots={'source_snapshot/'+str(x.relative_to(root)):str(x) for x in sorted(set(sources))}
    inputs={x:sha(x) for x in snapshots.values()}
    for x in (records/'decisions/D066.md',records/'protocols'/(protocol+'.md')):inputs[str(x)]=sha(x)
    caps={x:Path(source).stat().st_size for x,source in snapshots.items()}
    caps.update({'config.json':131072,'environment.json':32768,'status.json':65536,'error.txt':65536,'combined_log.txt':65536})
    c=dict(protocol=protocol,seed=1701,timeout_seconds=60 if a.stage=='tests' else 600,GPU_allocation_seconds=0,HPO_trials=0,
           source_snapshots=snapshots,input_hashes=inputs,artifact_file_caps=caps,artifact_manifest_cap=131072,
           artifact_ceiling_bytes=2<<20 if a.stage=='tests' else 20<<20)
    if a.stage=='tests':c['software_only']=True;caps['estimated_clock_tests.json']=32768
    else:
        parents={}
        for id in ('MSESTTEST001','MSNATIVECOVER001','MSCLKDIAG002'):
            reg=json.loads((records/f'experiments/{id}_registration.json').read_text());parent=Path(reg['output_dir']);parents[id]=parent
            assert json.loads((parent/'status.json').read_text())['status']==('failed' if id=='MSNATIVECOVER001' else 'completed')
            if id=='MSESTTEST001':assert json.loads((parent/'estimated_clock_tests.json').read_text())['passed']
            for x in parent.rglob('*'):
                if x.is_file():inputs[str(x)]=sha(x)
        for x in (records/'decisions/D065.md',records/'inventory/MSCLK_BINDING_VERIFICATION.json'):inputs[str(x)]=sha(x)
        nroot=parents['MSNATIVECOVER001'];clock=parents['MSCLKDIAG002'];prior=json.loads((nroot/'config.json').read_text())
        candidate=Path(prior['candidate_records_path']);inputs[str(candidate)]=sha(candidate)
        dataset=Path('/home/dongvk/datasets/MS-ASL');raw=dataset/'video'
        for split in ('train','val'):inputs[str(dataset/f'MSASL_{split}.json')]=sha(dataset/f'MSASL_{split}.json')
        rows=json.loads((clock/'clock_records.json').read_text())
        for x in rows:inputs[str(raw/(x['parent_id']+'.mp4'))]=x['source_sha256']
        ffprobe='/home/haipd/miniconda3/bin/ffprobe';inputs[ffprobe]=sha(ffprobe)
        c.update(native_runtime=str(nroot),clock_runtime=str(clock),candidate_records_path=str(candidate),dataset_root=str(dataset),
                 raw_root=str(raw),affected_ids=[x['sample_id'] for x in rows],ffprobe=ffprobe,max_selected_frames=1024,max_decoded_frames=4096,
                 reference_timeout_seconds=30,reference_byte_limit=1<<20)
        caps.update({'adapted_cohort_records.json':12<<20,'changed_scan_records.json':4<<20,
                     'source_stat_validation.json':512<<10,'adapted_cohort_summary.json':131072})
    sys.path.insert(0,str(root));from bounded_artifacts import inspect_file_plan
    plan=inspect_file_plan({**caps,'artifact_manifest.json':c['artifact_manifest_cap']},c['artifact_ceiling_bytes'],('status.json','error.txt'));assert plan['passed']
    cfg=root/'configs'/(name+'.json');put(cfg,c);h=sha(cfg);rid=f'{name}_20261004_{h[:8]}_s1701'
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),run_id=rid,config=str(cfg),config_sha256=h,output_dir=str(root/'runtime'/rid),
             CPU_limit_seconds=c['timeout_seconds'],GPU_hours=0,durable_bytes_ceiling=c['artifact_ceiling_bytes'],
             maximum_planned_logical_bytes=plan['maximum_logical_bytes'],input_hashes=inputs,
             command=['/home/haipd/miniconda3/bin/python',str(entry),'--config',str(cfg),'--run-id',rid,'--output-dir',str(root/'runtime'/rid),'--seed','1701'])
    put(records/f'experiments/{name}_registration.json',reg)
    with (records/f'experiments/{name}.md').open('x') as f:f.write(f'# {name} — prospective input diagnostic\n\n{rid}; outcome pending.\n')
    print(json.dumps({k:v for k,v in reg.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':main()
