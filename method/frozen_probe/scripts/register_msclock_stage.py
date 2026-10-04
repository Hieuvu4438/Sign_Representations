"""Prospective registration for temporal fixtures and all38 dual-clock diagnosis."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from register_axis_stage import dependencies,put,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('tests','audit'),required=True);a=p.parse_args()
    root,records=Path('method/frozen_probe'),Path('method/research_lab/records')
    name='MSCLKTEST001' if a.stage=='tests' else 'MSCLKDIAG001'
    entry=root/'scripts/audit_msasl_clocks.py'
    sources=dependencies(entry)+[root/'scripts/register_msclock_stage.py',root/'scripts/register_axis_stage.py']
    if a.stage=='tests':sources.append(root/'tests/test_msasl_clock_audit.py')
    snapshots={'source_snapshot/'+str(x.relative_to(root)):str(x) for x in sorted(set(sources))}
    inputs={x:sha(x) for x in snapshots.values()}
    protocol='MSCLKTEST001_manual_temporal_oracles' if a.stage=='tests' else 'MSCLKDIAG001_dual_clock_correspondence'
    for x in (records/'decisions/D063.md',records/'protocols'/(protocol+'.md')):inputs[str(x)]=sha(x)
    caps={x:Path(source).stat().st_size for x,source in snapshots.items()}
    caps.update({'config.json':131072,'environment.json':32768,'status.json':65536,'error.txt':65536,'combined_log.txt':65536})
    c=dict(protocol=protocol,seed=1701,timeout_seconds=60 if a.stage=='tests' else 900,
           GPU_allocation_seconds=0,HPO_trials=0,source_snapshots=snapshots,input_hashes=inputs,
           artifact_file_caps=caps,artifact_manifest_cap=131072,artifact_ceiling_bytes=2<<20 if a.stage=='tests' else 20<<20)
    if a.stage=='tests':c.update(software_only=True);caps['clock_tests.json']=16384
    else:
        test=json.loads((records/'experiments/MSCLKTEST001_registration.json').read_text());tr=Path(test['output_dir'])
        assert json.loads((tr/'status.json').read_text())['status']=='completed' and json.loads((tr/'clock_tests.json').read_text())['passed']
        old=json.loads((records/'experiments/MSPTSDIAG001_registration.json').read_text());parent=Path(old['output_dir'])
        assert json.loads((parent/'status.json').read_text())['status']=='completed'
        data=json.loads((parent/'pts_diagnosis_records.json').read_text());assert len(data)==38
        for run in (tr,parent):
            for x in run.rglob('*'):
                if x.is_file():inputs[str(x)]=sha(x)
        train=Path('/home/dongvk/datasets/MS-ASL/MSASL_train.json');raw=Path('/home/dongvk/datasets/MS-ASL/video')
        inputs[str(train)]=sha(train)
        for row in data:
            path=raw/(row['parent_id']+'.mp4');inputs[str(path)]=row['source_sha256']
        ffprobe='/home/haipd/miniconda3/bin/ffprobe';ffmpeg='/home/haipd/miniconda3/bin/ffmpeg'
        for x in (ffprobe,ffmpeg):inputs[x]=sha(x)
        c.update(diagnostic_runtime=str(parent),sample_ids=[x['sample_id'] for x in data],train_metadata=str(train),raw_root=str(raw),
                 max_decoded_frames=4096,reference_timeout_seconds=30,reference_byte_limit=1<<20,parent_byte_limit=512<<20,
                 maximum_frame_bytes=32<<20,ffprobe=ffprobe,ffmpeg=ffmpeg)
        caps.update({'clock_records.json':16<<20,'clock_summary.json':131072})
    sys.path.insert(0,str(root));from bounded_artifacts import inspect_file_plan
    plan=inspect_file_plan({**caps,'artifact_manifest.json':c['artifact_manifest_cap']},c['artifact_ceiling_bytes'],('status.json','error.txt'))
    assert plan['passed']
    cfg=root/'configs'/(name+'.json');put(cfg,c);digest=sha(cfg);rid=f'{name}_20261004_{digest[:8]}_s1701'
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),run_id=rid,config=str(cfg),config_sha256=digest,
             output_dir=str(root/'runtime'/rid),CPU_limit_seconds=c['timeout_seconds'],GPU_hours=0,
             durable_bytes_ceiling=c['artifact_ceiling_bytes'],maximum_planned_logical_bytes=plan['maximum_logical_bytes'],input_hashes=inputs,
             command=['/home/haipd/miniconda3/bin/python',str(entry),'--config',str(cfg),'--run-id',rid,'--output-dir',str(root/'runtime'/rid),'--seed','1701'])
    put(records/'experiments'/(name+'_registration.json'),reg)
    with (records/'experiments'/(name+'.md')).open('x') as f:f.write(f'# {name} — prospective diagnostic\n\n{rid}; outcome pending.\n')
    print(json.dumps({k:v for k,v in reg.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':main()
