"""Exclusive preregistration of bounded native CPU fixtures and four sources."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from register_axis_stage import dependencies,sha,put

ROOT=Path('method/frozen_probe');R=Path('method/research_lab/records')


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['tests','smoke'],required=True);a=p.parse_args()
    name='MSNATIVETEST001' if a.stage=='tests' else 'MSNATIVE001';entry=ROOT/'scripts/audit_msasl_native.py'
    sources=dependencies(entry)
    if a.stage=='tests':sources.append(ROOT/'tests/test_msasl_native.py')
    snapshots={'source_snapshot/'+str(s.relative_to(ROOT)):str(s) for s in sources}
    inputs={str(s):sha(s) for s in sources}
    for s in (Path(__file__),ROOT/'scripts/register_axis_stage.py',R/'decisions/D044.md',R/'protocols/MSNATIVE001_cpu_smoke.md'):
        inputs[str(s)]=sha(s)
    prior=json.loads((R/'experiments/MSMETA001_registration.json').read_text());original=Path(prior['output_dir'])
    for s in (original/'official_annotation_archive.bin',original/'release_provenance.json',original/'status.json'):
        inputs[str(s)]=sha(s)
    caps={k:Path(v).stat().st_size for k,v in snapshots.items()}
    caps.update({'config.json':65536,'environment.json':32768,'status.json':65536,'error.txt':65536,'combined_log.txt':65536})
    c=dict(protocol='MSNATIVE001_cpu_smoke',seed=1701,timeout_seconds=120,GPU_allocation_seconds=0,HPO_trials=0,
           source_snapshots=snapshots,input_hashes=inputs,artifact_file_caps=caps,artifact_manifest_cap=65536,
           artifact_ceiling_bytes=(2 if a.stage=='tests' else 20)<<20)
    sys.path.insert(0,str(ROOT));from bounded_artifacts import array_spec,inspect_file_plan
    if a.stage=='tests':c['software_only']=True;caps['native_tests.json']=16384
    else:
        reg=json.loads((R/'experiments/MSNATIVETEST001_registration.json').read_text());test=Path(reg['output_dir'])
        assert json.loads((test/'status.json').read_text())['status']=='completed' and json.loads((test/'native_tests.json').read_text())['passed']
        for s in (test/'status.json',test/'native_tests.json',test/'artifact_manifest.json',R/'inventory/MSMETA002_metadata_audit.json'):
            inputs[str(s)]=sha(s)
        metadata=json.loads((ROOT/'configs/MSMETA002.json').read_text());poolpath=Path(metadata['public_pool_csv']);inputs[str(poolpath)]=sha(poolpath)
        pool={row[0] for row in csv.reader(poolpath.open())};assert len(pool)==39197
        from msasl_metadata_v2 import youtube_id
        for s in (ROOT/'msasl_metadata.py',ROOT/'msasl_metadata_v2.py'):inputs[str(s)]=sha(s)
        dataset=Path(metadata['dataset_root']);samples=[]
        for split in ('train','val'):
            p=dataset/f'MSASL_{split}.json';inputs[str(p)]=sha(p);rows=json.loads(p.read_bytes())
            for nonzero in (False,True):
                for i,row in enumerate(rows):
                    key=youtube_id(row['url']);path=dataset/'video'/f'{key}.mp4'
                    if path.is_file() and key not in pool and 0<row['end_time']-row['start_time']<=8 and (row['start_time']>0)==nonzero:
                        if path.stat().st_size>32<<20:raise ValueError('Fixed source exceeds byte limit; no reselection')
                        sid=f'{split}_{i:05d}';inputs[str(path)]=sha(path)
                        samples.append(dict(sample_id=sid,split=split,annotation_index=i,parent_id=key,path=str(path),annotation=row));break
                else:raise ValueError('Required fixed source unavailable')
        c.update(samples=samples,pts_reference_atol_seconds=1e-6)
        for item in samples:
            sid=item['sample_id'];caps[f'{sid}_first_native_window.npy']=array_spec((16,224,224,3),'u1')['file_bytes']
            caps[f'{sid}_native_pts.npy']=array_spec((240,),'<f8')['file_bytes'];caps[f'{sid}_crop_preview.png']=262144
        caps['native_audit.json']=524288
    plan=inspect_file_plan({**caps,'artifact_manifest.json':65536},c['artifact_ceiling_bytes'],('status.json','error.txt'));assert plan['passed']
    config=ROOT/'configs'/f'{name}.json';put(config,c);h=sha(config);rid=f'{name}_20261004_{h[:8]}_s1701'
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),run_id=rid,config=str(config),config_sha256=h,output_dir=str(ROOT/'runtime'/rid),
             CPU_limit_seconds=120,GPU_hours=0,durable_bytes_ceiling=c['artifact_ceiling_bytes'],maximum_planned_logical_bytes=plan['maximum_logical_bytes'],input_hashes=inputs)
    put(R/'experiments'/f'{name}_registration.json',reg)
    with (R/'experiments'/f'{name}.md').open('x') as f:f.write(f'# {name} — Registered CPU diagnostic\n\n{rid}; outcome pending.\n')
    print(json.dumps({k:v for k,v in reg.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':main()
