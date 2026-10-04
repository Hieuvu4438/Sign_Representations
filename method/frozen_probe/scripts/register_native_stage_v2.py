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
    name='MSNATIVETEST002' if a.stage=='tests' else 'MSNATIVE002';entry=ROOT/'scripts/audit_msasl_native_v2.py'
    sources=dependencies(entry)
    if a.stage=='tests':sources.extend([ROOT/'tests/test_msasl_native.py',ROOT/'tests/test_native_pts_artifact.py'])
    snapshots={'source_snapshot/'+str(s.relative_to(ROOT)):str(s) for s in sources}
    inputs={str(s):sha(s) for s in sources}
    for s in (Path(__file__),ROOT/'scripts/register_axis_stage.py',R/'decisions/D045.md',R/'protocols/MSNATIVE002_cpu_smoke.md'):
        inputs[str(s)]=sha(s)
    prior=json.loads((R/'experiments/MSMETA001_registration.json').read_text());original=Path(prior['output_dir'])
    for s in (original/'official_annotation_archive.bin',original/'release_provenance.json',original/'status.json'):
        inputs[str(s)]=sha(s)
    caps={k:Path(v).stat().st_size for k,v in snapshots.items()}
    caps.update({'config.json':65536,'environment.json':32768,'status.json':65536,'error.txt':65536,'combined_log.txt':65536})
    c=dict(protocol='MSNATIVE002_cpu_smoke',seed=1701,timeout_seconds=120,GPU_allocation_seconds=0,HPO_trials=0,
           source_snapshots=snapshots,input_hashes=inputs,artifact_file_caps=caps,artifact_manifest_cap=65536,
           artifact_ceiling_bytes=(2 if a.stage=='tests' else 20)<<20)
    sys.path.insert(0,str(ROOT));from bounded_artifacts import array_spec,inspect_file_plan
    if a.stage=='tests':c['software_only']=True;caps['native_tests.json']=16384
    else:
        reg=json.loads((R/'experiments/MSNATIVETEST002_registration.json').read_text());test=Path(reg['output_dir'])
        assert json.loads((test/'status.json').read_text())['status']=='completed' and json.loads((test/'native_tests.json').read_text())['passed']
        for s in (test/'status.json',test/'native_tests.json',test/'artifact_manifest.json',R/'inventory/MSMETA002_metadata_audit.json'):
            inputs[str(s)]=sha(s)
        originalconfig=ROOT/'configs/MSNATIVE001.json';prior=json.loads(originalconfig.read_text())
        samples=prior['samples'];inputs[str(originalconfig)]=sha(originalconfig)
        for item in samples:
            path=item['path'];inputs[path]=prior['input_hashes'][path]
            assert sha(path)==inputs[path]
        for path,digest in prior['input_hashes'].items():
            if path.startswith('/home/dongvk/datasets/MS-ASL/') or path.endswith('youtube-sl-25-metadata.csv'):
                inputs[path]=digest
        fail=json.loads((R/'experiments/MSNATIVE001_registration.json').read_text())
        for path in (Path(fail['output_dir'])/'status.json',Path(fail['output_dir'])/'error.txt'):
            inputs[str(path)]=sha(path)
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
