"""Freeze CPU metadata fixtures/census/recount with complete artifact bounds."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from register_axis_stage import dependencies,put,sha


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('tests','measure','verify'),required=True);a=p.parse_args()
    root=Path('method/frozen_probe');r=Path('method/research_lab/records')
    name=dict(tests='OSMETATEST001',measure='OSMETA001',verify='OSMETAVERIFY001')[a.stage]
    entry=root/'scripts'/('verify_open_set_metadata.py' if a.stage=='verify' else 'audit_open_set_metadata.py')
    sources=dependencies(entry)+[root/'scripts/register_open_set_metadata.py',root/'scripts/register_axis_stage.py']
    if a.stage=='tests':sources+=dependencies(root/'tests/test_open_set_metadata.py')
    snapshots={'source_snapshot/'+str(x.relative_to(root)):str(x) for x in sorted(set(sources))}
    inputs={x:sha(x) for x in snapshots.values()}
    for path in (r/'decisions/D074.md',r/'protocols/OSMETA001_calibration_feasibility_v1.md',r/'hypotheses/H007_DRAFT.md'):
        inputs[str(path)]=sha(path)
    caps={target:Path(source).stat().st_size for target,source in snapshots.items()}
    caps.update({'config.json':(256 if a.stage=='tests' else 1024)<<10,'environment.json':65536,
                 'status.json':262144,'error.txt':65536,'combined_log.txt':65536})
    c=dict(protocol='OSMETA001_calibration_feasibility_v1',stage=a.stage,seed=1701,timeout_seconds=60 if a.stage=='tests' else 300,
        GPU_allocation_seconds=0,HPO_trials=0,source_snapshots=snapshots,input_hashes=inputs,artifact_file_caps=caps,
        artifact_manifest_cap=131072 if a.stage=='tests' else 524288,artifact_ceiling_bytes=(2 if a.stage=='tests' else 8)<<20)
    if a.stage=='tests':caps['metadata_tests.json']=32768
    elif a.stage=='measure':
        tr=read(r/'experiments/OSMETATEST001_registration.json');tp=Path(tr['output_dir'])
        assert read(tp/'status.json')['status']=='completed' and read(tp/'metadata_tests.json')['passed']
        for x in tp.rglob('*'):
            if x.is_file():inputs[str(x)]=sha(x)
        c['asl_inputs']={}
        for role,cache,identity in [('ASLCitizen_base','CACHE005','BASE002'),('ASLCitizen_target','CACHE006','REPQUAL001')]:
            report_path=r/f'experiments/{cache}_report.json';report=read(report_path)
            runtime=root/'runtime'/report['run_id'];prior_path=root/f'configs/{identity}.json';prior=read(prior_path)
            manifest=read(report['cache_manifest'])
            assert read(runtime/'status.json')['status']=='completed'
            assert report['passed'] and sha(report['cache_manifest'])==report['cache_manifest_sha256']
            assert manifest['identity']==prior['cache_identity']
            c['asl_inputs'][role]=dict(report=str(report_path),identity_config=str(prior_path),runtime=str(runtime))
            for path in [report_path,prior_path,Path(report['cache_manifest']),Path(prior['cohort_manifest']),runtime/'status.json',runtime/'config.json']:
                inputs[str(path)]=sha(path)
            for e in manifest['samples']:
                path=Path(report['cache_manifest']).parent/e['file']
                assert path.name==e['file'] and not path.is_symlink() and path.stat().st_size==e['bytes']
                assert sha(path)==e['sha256'];inputs[str(path)]=e['sha256']
        reg=read(r/'experiments/MSFULLFEAT001_registration.json');parent=Path(reg['output_dir']);cfg=read(parent/'config.json')
        assert read(parent/'status.json')['status']=='completed'
        dictionary=Path(cfg['extraction_manifest']).parent/'dictionary_manifest.json'
        c['msasl_inputs']=dict(runtime=str(parent),dictionary_manifest=str(dictionary))
        for path in [parent/'config.json',parent/'status.json',parent/'artifact_manifest.json',Path(cfg['extraction_manifest']),dictionary]:
            inputs[str(path)]=sha(path)
        status=read(parent/'status.json')
        for shard in cfg['shards']:
            for name_in in (shard['records_name'],shard['array_name']):
                path=parent/name_in;assert sha(path)==status['output_hashes'][name_in];inputs[str(path)]=sha(path)
        c.update(support_seeds=[1701,1702,1703],gallery_sizes=[10,25,50],feature_arrays_loaded=0,test_access='none',
                 source_model='unmodified original raw SignRep; no geometry fitting or quality access')
        caps.update({'metadata_rows.json':2<<20,'candidate_partitions.json':262144,'metadata_census.json':131072})
    else:
        measured=read(r/'experiments/OSMETA001_registration.json');parent=Path(measured['output_dir'])
        assert read(parent/'status.json')['status']=='completed';c['measured_runtime']=str(parent)
        pc=read(parent/'config.json');inputs.update(pc['input_hashes'])
        for path in parent.rglob('*'):
            if path.is_file():inputs[str(path)]=sha(path)
        caps['metadata_verification.json']=65536
    sys.path.insert(0,str(root));from bounded_artifacts import inspect_file_plan
    plan=inspect_file_plan({**caps,'artifact_manifest.json':c['artifact_manifest_cap']},c['artifact_ceiling_bytes'],('status.json','error.txt'))
    assert plan['passed'],'Complete artifact bound failed'
    cfg=root/'configs'/(name+'.json');put(cfg,c);digest=sha(cfg);stamp=datetime.now(timezone.utc).strftime('%Y%m%d')
    rid=f'{name}_{stamp}_{digest[:8]}_s1701'
    command=['/home/haipd/miniconda3/bin/python',str(entry),'--config',str(cfg),'--run-id',rid,
             '--output-dir',str(root/'runtime'/rid),'--seed','1701']
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),run_id=rid,config=str(cfg),config_sha256=digest,
        output_dir=str(root/'runtime'/rid),CPU_limit_seconds=c['timeout_seconds'],GPU_seconds=0,
        durable_bytes_ceiling=c['artifact_ceiling_bytes'],maximum_planned_logical_bytes=plan['maximum_logical_bytes'],
        input_hashes=inputs,command=command)
    put(r/f'experiments/{name}_registration.json',reg)
    with (r/f'experiments/{name}.md').open('x') as f:f.write(f'# {name} — registered metadata-only stage\n\n{rid}; outcome pending; no scientific quality or test access.\n')
    print(json.dumps({k:v for k,v in reg.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':main()
