"""Prelaunch collector fixtures or the metadata-fixed396window cost measurement."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from register_axis_stage import dependencies,put,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('tests','cost'),required=True);a=p.parse_args()
    root,r=Path('method/frozen_probe'),Path('method/research_lab/records');tests=a.stage=='tests'
    name='MSFEATTEST001' if tests else 'MSFEATCOST001';protocol='MSFEATTEST001_collector_pooling_oracles' if tests else 'MSFEATCOST001_dense_pipeline_cost'
    entry=root/'scripts/measure_msasl_feature_cost.py';sources=dependencies(entry)+[root/'scripts/register_msfeature_cost.py',root/'scripts/register_axis_stage.py']
    if tests:sources+=dependencies(root/'tests/test_msasl_feature_inputs.py')
    else:sources+=dependencies(Path('method/stream_reliability/gpu_accounting.py'))
    snapshots={'source_snapshot/'+(str(x.relative_to(root)) if x.is_relative_to(root) else 'accounting/'+x.name):str(x) for x in sorted(set(sources))}
    inputs={x:sha(x) for x in snapshots.values()}
    for x in (r/'decisions/D069.md',r/'protocols'/(protocol+'.md')):inputs[str(x)]=sha(x)
    caps={x:Path(source).stat().st_size for x,source in snapshots.items()}
    caps.update({'config.json':131072,'environment.json':32768,'status.json':131072,'error.txt':65536,'combined_log.txt':65536})
    c=dict(protocol=protocol,seed=1701,timeout_seconds=60 if tests else 180,GPU_allocation_seconds=0 if tests else 90,HPO_trials=0,
           source_snapshots=snapshots,input_hashes=inputs,artifact_file_caps=caps,artifact_manifest_cap=131072,artifact_ceiling_bytes=2<<20 if tests else 4<<20)
    if tests:c['software_only']=True;caps['feature_input_tests.json']=32768
    else:
        plan_path=r/'inventory/MSFEATURE_COST_METADATA_PLAN.json';plan=json.loads(plan_path.read_text());inputs[str(plan_path)]=sha(plan_path)
        for id in ('MSFEATTEST001','MSESTENV001','MSDICT001','MSFEATUREGPU002'):
            reg=json.loads((r/f'experiments/{id}_registration.json').read_text());parent=Path(reg['output_dir'])
            assert json.loads((parent/'status.json').read_text())['status']=='completed'
            if id=='MSFEATTEST001':assert json.loads((parent/'feature_input_tests.json').read_text())['passed']
            for x in parent.rglob('*'):
                if x.is_file():inputs[str(x)]=sha(x)
            if id=='MSDICT001':dictionary=parent;dc=json.loads((parent/'config.json').read_text())
            if id=='MSFEATUREGPU002':gc=json.loads((parent/'config.json').read_text())
        manifest=dictionary/'extraction_manifest.json';assert sha(manifest)==plan['extraction_manifest_sha256']
        rows=json.loads(manifest.read_text());byid={x['sample_id']:x for x in rows};ids=[x['sample_id'] for x in plan['samples']]
        assert sum(byid[x]['complete_stride2_windows'] for x in ids)==plan['expected_encoder_windows']==396
        cohort=Path(dc['cohort_runtime']);cc=json.loads((cohort/'config.json').read_text());parents=Path(cc['native_runtime'])/'source_parents.json'
        inputs[str(parents)]=sha(parents)
        for id in ids:inputs[byid[id]['raw_path']]=byid[id]['source_sha256']
        inputs[cc['ffprobe']]=sha(cc['ffprobe'])
        for x,h in gc['input_hashes'].items():
            if x.startswith(('third_party/','checkpoints/')):inputs[x]=h
        for k in ('checkpoint','H_id','program_ceiling_seconds','hypothesis_ceiling_seconds','runtime_roots','cost_errata','granted_uuid','min_free_mib','max_gpu_utilization'):c[k]=gc[k]
        for binding in c['cost_errata']:inputs[binding['path']]=binding['sha256']
        for x in (r/'decisions/D068.md',r/'inventory/MSDICTIONARY_BINDING_VERIFICATION.json'):inputs[str(x)]=sha(x)
        c.update(extraction_manifest=str(manifest),source_parents=str(parents),sample_ids=ids,ffprobe=cc['ffprobe'],
                 max_native_frames=327,max_decoded_frames=4096,expected_encoder_windows=396)
        sys.path.insert(0,str(root));from bounded_artifacts import array_spec
        c['array_specs']={n:array_spec(shape,dtype) for n,shape,dtype in [('features.npy',(396,768),'<f4'),('activity.npy',(396,2),'<f4'),('pooled_vectors.npy',(7,768),'<f8')]}
        caps.update({n:s['file_bytes'] for n,s in c['array_specs'].items()});caps.update({'feature_cost_records.json':524288,'feature_cost_summary.json':131072})
    sys.path.insert(0,str(root));from bounded_artifacts import inspect_file_plan
    plan=inspect_file_plan({**caps,'artifact_manifest.json':c['artifact_manifest_cap']},c['artifact_ceiling_bytes'],('status.json','error.txt'));assert plan['passed']
    cfg=root/'configs'/(name+'.json');put(cfg,c);h=sha(cfg);rid=f'{name}_20261004_{h[:8]}_s1701'
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),run_id=rid,config=str(cfg),config_sha256=h,output_dir=str(root/'runtime'/rid),
             CPU_or_job_limit_seconds=c['timeout_seconds'],GPU_seconds=c['GPU_allocation_seconds'],durable_bytes_ceiling=c['artifact_ceiling_bytes'],
             maximum_planned_logical_bytes=plan['maximum_logical_bytes'],input_hashes=inputs,
             command=[str(Path('.venv-signrep/bin/python').absolute()),str(entry),'--config',str(cfg),'--run-id',rid,'--output-dir',str(root/'runtime'/rid),'--seed','1701'])
    put(r/f'experiments/{name}_registration.json',reg)
    with (r/f'experiments/{name}.md').open('x') as f:f.write(f'# {name} — registered extraction prerequisite\n\n{rid}; outcome pending.\n')
    print(json.dumps({k:v for k,v in reg.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':main()
