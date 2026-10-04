"""Preregister complete-only feature shards or the full frozen target extraction."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from register_axis_stage import dependencies,put,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('tests','extract'),required=True);a=p.parse_args()
    root,r=Path('method/frozen_probe'),Path('method/research_lab/records');tests=a.stage=='tests'
    name='MSFULLTEST001' if tests else 'MSFULLFEAT001';protocol='MSFULLTEST001_reuse_shards' if tests else 'MSFULLFEAT001_all_dictionary_inputs'
    entry=root/'scripts/extract_msasl_dictionary.py';sources=dependencies(entry)+[root/'scripts/register_msfullfeature_stage.py',root/'scripts/register_axis_stage.py']
    if tests:sources+=dependencies(root/'tests/test_msasl_feature_shards.py')
    else:sources+=dependencies(Path('method/stream_reliability/gpu_accounting.py'))
    snapshots={'source_snapshot/'+(str(x.relative_to(root)) if x.is_relative_to(root) else 'accounting/'+x.name):str(x) for x in sorted(set(sources))}
    inputs={x:sha(x) for x in snapshots.values()}
    for x in (r/'decisions/D070.md',r/'protocols'/(protocol+'.md')):inputs[str(x)]=sha(x)
    caps={x:Path(source).stat().st_size for x,source in snapshots.items()}
    caps.update({'config.json':262144,'environment.json':32768,'status.json':131072,'error.txt':65536,'combined_log.txt':262144})
    c=dict(protocol=protocol,seed=1701,timeout_seconds=60 if tests else 6000,GPU_allocation_seconds=0 if tests else 5000,HPO_trials=0,
           source_snapshots=snapshots,input_hashes=inputs,artifact_file_caps=caps,artifact_manifest_cap=262144,artifact_ceiling_bytes=2<<20 if tests else 20<<20)
    if tests:c['software_only']=True;caps['feature_shard_tests.json']=32768
    else:
        for x in (r/'protocols/H006_v3_MSASL_quality_v1.md',r/'inventory/MSFEATURE_COST_BINDING_VERIFICATION.json',r/'inventory/MSFULL_EXTRACTION_RESOURCE_PLAN.json'):
            inputs[str(x)]=sha(x)
        assert json.loads((r/'inventory/MSFEATURE_COST_BINDING_VERIFICATION.json').read_text())['passed']
        for id in ('MSFULLTEST001','MSFEATCOST001','MSDICT001'):
            reg=json.loads((r/f'experiments/{id}_registration.json').read_text());parent=Path(reg['output_dir'])
            assert json.loads((parent/'status.json').read_text())['status']=='completed'
            if id=='MSFULLTEST001':assert json.loads((parent/'feature_shard_tests.json').read_text())['passed']
            for x in parent.rglob('*'):
                if x.is_file():inputs[str(x)]=sha(x)
            if id=='MSFEATCOST001':cost=parent;old=json.loads((parent/'config.json').read_text())
            if id=='MSDICT001':dictionary=parent
        manifest=dictionary/'extraction_manifest.json';items=json.loads(manifest.read_text());assert len(items)==1037
        for item in items:inputs[item['raw_path']]=item['source_sha256']
        for x,h in old['input_hashes'].items():
            if x.startswith(('third_party/','checkpoints/')):inputs[x]=h
        inputs[old['ffprobe']]=sha(old['ffprobe']);inputs[old['source_parents']]=sha(old['source_parents'])
        for k in ('checkpoint','H_id','program_ceiling_seconds','runtime_roots','cost_errata','granted_uuid','min_free_mib','max_gpu_utilization'):c[k]=old[k]
        for binding in c['cost_errata']:inputs[binding['path']]=binding['sha256']
        c.update(hypothesis_ceiling_seconds=5400,cost_runtime=str(cost),extraction_manifest=str(manifest),source_parents=old['source_parents'],
                 ffprobe=old['ffprobe'],max_native_frames=327,max_decoded_frames=4096,expected_clips=1037,sample_ids=[x['sample_id'] for x in items])
        sys.path.insert(0,str(root));from msasl_feature_shards import shard_plan
        c['shards']=shard_plan(len(items),25)
        for s in c['shards']:caps[s['array_name']]=s['spec']['file_bytes'];caps[s['records_name']]=131072
        caps['extraction_summary.json']=131072
    sys.path.insert(0,str(root));from bounded_artifacts import inspect_file_plan
    plan=inspect_file_plan({**caps,'artifact_manifest.json':c['artifact_manifest_cap']},c['artifact_ceiling_bytes'],('status.json','error.txt'));assert plan['passed']
    cfg=root/'configs'/(name+'.json');put(cfg,c);h=sha(cfg);rid=f'{name}_20261004_{h[:8]}_s1701'
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),run_id=rid,config=str(cfg),config_sha256=h,output_dir=str(root/'runtime'/rid),
             CPU_or_job_limit_seconds=c['timeout_seconds'],GPU_seconds=c['GPU_allocation_seconds'],durable_bytes_ceiling=c['artifact_ceiling_bytes'],
             maximum_planned_logical_bytes=plan['maximum_logical_bytes'],input_hashes=inputs,
             command=[str(Path('.venv-signrep/bin/python').absolute()),str(entry),'--config',str(cfg),'--run-id',rid,'--output-dir',str(root/'runtime'/rid),'--seed','1701'])
    put(r/f'experiments/{name}_registration.json',reg)
    with (r/f'experiments/{name}.md').open('x') as f:f.write(f'# {name} — registered full feature prerequisite\n\n{rid}; outcome pending.\n')
    print(json.dumps({k:v for k,v in reg.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':main()
