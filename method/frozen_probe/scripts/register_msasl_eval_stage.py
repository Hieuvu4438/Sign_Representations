"""Exclusive preregistration before synthetic tests, first scoring and replay."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from register_axis_stage import dependencies, put, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('tests','measure','verify'),required=True);a=p.parse_args()
    root,r=Path('method/frozen_probe'),Path('method/research_lab/records')
    name={'tests':'MSEVALTEST001','measure':'MSEVAL001','verify':'MSEVALVERIFY001'}[a.stage]
    entry=root/'scripts'/dict(tests='test_msasl_geometry_eval.py',measure='evaluate_msasl_geometry.py',verify='verify_msasl_geometry_eval.py')[a.stage]
    sources=dependencies(entry)+[root/'scripts/register_msasl_eval_stage.py',root/'scripts/register_axis_stage.py']
    if a.stage=='tests':sources+=dependencies(root/'tests/test_msasl_geometry_eval.py')
    snapshots={'source_snapshot/'+str(x.relative_to(root)):str(x) for x in sorted(set(sources))}
    inputs={x:sha(x) for x in snapshots.values()}
    protocol={'tests':'MSEVALTEST001_manual_geometry_oracles','measure':'H006_v3_MSASL_quality_v1','verify':'MSEVALVERIFY001_original_space'}[a.stage]
    for x in (r/'decisions/D071.md',r/'protocols'/(protocol+'.md')):inputs[str(x)]=sha(x)
    caps={x:Path(source).stat().st_size for x,source in snapshots.items()}
    caps.update({'config.json':262144,'environment.json':32768,'status.json':262144,'error.txt':65536,'combined_log.txt':65536})
    c=dict(protocol=protocol,seed=1701,GPU_allocation_seconds=0,timeout_seconds=60 if a.stage=='tests' else 300,
        HPO_trials=0,input_hashes=inputs,source_snapshots=snapshots,artifact_file_caps=caps,
        artifact_manifest_cap=262144,artifact_ceiling_bytes=20<<20 if a.stage=='measure' else 2<<20)
    if a.stage=='tests':c['software_only']=True;caps['geometry_eval_tests.json']=32768
    else:
        test=json.loads((r/'experiments/MSEVALTEST001_registration.json').read_text());tp=Path(test['output_dir'])
        assert json.loads((tp/'status.json').read_text())['status']=='completed'
        assert json.loads((tp/'geometry_eval_tests.json').read_text())['passed']
        for x in tp.rglob('*'):
            if x.is_file():inputs[str(x)]=sha(x)
        for key in ('MSFULLFEAT001','MSDICT001','AXISTRANSFER001'):
            reg=json.loads((r/f'experiments/{key}_registration.json').read_text());parent=Path(reg['output_dir'])
            assert json.loads((parent/'status.json').read_text())['status']=='completed'
            for x in parent.rglob('*'):
                if x.is_file():inputs[str(x)]=sha(x)
            c[dict(MSFULLFEAT001='feature_runtime',MSDICT001='dictionary_runtime',AXISTRANSFER001='base_runtime')[key]]=str(parent)
        proof=r/'inventory/MSFULL_FEATURE_BINDING_VERIFICATION.json';assert json.loads(proof.read_text())['passed'];inputs[str(proof)]=sha(proof)
        c.update(expected_clips=1037,expected_queries=748,expected_classes=100,expected_signers=19,
            support_seeds=[1701,1702,1703],modes=['RR','TR','RT','TT','raw','centered'],dimension=768,
            primary_gain_floor=.005,bootstrap_replicates=2000,numerical_score_atol=1e-10,row_metric_atol=1e-12,
            interval_atol=1e-12,workspace_plan_bytes=256<<20)
        if a.stage=='measure':
            sys.path.insert(0,str(root));from bounded_artifacts import array_spec
            c['array_specs']={n:array_spec(shape,dtype) for n,shape,dtype in (
                ('scores.npy',(13464,100),'<f8'),('ranks.npy',(13464,100),'<i2'),('row_metrics.npy',(13464,6),'<f8'))}
            for n,spec in c['array_specs'].items():caps[n]=spec['file_bytes']
            caps['shared_metadata.json']=262144;caps['geometry_eval.json']=2<<20
        else:
            reg=json.loads((r/'experiments/MSEVAL001_registration.json').read_text());parent=Path(reg['output_dir'])
            assert json.loads((parent/'status.json').read_text())['status']=='completed';c['measured_runtime']=str(parent)
            for x in parent.rglob('*'):
                if x.is_file():inputs[str(x)]=sha(x)
            caps['geometry_eval_verification.json']=131072
    sys.path.insert(0,str(root));from bounded_artifacts import inspect_file_plan
    plan=inspect_file_plan({**caps,'artifact_manifest.json':c['artifact_manifest_cap']},c['artifact_ceiling_bytes'],('status.json','error.txt'))
    assert plan['passed'],'Complete artifact plan exceeds ceiling'
    cfg=root/'configs'/(name+'.json');put(cfg,c);h=sha(cfg);rid=f'{name}_20261004_{h[:8]}_s1701'
    command=[str(Path('/home/haipd/miniconda3/bin/python')),str(entry),'--config',str(cfg),'--run-id',rid,'--output-dir',str(root/'runtime'/rid),'--seed','1701']
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),run_id=rid,config=str(cfg),config_sha256=h,
        output_dir=str(root/'runtime'/rid),CPU_limit_seconds=c['timeout_seconds'],GPU_seconds=0,
        durable_bytes_ceiling=c['artifact_ceiling_bytes'],maximum_planned_logical_bytes=plan['maximum_logical_bytes'],
        input_hashes=inputs,command=command)
    put(r/f'experiments/{name}_registration.json',reg)
    with (r/f'experiments/{name}.md').open('x') as f:f.write(f'# {name} — registered fixed MS-ASL geometry stage\n\n{rid}; outcome pending.\n')
    print(json.dumps({k:v for k,v in reg.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':main()
