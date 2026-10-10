"""Run actual CLI A3/A6/A7 -> features -> retrieval -> profile -> export.

All metrics are explicitly synthetic engineering evidence.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from signcoord.provenance import atomic_json,execution_stamp,sha256_file


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); a=p.parse_args()
    root=Path(a.output).resolve(); root.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(PROJECT/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    records=[]
    def run(name,args):
        command=[sys.executable]+args
        result=subprocess.run(command,env=env,text=True,capture_output=True)
        (root/(name+'.log')).write_text(result.stdout+result.stderr)
        records.append({'step':name,'command':command,'returncode':result.returncode,'log_sha256':sha256_file(root/(name+'.log'))})
        if result.returncode: raise RuntimeError(f'{name} failed; see {root/(name+".log")}')
        lines=[line for line in result.stdout.splitlines() if line.startswith('{')]
        return json.loads(lines[-1]) if lines else {}
    run('generate',[str(PROJECT/'tools/generate_synthetic_fixture.py'),'--output',str(root/'fixture')])
    config=str(root/'fixture/fixture_smoke.yaml')
    def cli(name,*args): return run(name,['-m','signcoord.cli',*args,'--config',config])
    cli('doctor','doctor','--output',str(root/'doctor.json'))
    cli('audit_data','audit-data','--output',str(root/'data_overlap_report.json'))
    cli('fit_stats','fit-target-stats','--output',str(root/'fixture/stats/target_stats.json'),
        '--phase-output',str(root/'fixture/stats/phase_calibration.json'))
    cli('audit_targets','audit-targets','--output',str(root/'target_audit.json'))
    summaries={}
    for arm in ('A3','A6','A7'):
        cli(arm+'_train','train','--arm',arm,'--set','run_dir='+str(root/arm))
        checkpoint=str(root/arm/'checkpoints/final.pt')
        for split in ('ssl_train','ssl_dev'):
            cli(arm+'_features_'+split,'extract-features','--checkpoint',checkpoint,
                '--manifest',str(root/'fixture/manifests'/(split+'.jsonl')),
                '--output',str(root/arm/(split+'.npz')))
        report=cli(arm+'_retrieval','eval-retrieval','--gallery',str(root/arm/'ssl_train.npz'),
                   '--queries',str(root/arm/'ssl_dev.npz'),'--labels',str(root/'fixture/labels.json'),
                   '--output',str(root/arm/'retrieval.json'))
        summaries[arm]=report['metrics']
        cli(arm+'_profile','profile','--arm',arm,'--warmup-steps','2','--measure-steps','5',
            '--output',str(root/arm/'profile.json'))
    cli('A7_export','export','--checkpoint',str(root/'A7/checkpoints/final.pt'),'--output',str(root/'A7/export'))
    report={**execution_stamp(),'status':'passed','scientific_evidence':False,'data_kind':'synthetic',
            'arms':['A3','A6','A7'],'optimizer_updates_per_arm':8,'steps':records,'synthetic_metrics':summaries,
            'profiles_are_short_cpu_checks_not_stable_gpu_estimates':True,'root':str(root)}
    atomic_json(root/'smoke_report.json',report)
    atomic_json(PROJECT/'docs/evidence/smoke_report.json',report)
    print(json.dumps({'status':'passed','root':str(root),'scientific_evidence':False}))


if __name__=='__main__': main()
