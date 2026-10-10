"""Build/install the current wheel in owned storage, leaving the venv unchanged."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from signcoord.provenance import atomic_json,execution_stamp,sha256_file


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    before=execution_stamp()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    def run(name,command,**kwargs):
        result=subprocess.run(command,env=env,capture_output=True,text=True,**kwargs)
        (root/(name+'.log')).write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(name+' failed; inspect log')
    run('wheel',[sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation',str(PROJECT),
                 '--wheel-dir',str(root/'wheels')])
    wheels=list((root/'wheels').glob('*.whl'))
    if len(wheels)!=1:raise ValueError('exactly one package wheel expected')
    target=root/'installed'
    run('install',[sys.executable,'-m','pip','install','--no-deps','--target',str(target),str(wheels[0])])
    env['PYTHONPATH']=str(target)
    with tempfile.TemporaryDirectory() as cwd:
        code='''import json,sys,signcoord
from pathlib import Path
assert Path(sys.argv[1]) in Path(signcoord.__file__).resolve().parents
from signcoord.provenance import execution_stamp
from signcoord.engine.distributed import TrainingContext
from signcoord.data.adapters import select_candidates
print(json.dumps(execution_stamp()))
'''
        result=subprocess.run([sys.executable,'-c',code,str(target)],env=env,cwd=cwd,capture_output=True,text=True)
        (root/'import.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('installed package import failed')
        installed=json.loads(result.stdout)
        result=subprocess.run([sys.executable,'-m','signcoord.cli','doctor','--config',str(target/'signcoord/base_config.yaml')],
                              env=env,cwd=cwd,capture_output=True,text=True)
        (root/'doctor.log').write_text(result.stdout+result.stderr)
        doctor=json.loads(result.stdout)
    if installed['implementation_hash']!=before['implementation_hash']:
        raise ValueError('wheel source differs from implementation')
    if execution_stamp()['implementation_hash']!=before['implementation_hash']:
        raise ValueError('source changed during packaging')
    if result.returncode!=1 or doctor['status']!='inputs_missing':
        raise ValueError('base doctor must disclose unresolved inputs and return nonzero')
    report={**before,'status':'passed','version':'recipe_bound_v8','wheel':str(wheels[0]),
            'wheel_sha256':sha256_file(wheels[0]),'isolated_target_install':True,
            'shared_environment_mutated':False,'source_unchanged_during_execution':True,
            'installed_source_hash_matches':True,'doctor_exit_code':result.returncode,
            'doctor_expected_missing_inputs':doctor}
    atomic_json(root/'package_report.json',report)
    atomic_json(PROJECT/'docs/evidence/package_report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='doctor_expected_missing_inputs'}))


if __name__=='__main__':main()
