"""One reviewed CPU continuation: complete cue extraction, then fixed readouts."""
import fcntl
import json
import os
import subprocess
import time

from _common import ROOT
from run_ncslgr_branch_ablation import CONFIG, FEATURES, OUTPUT, identities
from signrepr.io import sha256, write_json


def main():
    folder=ROOT/'state/ncslgr_branch_ablation_v2';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.pipeline.lock').open('a') as worker:
        fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'status.json').exists() or OUTPUT.exists():
            raise ValueError('Preserve prior continuation; no automatic retry')
        protocol=json.loads(CONFIG.read_text())
        audit_path=ROOT/'reports/ncslgr_branch_ablation_smoke_v2_audit.json'
        audit=json.loads(audit_path.read_text())
        smoke=json.loads((FEATURES/'status.json').read_text())
        if identities()!=protocol['identity_sha256'] or audit['status']!='PASS_TRAIN_ONLY_EIGHT_CONDITION_NUMERICAL_SMOKE' or audit['config_sha256']!=sha256(CONFIG) or smoke['status']!='PARTIAL_TRAIN_SMOKE' or smoke['success']!=1:
            raise ValueError('Reviewed numerical training smoke and unchanged protocol required')
        if sha256(FEATURES/'extraction_report.json')!=audit['report_sha256']:
            raise ValueError('Reviewed smoke report changed')
        started=time.perf_counter()
        state=dict(status='RUNNING_CPU_INTERVENTION_EXTRACTION',pid=os.getpid(),device='cpu',gpu_workers=0,
                   config_sha256=sha256(CONFIG),script_sha256=sha256(__file__),reviewed_smoke_sha256=sha256(audit_path),
                   source_split_of_smoke='train',scientific_policy_unchanged=True)
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
        try:
            stages=[('RUNNING_CPU_INTERVENTION_EXTRACTION',ROOT/'.venv-shubert-native/bin/python',
                     ['-X','faulthandler','scripts/extract_ncslgr_branch_ablation.py','--resume'],'logs/ncslgr_branch_ablation_extraction_v2.log'),
                    ('RUNNING_CPU_FIXED_READOUTS',ROOT/'.venv-signrep/bin/python',
                     ['scripts/run_ncslgr_branch_ablation.py'],'logs/ncslgr_branch_ablation_readouts_v2.log')]
            for phase,python,args,log in stages:
                if phase=='RUNNING_CPU_FIXED_READOUTS':
                    feature=json.loads((FEATURES/'extraction_report.json').read_text())
                    if feature['status']!='PASS_SAME_SOURCE_CUE_ABLATION_FEATURES' or feature['success']!=len(protocol['eligible_ids']):
                        raise ValueError('Incomplete extraction cannot trigger readout fitting')
                state.update(status=phase,command=[str(python),*args],log=log,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
                write_json(folder/'status.json',state)
                with (ROOT/log).open('w') as stream:
                    result=subprocess.run([str(python),*args],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                if result.returncode:
                    raise ValueError(f'Child {phase} exited {result.returncode}; see {log}. No automatic retry.')
            summary=json.loads((OUTPUT/'summary.json').read_text())
            if summary['status']!='PASS_EXPLORATORY_SAME_SOURCE_CUE_ABLATION' or summary['config_sha256']!=state['config_sha256']:
                raise ValueError('Terminal scientific output status or identity differs')
            state.update(status='PASS_EXPLORATORY_CUE_ABLATION_PIPELINE',elapsed_seconds=time.perf_counter()-started,
                         summary_sha256=sha256(OUTPUT/'summary.json'))
            write_json(folder/'status.json',state)
        except Exception as error:
            state.update(status='FAIL',failure_reason=type(error).__name__+': '+str(error),elapsed_seconds=time.perf_counter()-started)
            write_json(folder/'status.json',state);raise


if __name__=='__main__':main()
