"""Fresh timing gate after actual cache cost, before any locked-test score."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from signrel.utils import atomic_json, sha256, code_hash, read_jsonl
from signrel.config import ROOT
from signrel.budget import remaining

core='.venv-core/bin/python'
intervention=json.loads(Path('artifacts/fresh_timing_intervention.json').read_text())
child=intervention['active_extraction_pid']
while Path(f'/proc/{child}/stat').exists() and Path(f'/proc/{child}/stat').read_text().split()[2]!='Z':
    command=Path(f'/proc/{child}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert '-m signrel extract-cache --config configs/extract_eval_test.yaml' in command
    time.sleep(2)
cache_report=json.loads(Path('data/cache/eval_cache_manifest.extraction.json').read_text())
assert cache_report['status']=='complete' and cache_report['attempted']==80 and cache_report['retained']==200
assert not list(Path('results').glob('matrix*_test/metrics.json')), 'No test score may inform resource allocation'
parent=intervention['own_supervisor_pid']
command=Path(f'/proc/{parent}/cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
assert command=='.venv-core/bin/python artifacts/matrix_continue.py'
os.kill(parent,signal.SIGKILL)
intervention.update(status='replaced_after_existing_extraction_child_completed',
                    actual_test_extraction_report_sha256=sha256('data/cache/eval_cache_manifest.extraction.json'),
                    test_scores_observed=False,transition_epoch_seconds=time.time())
atomic_json('artifacts/fresh_timing_intervention.json',intervention)

freeze=json.loads(Path('artifacts/pretest_training_freeze.json').read_text())
assert code_hash(ROOT)==freeze['source_sha256']
for path,content in freeze['train_configs'].items():assert sha256(path)==content
assert sha256('artifacts/protocol.json')==freeze['protocol_sha256']
rows=json.loads(Path('artifacts/matrix_status.json').read_text())['rows']
assert all(r['state']=='complete' for r in rows[:6])
reference=json.loads(Path('artifacts/profile_final.json').read_text())['seconds_per_update']
decisions=[]

def record():
    atomic_json('artifacts/matrix_status.json',{'rows':rows})
    atomic_json('artifacts/matrix_fresh_budget_decisions.json',{
        'rule':'Latest completed row tail100 elapsed/update and full-loop measured profile maximum, times1.2; actual test extraction already charged;60-second global GPU headroom; no score-based decisions',
        'original_freeze_sha256':sha256('artifacts/pretest_training_freeze.json'),
        'test_cache_report_sha256':sha256('data/cache/eval_cache_manifest.extraction.json'),
        'no_test_metrics_before_training':True,'global_caps_unchanged':True,'decisions':decisions})

def run(command,log):
    with Path(log).open('w') as file:subprocess.run(command,stdout=file,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({'completed':command,'log':log}),flush=True)

for row in rows[6:]:
    assert row['state']=='incomplete_budget'
    last=next(r for r in reversed(rows) if r['state']=='complete')
    events=read_jsonl(Path(last['run_dir'])/'metrics.jsonl')
    assert len(events)>=101
    fresh=(events[-1]['elapsed_seconds']-events[-101]['elapsed_seconds'])/100
    predicted=row['planned_updates']*max(reference,fresh)*1.2
    available=remaining('gpu_seconds')
    allowed=predicted+60<available and predicted<remaining('wall_training_seconds')
    decisions.append({'variant':row['variant'],'seed':row['seed'],'previous_state':row['state'],
                      'timing_run':last['run_dir'],'tail100_seconds_per_update':fresh,
                      'reference_profile_seconds_per_update':reference,
                      'predicted_training_seconds_with20percent_reserve':predicted,
                      'actual_remaining_gpu_seconds':available,'headroom_seconds':60,
                      'allowed':allowed,'epoch_seconds':time.time()})
    if not allowed:
        record();print(json.dumps({'still_incomplete_budget':[row['variant'],row['seed']]}),flush=True);continue
    row['previous_resource_gate']='incomplete_budget_before_fresh_profile_and_actual_test_cache_cost'
    row['state']='running';record()
    run([core,'-m','signrel','train','--config',row['config']],f'artifacts/train_{Path(row["run_dir"]).name}.log')
    manifest=json.loads(Path(row['run_dir'],'run_manifest.json').read_text())
    row.update(state=manifest['status'],successful_updates=manifest['successful_updates'],checkpoint_sha256=manifest['checkpoint_sha256'])
    record()

run([core,'-m','signrel','audit-cache','--config','configs/extract_eval_test.yaml','--output','artifacts/test_cache_audit.json'],'artifacts/audit_test_cache.log')
eval_rows=rows+[{'variant':'A0','seed':0,'run_dir':'A0','eval_dir':'results/matrix_a0_test','eval_config':'configs/test_matrix_a0_test.yaml'}]
for row in eval_rows:
    if row['variant']!='A0' and row['state']!='complete':continue
    run([core,'-m','signrel','export-embeddings','--config',row['eval_config']],f'artifacts/export_{Path(row["eval_dir"]).name}.log')
    run([core,'-m','signrel','eval-retrieval','--config',row['eval_config']],f'artifacts/eval_{Path(row["eval_dir"]).name}.log')
    if row['variant']!='A0':
        run([core,'-c','from signrel.train.diagnostics import heldout_diagnostics; '
             f'heldout_diagnostics("{row["run_dir"]}/checkpoints/last.pt",'
             '"data/cache/ssl_dev_cache_manifest.jsonl",'
             f'"artifacts/diagnostics_{Path(row["run_dir"]).name}.json")'],f'artifacts/diagnostics_{Path(row["run_dir"]).name}.log')
run([core,'-m','signrel','aggregate','--config','configs/aggregate_confirmatory.yaml'],'artifacts/aggregate_confirmatory.log')
run([core,'-m','signrel','build-report','--config','configs/report_confirmatory.yaml'],'artifacts/build_confirmatory_report.log')
atomic_json('artifacts/matrix_status.json',{'status':'complete' if all(r['state']=='complete' for r in rows) else 'incomplete',
    'rows':rows,'test_evaluated':True,'config_freeze_sha256':sha256('artifacts/confirmatory_freeze.json')})
