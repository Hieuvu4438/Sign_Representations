"""Frozen bounded ablation matrix; resource gate depends on timings, never scores."""
import json,subprocess,time,shutil
from pathlib import Path
import yaml
from signrel.utils import atomic_json,sha256,fingerprint,read_jsonl,code_hash
from signrel.config import load_config,ROOT
from signrel.budget import remaining
from signrel.phase_state import PhaseState
core='.venv-core/bin/python';perception='.venv-perception/bin/python'
while not Path('results/pilot_results.json').exists():time.sleep(2)
pilot=json.loads(Path('results/pilot_results.json').read_text())
if pilot['status']!='complete':raise RuntimeError('Incomplete pilot; matrix cannot start')
code=json.loads(Path('artifacts/code_freeze_pilot.json').read_text())['code_sha256']
if code_hash(ROOT)!=code:raise RuntimeError('Core code changed after pilot freeze')
seconds=[]
for v in ('a1','a4'):
 manifest=json.loads(Path(f'runs/pilot_{v}_seed17/run_manifest.json').read_text())
 if manifest['status']!='complete' or manifest['successful_updates']!=1000:raise RuntimeError('Pilot not stable/completed')
 events=read_jsonl(f'runs/pilot_{v}_seed17/metrics.jsonl')
 seconds.extend([manifest['elapsed_seconds']/1000,(events[-1]['elapsed_seconds']-events[-101]['elapsed_seconds'])/100])
extraction=json.loads(Path('data/cache/eval_cache_manifest.extraction.json').read_text())
per_video=sum(x['seconds'] for x in extraction['profiles'])/len(extraction['profiles'])
rule=json.loads(Path('artifacts/matrix_budget_rule.json').read_text())
per_update=max(seconds)
test_reserve=1.2*80*per_video+60
available_gpu=remaining('gpu_seconds');available_wall=remaining('wall_training_seconds')
horizon=next((h for h in rule['candidate_fixed_horizons_descending'] if 9*h*per_update*1.2+test_reserve<available_gpu and 9*h*per_update*1.2<available_wall),None)
if horizon is None:
 atomic_json('artifacts/matrix_plan.json',{'status':'incomplete','reason':'No nine-row horizon fits unchanged global budget','available_gpu_seconds':available_gpu,'seconds_per_update':per_update,'test_reserve_seconds':test_reserve});raise RuntimeError('Measured matrix allocation does not fit')
rows=[]
for variant in ('a1','a4','a2','a3','a6'):
 for seed in ([17,29,43] if variant in ('a1','a4') else [17]):
  name=f'matrix{horizon}_{variant}_seed{seed}';path=f'configs/{name}.yaml'
  cfg={'extends':f'pilot_{variant}.yaml','experiment':{'name':f'matrix{horizon}_{variant}','seed':seed,'output_dir':f'runs/{name}'},'train':{'max_updates':horizon},'budget':{'phase':'confirmatory','max_gpu_hours':.3,'max_wall_minutes':18}}
  Path(path).write_text(yaml.safe_dump(cfg,sort_keys=False));resolved=load_config(path).values
  rows.append({'variant':variant.upper(),'seed':seed,'config':path,'config_sha256':sha256(path),'resolved_config_sha256':fingerprint(resolved),'planned_updates':horizon,'run_dir':f'runs/{name}','eval_dir':f'results/{name}_test','state':'planned'})
plan={'status':'allocated_before_locked_test','rule_sha256':sha256('artifacts/matrix_budget_rule.json'),'rows':rows,'horizon':horizon,'seconds_per_update_conservative':per_update,'expected_training_seconds_with20percent_reserve':9*horizon*per_update*1.2,'test_cache_reserve_seconds':test_reserve,'remaining_gpu_seconds_before_matrix':available_gpu,'source_sha256':code,'source_lock_sha256':sha256('artifacts/sources.lock.json'),'ssl_cache_manifest_sha256':sha256('data/cache/cache_manifest.jsonl'),'protocol_sha256':sha256('artifacts/protocol.json'),'selection':'last completed fixed-budget checkpoint; no test tuning','global_budget_unchanged':True}
atomic_json('artifacts/matrix_plan.json',plan)
print(json.dumps({'matrix_allocated':horizon,'expected_training_seconds':plan['expected_training_seconds_with20percent_reserve'],'remaining_gpu_seconds':available_gpu}),flush=True)
# Training/config freeze is recorded before creating any test config.
atomic_json('artifacts/pretest_training_freeze.json',{'protocol_sha256':sha256('artifacts/protocol.json'),'train_configs':{r['config']:r['config_sha256'] for r in rows},'source_sha256':code,'checkpoint_rule':plan['selection']})
configs=[]
eval_rows=rows+[{'variant':'A0','seed':0,'run_dir':'A0','eval_dir':'results/matrix_a0_test'}]
for row in eval_rows:
 cfg=yaml.safe_load(Path('configs/eval_dev_a0.yaml').read_text())
 cfg.update(stage='test',encoder_checkpoint='A0' if row['variant']=='A0' else row['run_dir']+'/checkpoints/last.pt',query_ids='data/splits/locked_test_query_ids.json',output_dir=row['eval_dir'])
 path='configs/test_'+Path(row['eval_dir']).name+'.yaml';Path(path).write_text(yaml.safe_dump(cfg,sort_keys=False));row['eval_config']=path;configs.append(path)
atomic_json('artifacts/confirmatory_freeze.json',{'protocol_sha256':sha256('artifacts/protocol.json'),'eval_config_sha256':[sha256(p) for p in configs],'training_freeze_sha256':sha256('artifacts/pretest_training_freeze.json'),'matrix_plan_sha256':sha256('artifacts/matrix_plan.json'),'locked_test_scores_observed':False,'selection':plan['selection']})
agg={'schema_version':1,'runs':[{k:r[k] for k in ('variant','seed','run_dir','eval_dir')} for r in eval_rows],'contrasts':[['A1','A4'],['A2','A4'],['A3','A4'],['A6','A4']],'require_complete':False,'output':'results/confirmatory_results.csv'}
Path('configs/aggregate_confirmatory.yaml').write_text(yaml.safe_dump(agg,sort_keys=False))
report=yaml.safe_load(Path('configs/report.yaml').read_text());report.update(aggregate='results/confirmatory_results.json',output='results/confirmatory_report.md')
report['limitations'].append(f'Matrix uses a separate fixed{horizon}-update horizon selected by a timing-only budget rule; it is not the1000-update pilot. Mechanism controls A2/A3/A6 each have one seed; H1 has three paired seeds if complete.')
Path('configs/report_confirmatory.yaml').write_text(yaml.safe_dump(report,sort_keys=False))

def run(command,log):
 with Path(log).open('w') as file:subprocess.run(command,stdout=file,stderr=subprocess.STDOUT,check=True)
 print(json.dumps({'completed':command,'log':log}),flush=True)
# Finish all planned training before the first locked-test score is read.
for row in rows:
 if remaining('gpu_seconds')<test_reserve:
  row['state']='incomplete_budget';continue
 row['state']='running';atomic_json('artifacts/matrix_status.json',{'rows':rows})
 run([core,'-m','signrel','train','--config',row['config']],f'artifacts/train_{Path(row["run_dir"]).name}.log')
 manifest=json.loads(Path(row['run_dir']+'/run_manifest.json').read_text())
 row['state']=manifest['status'];row['successful_updates']=manifest['successful_updates']
 row['checkpoint_sha256']=manifest['checkpoint_sha256']
 atomic_json('artifacts/matrix_status.json',{'rows':rows})
# Snapshot initial120-file cache index for reproducible dev exports before extending it.
shutil.copy2('data/cache/eval_cache_manifest.jsonl','artifacts/eval_dev_cache_manifest.lock.jsonl')
run([perception,'-m','signrel','extract-cache','--config','configs/extract_eval_test.yaml'],'artifacts/extract_eval_test.log')
run([core,'-m','signrel','audit-cache','--config','configs/extract_eval_test.yaml','--output','artifacts/test_cache_audit.json'],'artifacts/audit_test_cache.log')
for row in eval_rows:
 if row['variant']!='A0' and row['state']!='complete':continue
 run([core,'-m','signrel','export-embeddings','--config',row['eval_config']],f'artifacts/export_{Path(row["eval_dir"]).name}.log')
 run([core,'-m','signrel','eval-retrieval','--config',row['eval_config']],f'artifacts/eval_{Path(row["eval_dir"]).name}.log')
 if row['variant']!='A0':
  run([core,'-c',f'from signrel.train.diagnostics import heldout_diagnostics; heldout_diagnostics("{row["run_dir"]}/checkpoints/last.pt","data/cache/ssl_dev_cache_manifest.jsonl","artifacts/diagnostics_{Path(row["run_dir"]).name}.json")'],f'artifacts/diagnostics_{Path(row["run_dir"]).name}.log')
run([core,'-m','signrel','aggregate','--config','configs/aggregate_confirmatory.yaml'],'artifacts/aggregate_confirmatory.log')
run([core,'-m','signrel','build-report','--config','configs/report_confirmatory.yaml'],'artifacts/build_confirmatory_report.log')
atomic_json('artifacts/matrix_status.json',{'status':'complete' if all(r['state']=='complete' for r in rows) else 'incomplete','rows':rows,'test_evaluated':True,'config_freeze_sha256':sha256('artifacts/confirmatory_freeze.json')})
