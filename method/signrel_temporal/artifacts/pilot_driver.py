"""Sequential local pilot driver; wait for extraction audit, fail on command errors."""
import json,subprocess,time
from pathlib import Path
core=str(Path('.venv-core/bin/python').resolve())
perception=str(Path('.venv-perception/bin/python').resolve())
# Use venv executable paths rather than resolved symlinks, preserving own package.
core='.venv-core/bin/python';perception='.venv-perception/bin/python'
while True:
 report=Path('data/cache/eval_cache_manifest.extraction.json')
 if report.exists():
  r=json.loads(report.read_text())
  if r.get('attempted')==120 and r.get('status')=='complete':break
  if r.get('attempted')==120 and r.get('status')!='complete':raise RuntimeError('Evaluation cache extraction reported errors')
 time.sleep(2)

def run(command,log):
 with Path(log).open('w') as file:subprocess.run(command,stdout=file,stderr=subprocess.STDOUT,check=True)
 print(json.dumps({'completed':command,'log':log}),flush=True)
run([core,'-m','signrel','audit-cache','--config','configs/extract_eval_dev.yaml','--output','artifacts/eval_cache_audit.json'],'artifacts/audit_eval_cache.log')
run([perception,'-c','from signrel.config import load_config; from signrel.data.profile import component_profile; component_profile(load_config("configs/pilot_a4.yaml").values,"artifacts/component_profile.json")'],'artifacts/component_profile.log')
run([core,'-m','signrel','profile','--config','configs/pilot_a4.yaml','--output','artifacts/profile_final.json'],'artifacts/profile_final.log')
run([core,'-m','signrel','export-embeddings','--config','configs/eval_dev_a0.yaml'],'artifacts/export_dev_a0.log')
run([core,'-m','signrel','eval-retrieval','--config','configs/eval_dev_a0.yaml'],'artifacts/eval_dev_a0.log')
for variant in ('a1','a4'):
 run([core,'-m','signrel','train','--config',f'configs/pilot_{variant}.yaml'],f'artifacts/train_pilot_{variant}.log')
 run([core,'-m','signrel','export-embeddings','--config',f'configs/eval_dev_{variant}.yaml'],f'artifacts/export_dev_{variant}.log')
 run([core,'-m','signrel','eval-retrieval','--config',f'configs/eval_dev_{variant}.yaml'],f'artifacts/eval_dev_{variant}.log')
 run([core,'-c',f'from signrel.train.diagnostics import heldout_diagnostics; heldout_diagnostics("runs/pilot_{variant}_seed17/checkpoints/last.pt","data/cache/ssl_dev_cache_manifest.jsonl","artifacts/diagnostics_{variant}.json")'],f'artifacts/diagnostics_{variant}.log')
run([core,'-m','signrel','train','--config','configs/pilot_a4.yaml','--resume','runs/pilot_a4_seed17/checkpoints/last.pt'],'artifacts/pilot_complete_resume.log')
run([core,'-m','signrel','aggregate','--config','configs/aggregate_pilot.yaml'],'artifacts/aggregate_pilot.log')
run([core,'-m','signrel','build-report','--config','configs/report.yaml'],'artifacts/build_report.log')
