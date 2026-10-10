"""Continue frozen matrix with a timing-only reserve guard; never regenerate locks."""
import json
import subprocess
import time
import shutil
from pathlib import Path

import yaml
from signrel.utils import atomic_json, sha256, code_hash
from signrel.config import ROOT
from signrel.budget import remaining

core = '.venv-core/bin/python'
perception = '.venv-perception/bin/python'
plan = json.loads(Path('artifacts/matrix_plan.json').read_text())
freeze = json.loads(Path('artifacts/pretest_training_freeze.json').read_text())
assert code_hash(ROOT) == freeze['source_sha256'] == plan['source_sha256']
assert sha256('artifacts/protocol.json') == freeze['protocol_sha256']
for path, content in freeze['train_configs'].items():
    assert sha256(path) == content, path
status = json.loads(Path('artifacts/matrix_status.json').read_text())
rows = status['rows']
reserve = plan['test_cache_reserve_seconds']
decisions = []


def run(command, log):
    with Path(log).open('w') as file:
        subprocess.run(command, stdout=file, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({'completed': command, 'log': log}), flush=True)


def record():
    atomic_json('artifacts/matrix_status.json', {'rows': rows})
    atomic_json('artifacts/matrix_runtime_budget_decisions.json', {
        'rule': 'same frozen updates/configs; latest three completed same-horizon timing maxima times1.2 plus original test reserve; no score-based decisions',
        'original_plan_sha256': sha256('artifacts/matrix_plan.json'),
        'global_caps_unchanged': True, 'decisions': decisions})


# The paused original supervisor was terminated only after its current child
# wrote a terminal manifest. Reconcile that row without another training call.
for row in rows:
    if row['state'] == 'running':
        manifest = json.loads(Path(row['run_dir'], 'run_manifest.json').read_text())
        assert manifest['status'] in ('complete', 'budget_stopped'), manifest['status']
        row.update(state=manifest['status'], successful_updates=manifest['successful_updates'],
                   checkpoint_sha256=manifest['checkpoint_sha256'])
record()

for row in rows:
    if row['state'] != 'planned':
        continue
    completed = [json.loads(Path(r['run_dir'], 'run_manifest.json').read_text())
                 for r in rows if r['state'] == 'complete']
    recent = completed[-3:]
    per_update = max([plan['seconds_per_update_conservative']] +
                     [m['elapsed_seconds'] / m['successful_updates'] for m in recent])
    predicted = row['planned_updates'] * per_update * 1.2
    available_gpu = remaining('gpu_seconds')
    available_wall = remaining('wall_training_seconds')
    allowed = predicted + reserve < available_gpu and predicted < available_wall
    decisions.append({'variant': row['variant'], 'seed': row['seed'],
                      'remaining_gpu_seconds': available_gpu,
                      'remaining_training_wall_seconds': available_wall,
                      'predicted_training_seconds_with20percent_reserve': predicted,
                      'locked_test_extraction_reserve_seconds': reserve,
                      'allowed': allowed, 'epoch_seconds': time.time()})
    if not allowed:
        row['state'] = 'incomplete_budget'
        record()
        print(json.dumps({'skipped_budget': [row['variant'], row['seed']]}), flush=True)
        continue
    row['state'] = 'running'
    record()
    run([core, '-m', 'signrel', 'train', '--config', row['config']],
        f'artifacts/train_{Path(row["run_dir"]).name}.log')
    manifest = json.loads(Path(row['run_dir'], 'run_manifest.json').read_text())
    row.update(state=manifest['status'], successful_updates=manifest['successful_updates'],
               checkpoint_sha256=manifest['checkpoint_sha256'])
    record()

# Preserve the exact original dev cache index before adding test caches.
snapshot = Path('artifacts/eval_dev_cache_manifest.lock.jsonl')
if not snapshot.exists():
    shutil.copy2('data/cache/eval_cache_manifest.jsonl', snapshot)
run([perception, '-m', 'signrel', 'extract-cache', '--config', 'configs/extract_eval_test.yaml'],
    'artifacts/extract_eval_test.log')
run([core, '-m', 'signrel', 'audit-cache', '--config', 'configs/extract_eval_test.yaml',
     '--output', 'artifacts/test_cache_audit.json'], 'artifacts/audit_test_cache.log')
eval_rows = rows + [{'variant': 'A0', 'seed': 0, 'run_dir': 'A0',
                     'eval_dir': 'results/matrix_a0_test',
                     'eval_config': 'configs/test_matrix_a0_test.yaml'}]
for row in eval_rows:
    if row['variant'] != 'A0' and row['state'] != 'complete':
        continue
    run([core, '-m', 'signrel', 'export-embeddings', '--config', row['eval_config']],
        f'artifacts/export_{Path(row["eval_dir"]).name}.log')
    run([core, '-m', 'signrel', 'eval-retrieval', '--config', row['eval_config']],
        f'artifacts/eval_{Path(row["eval_dir"]).name}.log')
    if row['variant'] != 'A0':
        run([core, '-c', 'from signrel.train.diagnostics import heldout_diagnostics; '
             f'heldout_diagnostics("{row["run_dir"]}/checkpoints/last.pt",'
             '"data/cache/ssl_dev_cache_manifest.jsonl",'
             f'"artifacts/diagnostics_{Path(row["run_dir"]).name}.json")'],
            f'artifacts/diagnostics_{Path(row["run_dir"]).name}.log')
run([core, '-m', 'signrel', 'aggregate', '--config', 'configs/aggregate_confirmatory.yaml'],
    'artifacts/aggregate_confirmatory.log')
run([core, '-m', 'signrel', 'build-report', '--config', 'configs/report_confirmatory.yaml'],
    'artifacts/build_confirmatory_report.log')
atomic_json('artifacts/matrix_status.json', {
    'status': 'complete' if all(r['state'] == 'complete' for r in rows) else 'incomplete',
    'rows': rows, 'test_evaluated': True,
    'config_freeze_sha256': sha256('artifacts/confirmatory_freeze.json')})
