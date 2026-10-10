"""Promote tested validation/inference fixes only after frozen experiment terminal state."""
import json
from pathlib import Path
from signrel.config import ROOT
from signrel.utils import code_hash, atomic_json, sha256

status=json.loads(Path('artifacts/matrix_status.json').read_text())
assert status.get('test_evaluated') is True, 'Keep main scientific source frozen until test evaluation finishes'
before=json.loads(Path('artifacts/code_freeze_pilot.json').read_text())['code_sha256']
assert code_hash(ROOT)==before
staging=Path('artifacts/engineering_validation')
proof=json.loads(Path('artifacts/staged_inference_verification.json').read_text())
assert proof['status']=='passed' and proof['frozen_training_source_sha256']==before
assert proof['staged_source_sha256']==code_hash(staging)
assert '42 passed' in Path('artifacts/pytest_staged.log').read_text()
paths=['src/signrel/data/provenance.py','src/signrel/data/windows.py','src/signrel/models/encoder.py',
       'src/signrel/train/loop.py','src/signrel/train/diagnostics.py','src/signrel/eval/embed.py',
       'src/signrel/pipelines.py','tests/test_inference_provenance.py']
changes=[]
for relative in paths:
    target=Path(relative);source=staging/relative
    previous=sha256(target) if target.exists() else None
    target.write_bytes(source.read_bytes())
    changes.append({'path':relative,'before_sha256':previous,'after_sha256':sha256(target)})
assert code_hash(ROOT)==proof['staged_source_sha256']
atomic_json('artifacts/postmatrix_validation_patch.json',{
    'status':'applied_after_all_budget_permitted_training_and_locked_test',
    'scientific_source_sha256':before,'current_package_source_sha256':code_hash(ROOT),
    'changes':changes,'tests_receipt_sha256':sha256('artifacts/pytest_staged.log'),
    'real_inference_proof_sha256':sha256('artifacts/staged_inference_verification.json'),
    'scope':'Stricter preprocessing/cache provenance, portable inference bindings and no decoder computation during encoder inference. No losses, optimizer, masks, target values or pooling mathematics changed. Historical source archives remain authoritative for exact scientific-run resume.'})
print('Verified validation/inference changes promoted after terminal test state.')
