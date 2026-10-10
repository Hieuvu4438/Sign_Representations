"""Verify post-documentation record bindings without rerunning a terminal job."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cache import file_sha


def read(path): return json.loads(Path(path).read_text())


def main():
    started = time.monotonic(); root = Path('method/research_lab/records')
    proof_path = root / 'inventory/H007_HEAD_CYCLE_BINDING_VERIFICATION_r1.json'
    proof = read(proof_path); assert proof['passed'] and not proof['scientific_screen_passed']
    bindings = {}; experiment_paths = []
    for key, checked in proof['runs'].items():
        regpath = root / f'experiments/{key}_registration.json'; reg = read(regpath)
        directory = Path(reg['output_dir']); status_path = directory / 'status.json'; status = read(status_path)
        assert file_sha(status_path) == checked['status_sha256']
        assert status['status'] == checked['status'] and status['pid'] == checked['pid']
        assert file_sha(reg['config']) == checked['config_sha256']
        config = read(directory / 'config.json')
        for path, digest in config['input_hashes'].items():
            assert path not in bindings or bindings[path] == digest
            bindings[path] = digest
        experiment_paths.extend([regpath, root / f'experiments/{key}.md'])
        card = (root / f'experiments/{key}.md').read_text()
        assert 'Actual terminal outcome, D090/r34' in card and checked['run_id'] in card
    assert len(bindings) == proof['current_bound_inputs_verified']
    for path, digest in bindings.items(): assert file_sha(path) == digest, path
    records = [root / 'STATE.md', root / 'SESSION_AUDIT_r34.md', root / 'experiments/INDEX.md',
        root / 'claims/LEDGER.md', root / 'paper/H007_EVIDENCE_DOSSIER_r2.md',
        root / 'literature/SEARCH_LOG.md', root / 'literature/matrix.csv', proof_path,
        root / 'inventory/FRDISC_MODEL_SOURCE_CAPTURE_r1.json']
    records.extend(root / f'decisions/D{i:03}.md' for i in range(86, 91))
    records.extend(experiment_paths)
    state = (root / 'STATE.md').read_text(); ledger = (root / 'claims/LEDGER.md').read_text()
    assert state.startswith('# Current research state — D090') and '**Not PAPER_READY' in state
    for i in range(99, 104): assert f'C{i:03}' in ledger
    for i in range(95, 100): assert f'F{i:03}' in ledger
    assert 'Update39/D090' in (root / 'experiments/INDEX.md').read_text()
    import csv
    matrix = list(csv.DictReader((root / 'literature/matrix.csv').open()))
    assert len(matrix) == 50 and len({r['id'] for r in matrix}) == 50
    s48 = next(r for r in matrix if r['id'] == 'S48'); assert 'D090' in s48['follow_up']
    for field in ('program_used_seconds', 'H007_used_seconds'):
        assert str(proof[field]) in state
    result = dict(passed=True, closed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=file_sha(__file__), record_bindings={str(p): file_sha(p) for p in records},
        current_run_input_bindings_verified=len(bindings), terminal_runs=len(proof['runs']),
        scientific_screen_passed=False, decision='D090', phase='DISCOVER', goal_status='active',
        PAPER_READY=False, closure_wall_seconds=time.monotonic() - started,
        scope='Post-documentation current hashes/actual experiment cards/state/claim IDs/matrix. No terminal job rerun or new scientific measurement.')
    payload = (json.dumps(result, indent=2) + '\n').encode(); assert len(payload) <= 65536
    with (root / 'inventory/H007_HEAD_RECORDS_CLOSURE_r1.json').open('xb') as f: f.write(payload)
    print(json.dumps({k: v for k, v in result.items() if k != 'record_bindings'}, indent=2))


if __name__ == '__main__': main()
