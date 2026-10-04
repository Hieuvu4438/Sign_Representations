"""Effective resource ledger, preserving original statuses and bound cost errata."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'method/frozen_probe'))
from budget import budget_from_statuses
from cache import file_sha


def inspect_effective_budget(config):
    expected = {REPO / 'method' / direction / 'runtime' for direction in
                ('frozen_probe', 'research_lab', 'stream_reliability')}
    roots = {REPO / name for name in config['runtime_roots']}
    if roots != expected:
        raise ValueError('All three program namespaces required')
    statuses = {}
    hashes = {}
    paths = {}
    for root in sorted(roots):
        for path in sorted(root.glob('*/status.json')):
            status = json.loads(path.read_text())
            if status['run_id'] in statuses:
                raise ValueError('Duplicate actual run ID')
            statuses[status['run_id']] = status
            paths[status['run_id']] = str(path.relative_to(REPO))
            hashes[str(path.resolve())] = file_sha(path)
    applied = []
    seen = set()
    for binding in config['cost_errata']:
        path = REPO / binding['path']
        if file_sha(path) != binding['sha256']:
            raise ValueError('Registered GPU accounting erratum changed')
        correction = json.loads(path.read_text())
        run_id = correction['charged_run_id']
        if run_id in seen or run_id not in statuses:
            raise ValueError('Ambiguous accounting correction')
        seen.add(run_id)
        original = statuses[run_id]
        status_path = REPO / paths[run_id]
        if (paths[run_id] != correction['original_status_path'] or
                file_sha(status_path) != correction['original_status_sha256'] or
                original['H_id'] != correction['H_id'] or
                original['GPU_hours'] != correction['original_GPU_hours']):
            raise ValueError('Original resource status identity differs')
        charge = correction['conservative_GPU_seconds']
        if charge != original['wall_seconds'] or charge < original['GPU_hours'] * 3600:
            raise ValueError('Conservative whole-run charge required')
        effective = dict(original, GPU_hours=charge / 3600, GPU_allocated=True)
        statuses[run_id] = effective
        applied.append(dict(run_id=run_id, conservative_GPU_seconds=charge,
                            accounting_path=str(path), accounting_sha256=file_sha(path)))
    result = budget_from_statuses(list(statuses.values()), config['H_id'],
                                  config['program_ceiling_seconds'], config['hypothesis_ceiling_seconds'])
    result.update(status_sha256=hashes, accounting_errata=applied)
    return result
