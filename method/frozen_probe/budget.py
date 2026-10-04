"""Count every owned terminal GPU attempt before using the granted device."""
import json
import math
from pathlib import Path

from cache import file_sha

TERMINAL = {'completed', 'failed', 'interrupted', 'invalidated'}


def budget_from_statuses(statuses, hypothesis='H001', project_limit_seconds=14400,
                         hypothesis_limit_seconds=3600):
    if min(project_limit_seconds, hypothesis_limit_seconds) <= 0:
        raise ValueError('Invalid registered GPU budget')
    seen = set()
    project = hypothesis_used = 0.0
    counted = []
    for status in statuses:
        run_id = status['run_id']
        if run_id in seen: raise ValueError('Duplicate runtime run ID')
        seen.add(run_id)
        hours = status.get('GPU_hours')
        if (not isinstance(hours, (int, float)) or isinstance(hours, bool)
                or not math.isfinite(hours) or hours < 0):
            raise ValueError('Missing/invalid GPU spend; audit before launch')
        is_gpu = hours > 0 or status.get('GPU_allocated', False)
        if not is_gpu: continue
        if status['status'] not in TERMINAL:
            raise RuntimeError('Unreconciled GPU attempt; inspect process and cost before launching')
        if not status.get('H_id'):
            raise ValueError('GPU attempt missing hypothesis binding')
        seconds = float(hours) * 3600
        project += seconds
        if status['H_id'] == hypothesis: hypothesis_used += seconds
        counted.append(dict(run_id=run_id, H_id=status['H_id'], status=status['status'], GPU_seconds=seconds))
    remaining = min(project_limit_seconds - project, hypothesis_limit_seconds - hypothesis_used)
    if remaining <= 0: raise RuntimeError('Registered GPU budget exhausted')
    return dict(hypothesis=hypothesis, project_limit_seconds=project_limit_seconds,
                hypothesis_limit_seconds=hypothesis_limit_seconds, project_used_seconds=project,
                hypothesis_used_seconds=hypothesis_used, remaining_seconds=remaining, counted_attempts=counted)


def inspect_budget(runtime_roots, hypothesis='H001', project_limit_seconds=14400,
                   hypothesis_limit_seconds=3600):
    statuses, hashes = [], {}
    for root in runtime_roots:
        root = Path(root)
        for path in sorted(root.glob('*/status.json')):
            statuses.append(json.loads(path.read_text()))
            hashes[str(path.resolve())] = file_sha(path)
    result = budget_from_statuses(statuses, hypothesis, project_limit_seconds, hypothesis_limit_seconds)
    result['status_sha256'] = hashes
    return result
