"""Freeze P004 held-out configuration only after the fixed real dev pilot passes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from budget import inspect_budget
from cache import CacheReader, file_sha
from replication import check_gate, quality_report
from replication_test import validate_test_config
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'output-dir', 'software-run', 'pilot-run', 'dev-gate'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import numpy as np
        test_status = json.loads((args.software_run / 'status.json').read_text())
        tests = json.loads((args.software_run / 'report.json').read_text())
        if (test_status['status'] != 'completed' or not tests['passed'] or tests['tests_run'] != 2
                or test_status['output_sha256'] != file_sha(args.software_run / 'report.json')
                or tests['code_hashes'] != run.code_hashes):
            raise ValueError('Held-out software gate requires current source/config oracles')
        status = json.loads((args.pilot_run / 'status.json').read_text())
        if status['status'] != 'completed' or status['protocol'] != 'P004_v1':
            raise ValueError('Real fixed-policy dev pilot required')
        pilot_config = REPO / config['pilot_config']
        check_gate(args.dev_gate, pilot_config, REPO)
        if (status['config_sha256'] != file_sha(pilot_config)
                or file_sha(args.pilot_run / 'config.json') != file_sha(pilot_config)):
            raise ValueError('Fixed pilot config differs from its dev registration')
        for name, digest in status['output_hashes'].items():
            if file_sha(args.pilot_run / name) != digest:
                raise ValueError('Fixed pilot output changed')
        quality = json.loads((args.pilot_run / 'quality.json').read_text())
        schema = json.loads((args.pilot_run / 'schema.json').read_text())
        with np.load(args.pilot_run / 'rankings.npz', allow_pickle=False) as archive:
            ranks = {name: archive[name] for name in archive.files}
        reproduced = quality_report(schema, ranks)
        if (not reproduced['quality_screen'] or not quality['continue_to_test_preparation']
                or not quality['window_count_screen']
                or any(quality[k] != reproduced[k] for k in reproduced)):
            raise ValueError('Negative or inconsistent dev pilot; no held-out scaling')
        scientific = [REPO / name for name in config['scientific_configs']]
        configs = [validate_test_config(json.loads(p.read_text())) for p in scientific]
        first = configs[0]
        shared = ('cohort_manifest', 'video_audit', 'cache_identity', 'cache_manifest', 'train_thresholds')
        if any(any(c[k] != first[k] for k in shared) for c in configs):
            raise ValueError('Held-out preparation/prediction/analysis inputs disagree')
        cohort = json.loads((REPO / first['cohort_manifest']).read_text())
        audit = json.loads((REPO / first['video_audit']).read_text())
        CacheReader(first['cache_manifest'], first['cache_identity'], cohort, audit)
        if file_sha(first['cache_manifest']) != quality['cache_manifest_sha256']:
            raise ValueError('Held-out gallery cache differs from valid pilot')
        budget = inspect_budget([REPO / p for p in first['budget_runtime_roots']], 'H002', 14400, 7200)
        bound = {str((ROOT / name).relative_to(REPO)): digest for name, digest in run.code_hashes.items()}
        for name in config['bound_documents']:
            bound[name] = file_sha(REPO / name)
        for folder in (args.pilot_run, args.software_run):
            for name in ('status.json', 'config.json'):
                bound[str((folder / name).resolve().relative_to(REPO))] = file_sha(folder / name)
        for name in status['output_hashes']:
            bound[str((args.pilot_run / name).resolve().relative_to(REPO))] = file_sha(args.pilot_run / name)
        gate = dict(verdict='FROZEN_REPLICATION_BEFORE_SELECTED_TEST_ACCESS', protocol='P004_v1', H_id='H002',
                    frozen_utc=datetime.now(timezone.utc).isoformat(), fixed_dev_pilot_passed=True,
                    selected_test_access='none for COHORT002 before this gate; old P003 outcomes/global CSV metadata known',
                    config_hashes={str(p.relative_to(REPO)): file_sha(p) for p in scientific}, file_hashes=bound,
                    cache_manifest_sha256=file_sha(first['cache_manifest']), budget_before_freeze=budget,
                    test_GPU_launch_requires_exact_source_count_cost_gate=True, new_policy_trials=0,
                    original_P003_unchanged=True, role_review=config['role_review'])
        run.save('gate.json', gate)
        run.status['output_sha256'] = file_sha(run.output / 'gate.json')


if __name__ == '__main__':
    main()
