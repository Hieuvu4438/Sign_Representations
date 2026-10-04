"""Issue a prospective dev-only P004 gate after real source audits and CPU oracles."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from budget import inspect_budget
from cache import file_sha
from replication import validate_cohort
from resources import gpu_preflight, scratch_preflight
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'output-dir', 'software-run'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        status = json.loads((args.software_run / 'status.json').read_text())
        report = json.loads((args.software_run / 'report.json').read_text())
        if (status['status'] != 'completed' or not report['passed'] or report['tests_run'] != 4
                or status['output_sha256'] != file_sha(args.software_run / 'report.json')
                or report['code_hashes'] != run.code_hashes):
            raise ValueError('Missing/current replication integration oracles')
        original = json.loads((REPO / config['original_scientific_gate']).read_text())
        for name, digest in original['file_hashes'].items():
            if file_sha(REPO / name) != digest:
                raise ValueError('Immutable original P003 binding changed')
        paths = [REPO / p for p in config['scientific_configs']]
        cache_config = json.loads(paths[0].read_text())
        cohort = json.loads((REPO / cache_config['cohort_manifest']).read_text())
        audit = json.loads((REPO / cache_config['video_audit']).read_text())
        validate_cohort(cohort, audit, cache_config['expected'], cache_config['cache_identity']['contract'])
        budget = inspect_budget([REPO / p for p in cache_config['budget_runtime_roots']], 'H002', 14400, 7200)
        if config['estimate_full_dev_first_seconds'] > budget['remaining_seconds']:
            raise RuntimeError('Prospective dev-first projection exceeds remaining allowance')
        scratch = scratch_preflight('/dev/shm', config['planned_cache_bytes'], 1 << 30)
        gpu = gpu_preflight(8192, 100)
        bound = {str((ROOT / name).relative_to(REPO)): digest for name, digest in run.code_hashes.items()}
        for name in config['bound_documents']:
            bound[name] = file_sha(REPO / name)
        for name in ('status.json', 'report.json', 'tests.txt'):
            bound[str((args.software_run / name).resolve().relative_to(REPO))] = file_sha(args.software_run / name)
        gate = dict(verdict='REGISTERED_FIXED_POLICY_DEV_PILOT', protocol='P004_v1', H_id='H002',
                    registered_utc=datetime.now(timezone.utc).isoformat(),
                    config_hashes={str(p.relative_to(REPO)): file_sha(p) for p in paths},
                    file_hashes=bound, original_P003_unchanged=True,
                    new_dev_outcomes_seen=False, new_selected_test_access='none',
                    earlier_global_test_metadata_and_P003_outcomes_known=True,
                    new_policy_trials=0, budget_before_launch=budget,
                    resources=dict(GPU=gpu, scratch=scratch),
                    estimate_full_dev_first_seconds=config['estimate_full_dev_first_seconds'],
                    projection_is_not_test_launch_gate=True,
                    role_review=config['role_review'])
        run.save('gate.json', gate)
        run.status['output_sha256'] = file_sha(run.output / 'gate.json')


if __name__ == '__main__':
    main()
