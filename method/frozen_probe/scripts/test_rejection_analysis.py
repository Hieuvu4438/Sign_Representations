"""Registered conditional-statistics/compact-curve software prerequisite."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext


def main():
    p = argparse.ArgumentParser()
    for k in ('config', 'run-id', 'output-dir'): p.add_argument('--' + k, required=True)
    p.add_argument('--seed', type=int, required=True); a = p.parse_args()
    with BoundedCPUContext(a.output_dir, a.run_id, a.config, a.seed) as run:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
        from test_rejection_analysis import run_tests
        run.save('rejection_analysis_tests.json', run_tests())


if __name__ == '__main__': main()
