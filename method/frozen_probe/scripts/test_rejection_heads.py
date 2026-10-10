import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext


def main():
    p = argparse.ArgumentParser()
    for name in ('config', 'run-id', 'output-dir'): p.add_argument('--' + name, required=True)
    p.add_argument('--seed', type=int, required=True); a = p.parse_args()
    with BoundedCPUContext(a.output_dir, a.run_id, a.config, a.seed) as run:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
        from test_rejection_heads import run_tests
        run.save('head_tests.json', run_tests(run.config['reference_source']))


if __name__ == '__main__': main()
