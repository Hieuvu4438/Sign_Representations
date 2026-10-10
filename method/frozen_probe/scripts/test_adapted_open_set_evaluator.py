"""Registered adapted evaluator software fixtures before sign quality access."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'run-id', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    with BoundedCPUContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
        from test_open_set_evaluation import run_tests
        run.save('adapted_tests.json', run_tests())


if __name__ == '__main__':
    main()
