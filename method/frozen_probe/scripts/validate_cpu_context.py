"""Registered whole-context integration checks; no scientific measurements."""
import argparse
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext


def main():
    parser = argparse.ArgumentParser()
    for option in ('config', 'run-id', 'output-dir'):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    with BoundedCPUContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        scratch = Path(run.config['fixture_directory'])
        scratch.mkdir(exist_ok=False)
        os.environ['SIGNREP_CONTEXT_FIXTURES'] = str(scratch)
        suite = unittest.defaultTestLoader.discover(
            str(Path(__file__).resolve().parents[1] / 'tests'), pattern='test_bounded_cpu_context.py')
        result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
        fixture_bytes = sum(p.lstat().st_size for p in scratch.rglob('*') if p.is_file() or p.is_symlink())
        if fixture_bytes > run.config['fixture_bytes_ceiling']:
            raise ValueError('CPU-context fixtures exceed registered cap')
        run.save('context_validation.json', dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
            errors=len(result.errors), failures=len(result.failures), retained_fixture_logical_bytes=fixture_bytes,
            fixture_directory=str(scratch), fixture_ceiling_bytes=run.config['fixture_bytes_ceiling'],
            GPU_hours=0, new_encoder_calls=0, scientific_measurements=0,
            scope='CPU-only whole-context software integration; all parent artifacts governed by declared store; no old output conversion or new scientific evidence'))
        if not result.wasSuccessful():
            raise ValueError('CPU-context integration checks failed')


if __name__ == '__main__':
    main()
