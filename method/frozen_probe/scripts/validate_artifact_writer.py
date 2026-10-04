"""Registered software validation; synthetic records only, no scientific rerun."""
import argparse
import io
import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cache import file_sha
from run_context import RunContext


class BoundedTestLog(io.StringIO):
    def write(self, text):
        if len((self.getvalue() + text).encode('utf-8')) > 1024**2:
            raise ValueError('Software-test log exceeds 1MiB')
        return super().write(text)


def main():
    parser = argparse.ArgumentParser()
    for option in ('config', 'run-id', 'output-dir'):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        c = run.config
        for path, digest in c['input_hashes'].items():
            if file_sha(path) != digest:
                raise ValueError('Changed writer-validation input ' + path)
        scratch = Path(c['fixture_directory'])
        scratch.mkdir(exist_ok=False)
        os.environ['SIGNREP_ARTIFACT_FIXTURES'] = str(scratch)
        log = BoundedTestLog()
        suite = unittest.defaultTestLoader.discover(
            str(Path(__file__).resolve().parents[1] / 'tests'), pattern='test_bounded_artifacts.py')
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        (run.output / 'test_log.txt').write_text(log.getvalue())
        from bounded_artifacts import verify_store
        full = scratch / 'test_full_size_shape_plan_and_lossless_covariance_rank_metric_records'
        verified = verify_store(full)
        fixture_bytes = sum(p.lstat().st_size for p in scratch.rglob('*') if p.is_file() or p.is_symlink())
        if fixture_bytes > c['fixture_bytes_ceiling']:
            raise ValueError('Synthetic fixture store exceeds registered cap')
        report = dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
                      failures=len(result.failures), errors=len(result.errors),
                      full_shape_fixture_verification=verified, fixture_directory=str(scratch),
                      retained_fixture_logical_bytes=fixture_bytes, fixture_ceiling_bytes=c['fixture_bytes_ceiling'],
                      fixture_manifest_sha256=file_sha(full / 'artifact_manifest.json'),
                      GPU_hours=0, new_encoder_calls=0, scientific_measurements=0,
                      scope='software serialization/budget/identity checks on synthetic arrays; not actual H006 output conversion, task-quality confirmation or resource repair')
        run.save('writer_validation.json', report)
        run.status['output_hashes'] = {n: file_sha(run.output / n) for n in ('test_log.txt', 'writer_validation.json')}
        if not result.wasSuccessful():
            raise ValueError('Artifact writer software validation failed')


if __name__ == '__main__':
    main()
