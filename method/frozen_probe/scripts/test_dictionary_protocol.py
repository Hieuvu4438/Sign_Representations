"""Registered synthetic metric/sampling prerequisite; never target galleries."""
import argparse
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext


def main():
    parser=argparse.ArgumentParser()
    for key in ('config','run-id','output-dir'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
    with BoundedCPUContext(args.output_dir,args.run_id,args.config,args.seed) as run:
        result=unittest.TextTestRunner(stream=sys.stderr,verbosity=2).run(
            unittest.defaultTestLoader.discover('method/frozen_probe/tests',pattern='test_dictionary_protocol.py'))
        run.save('dictionary_tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
                 errors=len(result.errors),failures=len(result.failures),GPU_hours=0,new_encoder_calls=0,
                 target_gallery_selection='none; synthetic fixtures only',scope='Adapted micro/macro ranking and one-shot sampling guards; no official benchmark reproduction or quality measurement.'))
        if not result.wasSuccessful():raise ValueError('Dictionary protocol oracles failed')


if __name__=='__main__':main()
