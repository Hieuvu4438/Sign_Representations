"""Run only registered synthetic geometry/evaluator fixtures."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext


def main():
    p=argparse.ArgumentParser()
    for key in ('config','run-id','output-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with BoundedCPUContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        assert run.config['software_only']
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
        from test_msasl_geometry_eval import run_tests
        run.save('geometry_eval_tests.json',run_tests())


if __name__=='__main__':main()
