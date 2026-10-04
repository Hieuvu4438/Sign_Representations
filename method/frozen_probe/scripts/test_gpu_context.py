"""Registered CPU execution of bounded GPU context software fixtures."""
import argparse
import importlib.util
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from resources import scratch_preflight


def main():
    p=argparse.ArgumentParser()
    for n in ('config','run-id','output-dir'):p.add_argument('--'+n,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with BoundedCPUContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        resource=scratch_preflight('/dev/shm',1<<20)
        path=Path('method/frozen_probe/tests/test_bounded_gpu_context.py')
        spec=importlib.util.spec_from_file_location('gpu_context_software_tests',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        root=Path('/dev/shm')/('signrep_gpu_context_'+a.run_id);root.mkdir(exist_ok=False);module.FIXTURE_ROOT=root
        result=unittest.TextTestRunner(stream=sys.stderr,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
        total=sum(p.stat().st_size for p in root.rglob('*') if p.is_file());assert total<=1<<20
        run.save('gpu_context_tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
            fixtures=module.FIXTURE_RECORDS,fixture_root=str(root),fixture_retained_logical_bytes=total,resource=resource,GPU_hours=0,
            scope='CPU software simulated accounting/lifecycle, CUDA not imported or initialized; child fixtures outside program runtime roots, their artificial GPU_hours must never enter hardware ledger'))
        if not result.wasSuccessful():raise ValueError('Bounded GPU context software checks failed')


if __name__=='__main__':main()
