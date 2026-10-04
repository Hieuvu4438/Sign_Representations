"""Software-only GPU accounting fixtures; never initialize CUDA."""
import hashlib
import io
import json
import signal
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from bounded_gpu_context import BoundedGPUContext,gpu_allowance,validate_gpu_allocation
from bounded_artifacts import verify_store

FIXTURE_ROOT=None
FIXTURE_RECORDS=[]


def config(stage):
    sources=[Path('method/frozen_probe')/name for name in ('bounded_gpu_context.py','bounded_artifacts.py','cache.py','temporal.py')]
    snapshots={'source_snapshot/'+p.name:str(p) for p in sources}
    caps={k:Path(v).stat().st_size for k,v in snapshots.items()}
    caps.update({'config.json':16384,'environment.json':32768,'status.json':32768,'error.txt':16384,'combined_log.txt':16384,'result.json':1024})
    c=dict(protocol='GPU_CONTEXT_SOFTWARE_FIXTURE',H_id='SOFTWARE_SIMULATION',seed=1701,timeout_seconds=30,
           GPU_allocation_seconds=20,program_ceiling_seconds=100,hypothesis_ceiling_seconds=20,
           source_snapshots=snapshots,input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
           artifact_file_caps=caps,artifact_manifest_cap=32768,artifact_ceiling_bytes=1<<20,simulation_only=True)
    path=FIXTURE_ROOT/(stage+'_config.json');path.write_text(json.dumps(c));return path


class GPUContextTests(unittest.TestCase):
    def test_allowance_limits_and_rejections(self):
        c=dict(H_id='H006',timeout_seconds=30,GPU_allocation_seconds=20,program_ceiling_seconds=100,hypothesis_ceiling_seconds=20)
        b=dict(hypothesis='H006',project_limit_seconds=100,hypothesis_limit_seconds=20,remaining_seconds=15.)
        self.assertEqual(gpu_allowance(c,b,0),15.)
        self.assertEqual(gpu_allowance(c,b,25),5.)
        for allocation in (0,-1,True,31):
            with self.assertRaises(ValueError):validate_gpu_allocation(dict(c,GPU_allocation_seconds=allocation))
        with self.assertRaises(ValueError):gpu_allowance(c,dict(b,hypothesis='other'),0)
        with self.assertRaises(ValueError):gpu_allowance(c,dict(b,remaining_seconds=float('nan')),0)
        with self.assertRaises(ValueError):gpu_allowance(c,dict(b,project_limit_seconds=101),0)
        with self.assertRaises(RuntimeError):gpu_allowance(c,b,29)

    def test_completed_simulated_charge_and_unique_start(self):
        path=config('completed');root=FIXTURE_ROOT/'completed'
        before=signal.getsignal(signal.SIGTERM)
        with redirect_stdout(io.StringIO()):
            with BoundedGPUContext(root,'simulated_completed',path,1701) as run:
                b=dict(hypothesis='SOFTWARE_SIMULATION',project_limit_seconds=100,hypothesis_limit_seconds=20,remaining_seconds=20.)
                with patch('bounded_gpu_context.time.monotonic',return_value=run.started+1):run.start_gpu(b)
                with self.assertRaises(ValueError):run.start_gpu(b)
                run.save('result.json',dict(software_simulation=True))
                # Fix only final accounting clock; no actual GPU work performed.
                run.gpu_started=run.started-2
        s=json.loads((root/'status.json').read_text());self.assertEqual(s['status'],'completed')
        self.assertTrue(s['GPU_allocated']);self.assertGreater(s['GPU_hours']*3600,2)
        self.assertEqual(signal.getsignal(signal.SIGTERM),before)
        record=verify_store(root);record.update(path=str(root),scope='software simulation; no hardware GPU spend')
        FIXTURE_RECORDS.append(record)

    def test_failure_preserves_charge_and_byte_limits(self):
        path=config('failed');root=FIXTURE_ROOT/'failed'
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                with BoundedGPUContext(root,'simulated_failed',path,1701) as run:
                    b=dict(hypothesis='SOFTWARE_SIMULATION',project_limit_seconds=100,hypothesis_limit_seconds=20,remaining_seconds=20.)
                    run.start_gpu(b);run.gpu_started=run.started-3
                    run.store.write_bytes('result.json',b'x'*1025)
        s=json.loads((root/'status.json').read_text());self.assertEqual(s['status'],'failed')
        self.assertGreater(s['GPU_hours']*3600,3);self.assertIn('result.json',s['missing_artifacts'])
        self.assertFalse((root/'artifact_manifest.json').exists())
        c=json.loads(path.read_text());total=0
        for p in root.rglob('*'):
            if p.is_file():
                n=p.relative_to(root).as_posix();self.assertIn(n,c['artifact_file_caps']);self.assertLessEqual(p.stat().st_size,c['artifact_file_caps'][n]);total+=p.stat().st_size
        FIXTURE_RECORDS.append(dict(path=str(root),logical_bytes=total,passed_partial_caps=True,scope='software simulation; no hardware GPU spend'))
