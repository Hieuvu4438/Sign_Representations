"""Exclusive run records and bounded active-session execution."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import time
import traceback

from cache import atomic_json, file_sha

ROOT = Path(__file__).resolve().parent


class RunContext:
    def __init__(self, output, run_id, config_path, seed):
        self.output = Path(output)
        self.run_id = run_id
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text())
        if self.config['seed'] != seed:
            raise ValueError('Seed/config mismatch')
        self.output.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.status = dict(run_id=run_id, protocol=self.config['protocol'], seed=seed,
                           status='running', GPU_hours=0, pid=os.getpid(), host=platform.node(),
                           start_utc=datetime.now(timezone.utc).isoformat(),
                           command=[sys.executable] + sys.argv, config_sha256=file_sha(config_path))
        if 'H_id' in self.config: self.status['H_id'] = self.config['H_id']
        self.save('status.json', self.status)
        self.code_hashes = {}
        try:
            self._snapshot()
        except BaseException as error:
            self.status.update(status='failed', exit_code=1, error=repr(error),
                               end_utc=datetime.now(timezone.utc).isoformat(),
                               wall_seconds=time.monotonic() - self.started)
            self.save('status.json', self.status)
            (self.output / 'error.txt').write_text(traceback.format_exc())
            raise

    def _snapshot(self):
        for source in sorted(ROOT.rglob('*.py')):
            relative = source.relative_to(ROOT)
            if 'runtime' in relative.parts or '__pycache__' in relative.parts:
                continue
            target = self.output / 'source_snapshot' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            self.code_hashes[str(relative)] = file_sha(source)
        shutil.copyfile(self.config_path, self.output / 'config.json')
        self.save('environment.json', dict(python=sys.version, executable=sys.executable,
                                          CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                          threads={k: os.environ.get(k) for k in (
                                              'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                                          code_hashes=self.code_hashes))
        self.save('status.json', self.status)

    def start_gpu(self, budget):
        if hasattr(self, 'gpu_started') or 'H_id' not in self.status:
            raise ValueError('GPU attempt needs unique start and hypothesis binding')
        remaining = min(budget['remaining_seconds'],
                        self.config['timeout_seconds'] - (time.monotonic() - self.started))
        if not remaining > 1: raise RuntimeError('Insufficient registered GPU/job allowance')
        self.gpu_started = time.monotonic()
        self.status.update(GPU_allocated=True, gpu_start_utc=datetime.now(timezone.utc).isoformat(),
                           budget_before_launch=budget, gpu_deadline_seconds=min(self.config['timeout_seconds'], int(remaining)))
        self.save('status.json', self.status)
        signal.alarm(self.status['gpu_deadline_seconds'])

    def save(self, name, value):
        atomic_json(self.output / name, value)

    def __enter__(self):
        def timeout(*_): raise TimeoutError('Registered run time budget exhausted')
        def interrupted(*_): raise KeyboardInterrupt('Run interrupted')
        signal.signal(signal.SIGALRM, timeout)
        signal.signal(signal.SIGTERM, interrupted)
        signal.alarm(self.config['timeout_seconds'])
        return self

    def __exit__(self, kind, error, tb):
        signal.alarm(0)
        if hasattr(self, 'gpu_started'):
            self.status['GPU_hours'] = (time.monotonic() - self.gpu_started) / 3600
        self.status.update(end_utc=datetime.now(timezone.utc).isoformat(),
                           wall_seconds=time.monotonic() - self.started)
        if kind is None:
            self.status.update(status='completed', exit_code=0)
        else:
            self.status.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                               exit_code=130 if isinstance(error, KeyboardInterrupt) else 1, error=repr(error))
            (self.output / 'error.txt').write_text(''.join(traceback.format_exception(kind, error, tb)))
        self.save('status.json', self.status)
        print(json.dumps(self.status), flush=True)
        return False
