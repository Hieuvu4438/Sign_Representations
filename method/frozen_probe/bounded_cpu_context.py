"""CPU-only run context routing every owned artifact through a registered store."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
import traceback

from bounded_artifacts import BoundedArtifactStore, json_bytes, inspect_file_plan
from cache import file_sha


class BoundedLog(io.StringIO):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap
        self.byte_count = 0

    def write(self, value):
        size = len(value.encode('utf-8'))
        if self.byte_count + size > self.cap:
            raise ValueError('Registered combined log bound exhausted')
        self.byte_count += size
        return super().write(value)


class BoundedCPUContext:
    def __init__(self, output, run_id, config_path, seed):
        self.started = time.monotonic()
        self.config_path = Path(config_path)
        config_bytes = self.config_path.read_bytes()
        self.config = c = json.loads(config_bytes)
        if c['seed'] != seed or c['GPU_allocation_seconds'] != 0:
            raise ValueError('CPU run seed/allocation mismatch')
        if type(c['timeout_seconds']) is not int or c['timeout_seconds'] <= 0:
            raise ValueError('Invalid run deadline')
        caps = dict(c['artifact_file_caps'])
        required = {'config.json', 'environment.json', 'status.json', 'error.txt', 'combined_log.txt'}
        if not required <= set(caps):
            raise ValueError('Missing mandatory context file budgets')
        plan = inspect_file_plan({**caps, 'artifact_manifest.json': c['artifact_manifest_cap']},
                                 c['artifact_ceiling_bytes'], ('status.json', 'error.txt'))
        if not plan['passed']:
            raise ValueError('Complete run and status-replacement plan exceeds ceiling')
        for source, digest in c['input_hashes'].items():
            if file_sha(source) != digest:
                raise ValueError('Changed bound run input ' + source)
        sources = {}
        if str(Path(__file__).relative_to(Path.cwd())) not in c['source_snapshots'].values():
            raise ValueError('CPU context source must be snapshotted and bound')
        for target, source in c['source_snapshots'].items():
            if target not in caps or not target.startswith('source_snapshot/') or source not in c['input_hashes']:
                raise ValueError('Snapshot lacks declared path/byte/input-hash binding')
            if Path(source).stat().st_size > caps[target]:
                raise ValueError('Snapshot exceeds registered cap')
            sources[target] = Path(source).read_bytes()
            if hashlib.sha256(sources[target]).hexdigest() != c['input_hashes'][source]:
                raise ValueError('Snapshot changed after input verification')
        self.status = dict(run_id=run_id, protocol=c['protocol'], seed=seed, status='running',
                           pid=os.getpid(), host=platform.node(), GPU_hours=0,
                           start_utc=datetime.now(timezone.utc).isoformat(),
                           command=[sys.executable] + sys.argv,
                           config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                           artifact_ceiling_bytes=c['artifact_ceiling_bytes'])
        environment = dict(python=sys.version, executable=sys.executable,
                           CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                           threads={name: os.environ.get(name) for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                           source_hashes={target: c['input_hashes'][source] for target, source in c['source_snapshots'].items()},
                           packages={name: importlib.metadata.version(name) for name in ('numpy', 'scikit-learn')})
        initial = dict(sources, **{'config.json': config_bytes, 'environment.json': json_bytes(environment),
                                  'status.json': json_bytes(self.status)})
        if any(len(data) > caps[name] for name, data in initial.items()):
            raise ValueError('Initial context artifact exceeds registered cap')
        self.store = BoundedArtifactStore(output, caps, c['artifact_ceiling_bytes'],
                                          ('status.json', 'error.txt'), c['artifact_manifest_cap'])
        self.output = self.store.root
        for name, data in initial.items():
            self.store.write_bytes(name, data)
        self.log = BoundedLog(caps['combined_log.txt'])

    def save(self, name, value):
        self.store.write_json(name, value)

    def save_array(self, name, array, spec):
        self.store.write_array(name, array, spec)

    def __enter__(self):
        self.previous_alarm = signal.getitimer(signal.ITIMER_REAL)
        self.previous_handlers = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM)}
        self.entered = time.monotonic()
        def timeout(*_):
            raise TimeoutError('Registered CPU run deadline exhausted')
        def interrupted(*_):
            raise KeyboardInterrupt('CPU run interrupted')
        signal.signal(signal.SIGALRM, timeout)
        signal.signal(signal.SIGTERM, interrupted)
        remaining = self.config['timeout_seconds'] - (time.monotonic() - self.started)
        signal.setitimer(signal.ITIMER_REAL, max(remaining, 0.000001))
        self.redirects = ExitStack()
        self.redirects.enter_context(redirect_stdout(self.log))
        self.redirects.enter_context(redirect_stderr(self.log))
        return self

    def __exit__(self, kind, error, tb):
        self.redirects.close()
        signal.setitimer(signal.ITIMER_REAL, 0)
        for s, handler in self.previous_handlers.items():
            signal.signal(s, handler)
        if self.previous_alarm[0] > 0:
            signal.setitimer(signal.ITIMER_REAL,
                             max(0.000001, self.previous_alarm[0] - (time.monotonic() - self.entered)),
                             self.previous_alarm[1])
        terminal_error = error
        try:
            self.store.write_bytes('combined_log.txt', self.log.getvalue().encode('utf-8'))
            if kind is None:
                missing = set(self.store.caps) - {'artifact_manifest.json', 'error.txt'} - set(self.store.files)
                if missing:
                    raise ValueError('Successful body omitted declared artifacts: ' + repr(sorted(missing)))
            self._terminal(terminal_error, tb)
            if terminal_error is None:
                self.store.finalize()
        except BaseException as completion_error:
            if terminal_error is None:
                terminal_error = completion_error
                # Finalization errors must turn a completed body into a failed run.
                self._terminal(terminal_error, completion_error.__traceback__)
            print(json.dumps(self.status), flush=True)
            raise
        print(json.dumps(self.status), flush=True)
        return False

    def _terminal(self, error, tb):
        text = '' if error is None else ''.join(traceback.format_exception(type(error), error, tb))
        raw = text.encode('utf-8')
        cap = self.store.caps['error.txt']
        self.store.write_bytes('error.txt', raw[:cap])
        self.status.update(end_utc=datetime.now(timezone.utc).isoformat(),
                           wall_seconds=time.monotonic() - self.started,
                           status='completed' if error is None else ('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'),
                           exit_code=0 if error is None else (130 if isinstance(error, KeyboardInterrupt) else 1),
                           output_hashes={name: entry['sha256'] for name, entry in self.store.files.items() if name != 'status.json'})
        if error is not None:
            self.status.update(error=repr(error)[:4096], error_trace_bytes=len(raw),
                               error_trace_sha256=hashlib.sha256(raw).hexdigest(), error_trace_truncated=len(raw) > cap,
                               missing_artifacts=sorted(set(self.store.caps) - set(self.store.files) - {'artifact_manifest.json'}))
        self.store.write_json('status.json', self.status)
