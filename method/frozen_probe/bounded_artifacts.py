"""Exclusive artifact store with preflight file limits and bounded atomic writes.

Bounds cover logical file bytes (including temporary replacement files), not
filesystem allocation or RAM. Callers must route every run artifact through this
store; it checks for unexpected files but cannot constrain unrelated processes.
"""
from collections.abc import Mapping
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re

import numpy as np

MANIFEST = 'artifact_manifest.json'


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=True, allow_nan=False,
                       sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')


def _name(name):
    if not isinstance(name, str) or not name:
        raise ValueError('Artifact name must be a nonempty relative path')
    parts = name.split('/')
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', p) for p in parts):
        raise ValueError('Unsafe artifact path')
    if str(PurePosixPath(name)) != name:
        raise ValueError('Noncanonical artifact path')


def inspect_file_plan(file_caps, ceiling_bytes, replaceable=()):
    """Validate all file and atomic-replacement bounds before creating a run."""
    if not isinstance(file_caps, Mapping) or not file_caps:
        raise ValueError('A complete file budget is required')
    if type(ceiling_bytes) is not int or ceiling_bytes <= 0:
        raise ValueError('Invalid total byte ceiling')
    for name, cap in file_caps.items():
        _name(name)
        if type(cap) is not int or cap <= 0:
            raise ValueError('Invalid file cap')
        if any(other.startswith(name + '/') for other in file_caps):
            raise ValueError('Artifact file/directory path collision')
    if len(set(replaceable)) != len(replaceable) or not set(replaceable) <= set(file_caps):
        raise ValueError('Replacement files must be declared once')
    temporary = max((file_caps[n] for n in replaceable), default=0)
    maximum = sum(file_caps.values()) + temporary
    return dict(file_caps=dict(file_caps), ceiling_bytes=ceiling_bytes,
                replaceable=list(replaceable), replacement_scratch_bytes=temporary,
                maximum_logical_bytes=maximum, passed=maximum <= ceiling_bytes)


def array_spec(shape, dtype):
    if not shape or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError('Array shape must have positive integer dimensions')
    dt = np.dtype(dtype)
    if dt.hasobject or dt.fields or dt.subdtype or dt.kind not in 'biuf':
        raise ValueError('Only plain numeric arrays are permitted')
    header = io.BytesIO()
    np.lib.format.write_array_header_2_0(header, dict(
        descr=np.lib.format.dtype_to_descr(dt), fortran_order=False, shape=tuple(shape)))
    payload = int(np.prod(shape, dtype=object)) * dt.itemsize
    return dict(shape=list(shape), dtype=dt.str, payload_bytes=payload,
                header_bytes=len(header.getvalue()), file_bytes=payload + len(header.getvalue()))


class BoundedArtifactStore:
    def __init__(self, output, file_caps, ceiling_bytes, replaceable=(), manifest_cap=65536):
        caps = dict(file_caps)
        if MANIFEST in caps:
            raise ValueError('Manifest budget is set with manifest_cap')
        caps[MANIFEST] = manifest_cap
        self.plan = inspect_file_plan(caps, ceiling_bytes, replaceable)
        if not self.plan['passed']:
            raise ValueError('Complete artifact plan exceeds registered ceiling')
        self.root = Path(output)
        # Exclusive directory: no overwrite, resume or implicit parent creation.
        self.root.mkdir(exist_ok=False)
        self.caps = caps
        self.replaceable = frozenset(replaceable)
        self.files = {}
        self.partials = {}
        self.closed = False

    def _inventory(self):
        observed = {}
        for path in self.root.rglob('*'):
            if path.is_symlink():
                raise ValueError('Symlink in artifact directory')
            if path.is_file():
                observed[path.relative_to(self.root).as_posix()] = path.stat().st_size
            elif not path.is_dir():
                raise ValueError('Unsupported artifact file type')
        expected = {name: entry['bytes'] for name, entry in self.files.items()}
        expected.update(self.partials)
        if observed != expected:
            raise ValueError('Unexpected, missing or changed artifact bytes')
        if sum(observed.values()) > self.plan['ceiling_bytes']:
            raise ValueError('Observed run exceeds registered ceiling')
        return observed

    def write_bytes(self, name, data):
        if self.closed:
            raise ValueError('Artifact store is finalized')
        if name not in self.caps:
            raise ValueError('Undeclared artifact')
        if not isinstance(data, bytes):
            raise ValueError('Artifact data must be bytes')
        if len(data) > self.caps[name]:
            raise ValueError('File exceeds its registered bound')
        if name in self.files and name not in self.replaceable:
            raise ValueError('Immutable artifact already exists')
        current = self._inventory()
        if self.partials:
            raise ValueError('Incomplete write retained; use a new registered run')
        if sum(current.values()) + len(data) > self.plan['ceiling_bytes']:
            raise ValueError('Atomic write exceeds total ceiling')
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name('.' + target.name + '.partial')
        try:
            with temporary.open('xb') as stream:
                written = stream.write(data)
                if written != len(data):
                    raise OSError('Short artifact write')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            if temporary.exists():
                self.partials[temporary.relative_to(self.root).as_posix()] = temporary.stat().st_size
            raise
        self.files[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        self._inventory()

    def write_json(self, name, value):
        self.write_bytes(name, json_bytes(value))

    def write_array(self, name, array, spec):
        a = np.asarray(array)
        expected = array_spec(spec['shape'], spec['dtype'])
        if expected != spec or a.shape != tuple(spec['shape']) or a.dtype.str != spec['dtype']:
            raise ValueError('Array differs from registered shape/dtype/byte plan')
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite artifact array')
        if self.caps.get(name) != spec['file_bytes']:
            raise ValueError('Array file cap differs from exact NPY format bound')
        buffer = io.BytesIO()
        np.lib.format.write_array(buffer, np.ascontiguousarray(a), version=(2, 0), allow_pickle=False)
        data = buffer.getvalue()
        if len(data) != spec['file_bytes']:
            raise ValueError('Unexpected NPY serialization size')
        self.write_bytes(name, data)

    def finalize(self):
        if self.closed:
            raise ValueError('Artifact store already finalized')
        if self.partials or set(self.files) != set(self.caps) - {MANIFEST}:
            raise ValueError('Run missing declared artifacts or retains partial writes')
        self._inventory()
        # Rehash durable bytes, including replacements, before a completed manifest.
        for name, entry in self.files.items():
            if hashlib.sha256((self.root / name).read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('Artifact digest changed')
        manifest = dict(format_version=1, plan=self.plan, artifacts=dict(self.files),
                        bytes_excluding_manifest=sum(e['bytes'] for e in self.files.values()),
                        scope='logical file bytes; excludes RAM and filesystem allocation')
        self.write_json(MANIFEST, manifest)
        self.closed = True
        return dict(manifest, final_logical_bytes=sum(self._inventory().values()))


def verify_store(output):
    root = Path(output)
    manifest = json.loads((root / MANIFEST).read_text())
    plan = manifest['plan']
    if inspect_file_plan(plan['file_caps'], plan['ceiling_bytes'], plan['replaceable']) != plan or not plan['passed']:
        raise ValueError('Invalid stored file plan')
    artifacts = manifest['artifacts']
    if set(artifacts) != set(plan['file_caps']) - {MANIFEST}:
        raise ValueError('Manifest artifact coverage mismatch')
    actual = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symlink in artifact directory')
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = path.stat().st_size
        elif not path.is_dir():
            raise ValueError('Unsupported artifact file type')
    if set(actual) != set(plan['file_caps']):
        raise ValueError('Unexpected or missing artifact files')
    for name, size in actual.items():
        if size > plan['file_caps'][name]:
            raise ValueError('Artifact exceeds stored bound')
        if name != MANIFEST:
            if size != artifacts[name]['bytes'] or hashlib.sha256((root / name).read_bytes()).hexdigest() != artifacts[name]['sha256']:
                raise ValueError('Artifact size or digest mismatch')
    if sum(actual[n] for n in artifacts) != manifest['bytes_excluding_manifest']:
        raise ValueError('Manifest byte accounting mismatch')
    if sum(actual.values()) > plan['ceiling_bytes']:
        raise ValueError('Artifact directory exceeds stored ceiling')
    return dict(passed=True, file_count=len(actual), logical_bytes=sum(actual.values()),
                ceiling_bytes=plan['ceiling_bytes'])
