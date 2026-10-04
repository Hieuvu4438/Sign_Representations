"""Bounded, content-verified window cache. No model or dataset discovery."""
import hashlib
import json
from pathlib import Path

import numpy as np

from temporal import window_plan


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sample_map(cohort):
    samples = cohort['samples']
    indexed = {s['sample_id']: s for s in samples}
    classes = cohort['classes']
    if not samples or len(indexed) != len(samples) or not classes or len(set(classes)) != len(classes):
        raise ValueError('Invalid sample/class identities')
    for s in samples:
        if (s['split'] not in ('train', 'val') or type(s['class_id']) is not int
                or not 0 <= s['class_id'] < len(classes) or classes[s['class_id']] != s['gloss']):
            raise ValueError('Split or class mapping differs')
        h = s['source_sha256']
        if len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
            raise ValueError('Invalid source digest')
    train = {s['participant_id'] for s in samples if s['split'] == 'train'}
    dev = {s['participant_id'] for s in samples if s['split'] == 'val'}
    if not train or not dev or train & dev:
        raise ValueError('Participant overlap or empty split')
    return indexed


def expected_centers(video, contract):
    pts = video['pts_s']
    if len(pts) != video['frames']:
        raise ValueError('Audit frame count differs')
    plan = window_plan(pts, contract['window_length'], contract['stride'])
    if len(plan) != video['windows']:
        raise ValueError('Audit window coverage differs')
    return np.array([w['center_s'] for w in plan], dtype=np.float64)


def check_arrays(features, activity, centers, video, contract):
    expected = expected_centers(video, contract)
    arrays = (features, activity, centers)
    if (features.dtype != np.dtype('float32') or activity.dtype != np.dtype('float32')
            or centers.dtype != np.dtype('float64') or features.shape != (len(expected), contract['output_dim'])
            or activity.shape != (len(expected), 2) or centers.shape != expected.shape):
        raise ValueError('Cache shape/dtype mismatch')
    if any(not np.isfinite(x).all() for x in arrays) or not np.array_equal(centers, expected):
        raise ValueError('Nonfinite cache or native timeline mismatch')


class CacheWriter:
    """Exclusive creation; completed caches can only be reopened via CacheReader."""
    def __init__(self, directory, identity, cohort, audit, byte_limit):
        self.directory = Path(directory)
        self.identity = identity
        self.samples = sample_map(cohort)
        videos = audit['samples']
        self.videos = {s['sample_id']: s for s in videos}
        if not audit['passed'] or len(self.videos) != len(videos) or set(self.videos) != set(self.samples):
            raise ValueError('Incomplete video audit')
        if identity['contract']['output_layer'] != 'features' or identity['contract']['dtype'] != 'float32':
            raise ValueError('Only registered feature contract supported')
        for key, s in self.samples.items():
            if any(self.videos[key][field] != s[field] for field in ('source_sha256', 'split', 'participant_id', 'class_id')):
                raise ValueError('Audit/source identity differs')
        if type(byte_limit) is not int or byte_limit < 1:
            raise ValueError('Invalid byte limit')
        self.byte_limit = byte_limit
        self.bytes = 0
        self.entries = {}
        self.directory.mkdir(parents=True, exist_ok=False)
        self.manifest = {'cache_version': 1, 'status': 'running', 'identity': identity,
                         'identity_sha256': object_sha(identity), 'samples': [], 'payload_bytes': 0}
        atomic_json(self.directory / 'manifest.json', self.manifest)

    def write(self, sample_id, features, activity, centers):
        if self.manifest['status'] != 'running' or sample_id not in self.samples or sample_id in self.entries:
            raise ValueError('Unknown/duplicate source or terminal cache')
        check_arrays(features, activity, centers, self.videos[sample_id], self.identity['contract'])
        name = hashlib.sha256(sample_id.encode()).hexdigest() + '.npz'
        path = self.directory / name
        temporary = self.directory / (name + '.tmp')
        # Uncompressed NPZ bounds bytes; no pickle or hidden dtype conversions.
        predicted = sum(x.nbytes for x in (features, activity, centers)) + 2048
        if self.bytes + predicted > self.byte_limit:
            raise ValueError('Cache byte limit exceeded')
        with temporary.open('xb') as stream:
            np.savez(stream, features=features, raw_activity=activity, centers_s=centers)
        if self.bytes + temporary.stat().st_size > self.byte_limit:
            raise ValueError('Serialized cache byte limit exceeded')
        temporary.replace(path)
        s = self.samples[sample_id]
        entry = {k: s[k] for k in ('sample_id', 'source_sha256', 'split', 'participant_id', 'class_id')}
        entry.update(file=name, sha256=file_sha(path), bytes=path.stat().st_size,
                     shape=list(features.shape), status='completed')
        self.bytes += entry['bytes']
        self.entries[sample_id] = entry
        self.manifest.update(samples=list(self.entries.values()), payload_bytes=self.bytes)
        atomic_json(self.directory / 'manifest.json', self.manifest)

    def complete(self):
        if self.manifest['status'] != 'running' or set(self.entries) != set(self.samples):
            raise ValueError('Cannot complete partial/terminal cache')
        self.manifest['status'] = 'completed'
        atomic_json(self.directory / 'manifest.json', self.manifest)
        return self.manifest


class CacheReader:
    def __init__(self, manifest_path, identity, cohort, audit):
        self.path = Path(manifest_path)
        self.directory = self.path.parent.resolve()
        self.manifest = json.loads(self.path.read_text())
        m = self.manifest
        if (m['cache_version'] != 1 or m['status'] != 'completed' or m['identity'] != identity
                or m['identity_sha256'] != object_sha(identity)):
            raise ValueError('Cache identity/status mismatch')
        self.contract = identity['contract']
        self.samples = sample_map(cohort)
        self.entries = {s['sample_id']: s for s in m['samples']}
        self.videos = {s['sample_id']: s for s in audit['samples']}
        if (not audit['passed'] or len(self.videos) != len(audit['samples'])
                or set(self.videos) != set(self.samples) or len(self.entries) != len(m['samples'])
                or set(self.entries) != set(self.samples)):
            raise ValueError('Cache/audit coverage mismatch')
        total = 0
        files = set()
        for key, entry in self.entries.items():
            source = self.samples[key]
            if (entry['status'] != 'completed'
                    or any(entry[field] != source[field] or self.videos[key][field] != source[field]
                           for field in ('source_sha256', 'split', 'participant_id', 'class_id'))):
                raise ValueError('Cache/audit source binding differs')
            name = entry['file']
            if not isinstance(name, str) or Path(name).name != name or name in files or not name.endswith('.npz'):
                raise ValueError('Unsafe/duplicate shard path')
            files.add(name)
            path = self.directory / name
            if path.is_symlink() or path.resolve().parent != self.directory:
                raise ValueError('Shard escaped cache directory')
            if path.stat().st_size != entry['bytes'] or file_sha(path) != entry['sha256']:
                raise ValueError('Corrupt cache shard')
            total += entry['bytes']
        if total != m['payload_bytes']:
            raise ValueError('Cache byte accounting differs')

    def load(self, sample_id):
        entry = self.entries[sample_id]
        path = self.directory / entry['file']
        if path.is_symlink() or file_sha(path) != entry['sha256']:
            raise ValueError('Shard changed after verification')
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {'features', 'raw_activity', 'centers_s'}:
                raise ValueError('Unexpected cache arrays')
            features, activity, centers = (data[k] for k in ('features', 'raw_activity', 'centers_s'))
        check_arrays(features, activity, centers, self.videos[sample_id], self.contract)
        if list(features.shape) != entry['shape']:
            raise ValueError('Manifest shape differs')
        return features, activity, centers
