import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from baseline import pool_cache, run_controls, support_ids
from cache import CacheReader, CacheWriter, atomic_json, expected_centers, file_sha


def fixture():
    classes = ['A', 'B', 'C']
    samples, videos = [], []
    supports = {str(s): {'1': {}, '2': {}} for s in (1701, 1702, 1703)}
    for c, gloss in enumerate(classes):
        train = []
        for split, number in (('train', 0), ('train', 1), ('val', 0)):
            key = f'{split}_{c}_{number}'
            digest = hashlib.sha256(key.encode()).hexdigest()
            sample = dict(sample_id=key, gloss=gloss, class_id=c, split=split,
                          participant_id='P_train' if split == 'train' else 'P_dev', source_sha256=digest)
            samples.append(sample)
            videos.append({k: sample[k] for k in ('sample_id', 'class_id', 'split', 'participant_id', 'source_sha256')} |
                          dict(pts_s=[i * .04 for i in range(22)], frames=22, windows=4))
            if split == 'train': train.append(key)
        for seed in supports:
            supports[seed]['1'][gloss] = train[:1]
            supports[seed]['2'][gloss] = train
    cohort = dict(classes=classes, samples=samples, supports=supports)
    audit = dict(passed=True, samples=videos)
    identity = dict(cohort_sha256='a' * 64, audit_sha256='b' * 64,
                    checkpoint_sha256='c' * 64, upstream_commit='d' * 40,
                    preprocessing_code_sha256='e' * 64,
                    contract=dict(output_layer='features', output_dim=3, dtype='float32',
                                  window_length=16, stride=2, context='offline_native_window'))
    return cohort, audit, identity


class CacheBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'cache'
        self.cohort, self.audit, self.identity = fixture()
        writer = CacheWriter(self.root, self.identity, self.cohort, self.audit, 1 << 20)
        for sample, video in zip(self.cohort['samples'], self.audit['samples']):
            x = np.tile(np.eye(3, dtype=np.float32)[sample['class_id']], (4, 1))
            a = np.ones((4, 2), dtype=np.float32)
            writer.write(sample['sample_id'], x, a, expected_centers(video, self.identity['contract']))
        writer.complete()
        self.path = self.root / 'manifest.json'

    def read(self, **kwargs):
        return CacheReader(self.path, kwargs.get('identity', self.identity),
                           kwargs.get('cohort', self.cohort), kwargs.get('audit', self.audit))

    def edit(self, mutate):
        manifest = json.loads(self.path.read_text())
        mutate(manifest)
        atomic_json(self.path, manifest)

    def test_round_trip_and_complete_rankings(self):
        reader = self.read()
        ids, matrices, projection, meta = pool_cache(reader, self.cohort)
        queries, results, ranks = run_controls(self.cohort, ids, matrices, [1701, 1702, 1703], [1, 2])
        self.assertEqual(len(results), 54)
        self.assertEqual(projection.shape, (12, 3))
        self.assertEqual(set(meta), set(ids))
        labels = [next(s['class_id'] for s in self.cohort['samples'] if s['sample_id'] == q) for q in queries]
        for result in results:
            self.assertEqual(result['metrics']['top1'], 1)
            ranking = ranks[result['name']]
            self.assertEqual(ranking.shape, (3, 3))
            self.assertEqual(ranking[:, 0].tolist(), labels)
            for row in ranking: self.assertEqual(sorted(row.tolist()), [0, 1, 2])
        reordered = {name: x[::-1] for name, x in matrices.items()}
        q2, r2, p2 = run_controls(self.cohort, ids[::-1], reordered, [1701, 1702, 1703], [1, 2])
        self.assertEqual(queries, q2)
        self.assertEqual(results, r2)
        for name in ranks: np.testing.assert_array_equal(ranks[name], p2[name])

    def test_identity_coverage_and_unsafe_paths(self):
        identity = copy.deepcopy(self.identity)
        identity['checkpoint_sha256'] = 'f' * 64
        with self.assertRaises(ValueError): self.read(identity=identity)
        original = self.path.read_text()
        mutations = [lambda m: m['samples'].pop(), lambda m: m['samples'].append(m['samples'][0]),
                     lambda m: m['samples'][0].update(file='../outside.npz'),
                     lambda m: m.update(status='running'), lambda m: m.update(payload_bytes=1),
                     lambda m: m['samples'][0].update(split='val')]
        for mutate in mutations:
            self.path.write_text(original)
            self.edit(mutate)
            with self.assertRaises(ValueError): self.read()

    def test_corrupt_shard_and_timeline_dtype(self):
        reader = self.read()
        key = self.cohort['samples'][0]['sample_id']
        entry = reader.entries[key]
        path = self.root / entry['file']
        original = path.read_bytes()
        path.write_bytes(original + b'corrupt')
        with self.assertRaises(ValueError): self.read()
        for alter in ('timeline', 'dtype'):
            path.write_bytes(original)
            with np.load(path, allow_pickle=False) as data: values = {k: data[k] for k in data.files}
            if alter == 'timeline': values['centers_s'][0] += .001
            else: values['features'] = values['features'].astype(np.float64)
            np.savez(path, **values)
            manifest = json.loads(self.path.read_text())
            manifest['samples'][0].update(sha256=file_sha(path), bytes=path.stat().st_size)
            manifest['payload_bytes'] = sum(e['bytes'] for e in manifest['samples'])
            atomic_json(self.path, manifest)
            with self.assertRaises(ValueError): self.read().load(key)

    def test_exclusive_partial_cache_and_byte_limit(self):
        with self.assertRaises(FileExistsError):
            CacheWriter(self.root, self.identity, self.cohort, self.audit, 1 << 20)
        writer = CacheWriter(self.root.parent / 'partial', self.identity, self.cohort, self.audit, 1)
        with self.assertRaises(ValueError): writer.complete()
        video = self.audit['samples'][0]
        with self.assertRaises(ValueError):
            writer.write(video['sample_id'], np.ones((4, 3), np.float32), np.ones((4, 2), np.float32),
                         expected_centers(video, self.identity['contract']))

    def test_support_leakage_class_budget_and_participant(self):
        original = copy.deepcopy(self.cohort)
        mutations = [lambda c: c['supports']['1701']['1']['A'].__setitem__(0, 'val_0_0'),
                     lambda c: c['supports']['1701']['1']['A'].__setitem__(0, 'train_1_0'),
                     lambda c: c['supports']['1701']['1']['A'].append('train_0_1'),
                     lambda c: c['samples'][-1].update(participant_id='P_train')]
        for mutate in mutations:
            cohort = copy.deepcopy(original)
            mutate(cohort)
            with self.assertRaises(ValueError): support_ids(cohort, 1701, 1)
        cohort = copy.deepcopy(original)
        cohort['samples'].append(cohort['samples'][0])
        with self.assertRaises(ValueError): self.read(cohort=cohort)


if __name__ == '__main__': unittest.main()
