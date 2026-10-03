"""Extract verified SignRep windows with source/config-addressed resumable shards."""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import yaml

from _common import ROOT
from signrepr.io import read_jsonl, sha256, write_json, write_jsonl


def cache_key(row, source_hash, config, implementation_hash):
    payload = {'sample_id': row['sample_id'], 'source_sha256': source_hash,
               'dataset_metadata_sha256': sha256(row['metadata_source']),
               'config': config, 'implementation_sha256': implementation_hash,
               'local_interval': [row.get('start_sec'), row.get('end_sec')]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def valid_shard(path, key):
    metadata_path = path.with_suffix('.json')
    if not path.exists() or not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('cache_key') != key or metadata.get('shard_sha256') != sha256(path):
        return False
    with np.load(path, allow_pickle=False) as shard:
        return bool(np.isfinite(shard['embeddings']).all() and len(shard['embeddings']) == len(shard['valid_mask']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--backbone', required=True, choices=['signrep', 'shubert_native'])
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--determinism-check', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    rows = [row for _, row in read_jsonl(args.manifest)]
    if args.dry_run:
        print(json.dumps({'samples': len(rows), 'source_bytes': sum((Path(row['source_root']) / row['relative_path']).stat().st_size for row in rows if row.get('source_exists')),
                          'checkpoint': config['checkpoint_path'], 'free_bytes': shutil.disk_usage(ROOT).free,
                          'backbone': args.backbone, 'new_network_downloads': False}))
        return
    if args.backbone != 'signrep':
        raise ValueError('SHuBERT native adapter not implemented yet; refusing substitution')
    if any(row['input_kind'] != 'video' or row.get('start_sec') is not None for row in rows):
        raise ValueError('This adapter requires complete video files; interval or frame-directory extraction not implemented')
    from signrepr.signrep import SignRepAdapter
    args.output.mkdir(parents=True, exist_ok=True)
    implementation_hash = hashlib.sha256((ROOT / 'src/signrepr/signrep.py').read_bytes() + Path(__file__).read_bytes()).hexdigest()
    expected_config_hash = sha256(args.config)
    run_manifest_path = args.output / ('manifest_' + sha256(args.manifest)[:16] + '.json')
    if run_manifest_path.exists():
        previous = json.loads(run_manifest_path.read_text())
        if previous['config_sha256'] != expected_config_hash:
            raise ValueError('Resume config hash mismatch; choose a separate output directory')
    write_json(run_manifest_path, {'manifest_sha256': sha256(args.manifest), 'config_sha256': expected_config_hash,
                                 'implementation_sha256': implementation_hash, 'status': 'RUNNING', 'pid': os.getpid()})
    model = SignRepAdapter(ROOT, config)
    write_json(args.output / 'load_audit.json', model.load_audit)
    index, failures = [], []
    begin = time.perf_counter()
    for row in rows:
        try:
            if shutil.disk_usage(ROOT).free < config['disk_free_min_gib'] * 1024 ** 3:
                raise RuntimeError('DISK_RESERVE_VIOLATED')
            source = Path(row['source_root']) / row['relative_path']
            source_hash = sha256(source)
            key = cache_key(row, source_hash, config, implementation_hash)
            path = args.output / (key + '.npz')
            reused = args.resume and valid_shard(path, key)
            if reused:
                metadata = json.loads(path.with_suffix('.json').read_text())
            else:
                arrays, measurements = model.extract(source)
                if not measurements['rgb_check'] or not measurements['upstream_transform_equivalence']:
                    raise ValueError('RGB_OR_TRANSFORM_CHECK_FAILED')
                if args.determinism_check:
                    repeat, _ = model.extract(source)
                    difference = float(np.max(np.abs(arrays['embeddings'] - repeat['embeddings'])))
                    measurements['determinism_max_absolute_difference'] = difference
                    if difference > config['deterministic_tolerance']:
                        raise ValueError('DETERMINISM_CHECK_FAILED')
                temporary = path.with_suffix('.npz.partial')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
                with np.load(temporary, allow_pickle=False) as check:
                    if not np.isfinite(check['embeddings']).all():
                        raise ValueError('SHARD_VALIDATION_FAILED')
                os.replace(temporary, path)
                metadata = {'sample_id': row['sample_id'], 'cache_key': key, 'source_sha256': source_hash,
                            'config_sha256': expected_config_hash, 'manifest_sha256': sha256(args.manifest),
                            'implementation_sha256': implementation_hash, 'shard_sha256': sha256(path),
                            'shard_bytes': path.stat().st_size, 'dimension': config['expected_dimension'],
                            'dtype': str(arrays['embeddings'].dtype), 'measurements': measurements}
                write_json(path.with_suffix('.json'), metadata)
            index.append({'sample_id': row['sample_id'], 'status': 'SUCCESS', 'cache_key': key, 'shard': str(path.resolve()), 'reused': reused})
            print(json.dumps({'sample_id': row['sample_id'], 'status': 'SUCCESS', 'reused': reused, 'measurements': metadata['measurements']}), flush=True)
        except Exception as error:
            failure = {'sample_id': row['sample_id'], 'status': 'FAIL', 'error': f'{type(error).__name__}: {error}'}
            failures.append(failure)
            index.append(failure)
            print(json.dumps(failure), flush=True)
        write_jsonl(args.output / 'index.jsonl', index)
    report = {'status': 'FAIL' if failures else 'PASS', 'samples': len(rows), 'success': len(rows) - len(failures),
              'failures': failures, 'elapsed_seconds': time.perf_counter() - begin,
              'manifest_sha256': sha256(args.manifest), 'config_sha256': expected_config_hash,
              'implementation_sha256': implementation_hash, 'load_audit': model.load_audit}
    write_json(args.output / 'extraction_report.json', report)
    write_json(run_manifest_path, {**report, 'pid': os.getpid()})
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
