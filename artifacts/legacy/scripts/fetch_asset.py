"""Fetch an explicitly registered public asset, with size and checksum checks."""
import argparse
import json
import os
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from _common import ROOT
from signrepr.io import sha256, write_json


def validate_asset_bytes(path, asset):
    if asset.get('expected_bytes') and path.stat().st_size != asset['expected_bytes']:
        raise ValueError('Asset byte count mismatch')
    if asset.get('expected_sha256') and sha256(path) != asset['expected_sha256']:
        raise ValueError('Asset checksum mismatch')
    if asset.get('kind') == 'pdf':
        with path.open('rb') as stream:
            header = stream.read(1024)
            stream.seek(max(0, path.stat().st_size - 2048))
            tail = stream.read()
        if not header.lstrip().startswith(b'%PDF') or b'%%EOF' not in tail:
            raise ValueError('Incomplete PDF or HTML/error content; refusing to accept it')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=ROOT / 'provenance/assets.json')
    parser.add_argument('--asset', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    asset = registry['assets'][args.asset]
    target = ROOT / asset['local_path']
    if not target.resolve().is_relative_to(ROOT):
        raise ValueError('Asset target outside workspace')
    free = shutil.disk_usage(ROOT).free
    expected_bytes = asset.get('expected_bytes')
    max_bytes = expected_bytes or asset['max_bytes']
    reserve = registry['disk_free_min_bytes']
    estimate = {'asset_id': args.asset, 'url': asset['url'], 'max_bytes': max_bytes,
                'free_bytes': free, 'reserve_bytes': reserve, 'existing': target.exists()}
    print(json.dumps(estimate), flush=True)
    if args.dry_run:
        return
    if target.exists():
        validate_asset_bytes(target, asset)
        actual_hash = sha256(target)
        expected_hash = asset.get('expected_sha256')
        if expected_hash and actual_hash != expected_hash:
            raise ValueError('Existing asset checksum mismatch; refusing to load or overwrite')
    else:
        if free - max_bytes < reserve:
            raise RuntimeError('BLOCKED_COMPUTE: disk reserve would be violated')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + '.partial')
        request = urllib.request.Request(asset['url'], headers={'User-Agent': 'SignRepresentationResearch/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as stream:
            content_type = response.headers.get('Content-Type', '')
            content_range = response.headers.get('Content-Range')
            if response.status == 206 or content_range:
                raise ValueError('Unexpected partial HTTP response; use a complete registered source')
            content_length = response.headers.get('Content-Length')
            downloaded = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                downloaded += len(block)
                if downloaded > max_bytes:
                    raise ValueError('Asset exceeds registered size cap')
                stream.write(block)
        if content_length and downloaded != int(content_length):
            raise ValueError('HTTP Content-Length differs from downloaded bytes')
        if expected_bytes and downloaded != expected_bytes:
            raise ValueError('Asset byte count mismatch')
        actual_hash = sha256(temporary)
        if asset.get('expected_sha256') and actual_hash != asset['expected_sha256']:
            raise ValueError('Checksum mismatch; partial quarantined and must not be loaded')
        validate_asset_bytes(temporary, asset)
        os.replace(temporary, target)
    write_json(ROOT / 'provenance' / (args.asset + '.json'), {
        'asset_id': args.asset, 'url': asset['url'], 'revision': asset.get('revision'),
        'sha256': actual_hash, 'bytes': target.stat().st_size, 'local_path': str(target.relative_to(ROOT)),
        'utc_time': datetime.now(timezone.utc).isoformat(), 'status': 'VERIFIED_LOCAL_BYTES',
        'license_access_note': asset.get('license_access_note'), 'content_type': locals().get('content_type')})
    print(json.dumps({'asset_id': args.asset, 'status': 'VERIFIED_LOCAL_BYTES', 'sha256': actual_hash}), flush=True)


if __name__ == '__main__':
    main()
