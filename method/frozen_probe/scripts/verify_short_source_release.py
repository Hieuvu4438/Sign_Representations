"""Bounded official archive equality for a contract-failing source; no replacement."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import file_sha
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        helper = REPO / config['range_reader_source']
        if file_sha(helper) != config['range_reader_sha256']:
            raise ValueError('Original bounded archive-reader source changed')
        (run.output / 'range_reader_snapshot.py').write_bytes(helper.read_bytes())
        spec = importlib.util.spec_from_file_location('owned_release_range_reader', helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        remote = None
        try:
            remote = module.RangeReader(config['archive_url'], cap=config['max_download_bytes'])
            with zipfile.ZipFile(remote) as archive:
                expected = 'videos/' + config['sample_id']
                matches = [n for n in archive.namelist() if n == expected or n.endswith('/' + expected)]
                if len(matches) != 1:
                    raise ValueError('Ambiguous/missing official source member')
                info = archive.getinfo(matches[0])
                if max(info.file_size, info.compress_size) > config['max_member_bytes']:
                    raise ValueError('Registered single-source archive byte cap exceeded')
                official = archive.read(info)  # ZIP CRC checked, full video archive never fetched.
                local = Path(config['local_path'])
                if file_sha(local) != config['local_sha256']:
                    raise ValueError('Local source changed since failed audit')
                official_sha = hashlib.sha256(official).hexdigest()
                run.save('release_equality.json', dict(sample_id=config['sample_id'], archive_member=info.filename,
                         archive_url=config['archive_url'], archive_bytes=remote.length, http_headers=remote.headers,
                         member_bytes=len(official), compressed_member_bytes=info.compress_size, zip_crc32=info.CRC,
                         official_sha256=official_sha, local_sha256=config['local_sha256'],
                         byte_equal=official_sha == config['local_sha256'] and local.stat().st_size == len(official),
                         downloaded_bytes=remote.downloaded, scope='one contract-failing source byte equality only; no source replacement/test scores'))
                run.status['output_hashes'] = {'release_equality.json':file_sha(run.output/'release_equality.json')}
        finally:
            if remote is not None:
                run.save('ranges.json', remote.requests)
                run.status['downloaded_bytes'] = remote.downloaded


if __name__ == '__main__':
    main()
