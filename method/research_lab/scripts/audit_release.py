"""Read only bounded ZIP metadata ranges from the official ASL Citizen archive."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import time
import traceback
import urllib.request
import zipfile

URL = 'https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip'


def digest(data):
    return hashlib.sha256(data).hexdigest()


class RangeReader(io.RawIOBase):
    def __init__(self, url, cap=32*1024*1024):
        self.url, self.cap, self.position, self.downloaded = url, cap, 0, 0
        self.requests = []
        request = urllib.request.Request(url, method='HEAD', headers={'Accept-Encoding': 'identity'})
        with urllib.request.urlopen(request, timeout=30) as response:
            self.headers = dict(response.headers)
            self.length = int(response.headers['Content-Length'])
            self.etag = response.headers.get('ETag')
        if self.length < 22:
            raise ValueError('Invalid archive size')

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        target = offset if whence == 0 else self.position+offset if whence == 1 else self.length+offset
        if whence not in (0,1,2) or target < 0:
            raise ValueError('Invalid seek')
        self.position = target
        return target

    def read(self, size=-1):
        if size < 0:
            size = self.length-self.position
        size = min(size, self.length-self.position)
        if size <= 0:
            return b''
        if self.downloaded+size > self.cap:
            raise ValueError('Metadata byte budget exceeded; archive will not be downloaded')
        start, stop = self.position, self.position+size-1
        headers = {'Range': f'bytes={start}-{stop}', 'Accept-Encoding': 'identity'}
        if self.etag:
            headers['If-Match'] = self.etag
        request = urllib.request.Request(self.url, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{stop}/{self.length}':
                raise ValueError('Server did not honor exact byte range; refusing full archive')
            data = response.read(size+1)
        if len(data) != size:
            raise ValueError('Unexpected range response length')
        self.requests.append({'start': start, 'stop_inclusive': stop, 'bytes': len(data), 'sha256': digest(data)})
        self.downloaded += len(data)
        self.position += len(data)
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--local-root', type=Path, default=Path('/home/dongvk/datasets/ASL_Citizen'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {'run_id': args.run_id, 'status': 'running', 'start_utc': datetime.now(timezone.utc).isoformat(),
              'pid': os.getpid(), 'host': platform.node(), 'GPU_hours': 0,
              'command': [sys.executable]+sys.argv, 'script_sha256': digest(Path(__file__).read_bytes())}

    def save(name, value):
        temporary = args.output_dir/(name+'.tmp')
        temporary.write_text(json.dumps(value, indent=2)+'\n')
        temporary.replace(args.output_dir/name)

    def timeout_handler(*_):
        raise TimeoutError('Release audit time budget exhausted')

    remote = None
    save('status.json', status)
    shutil.copyfile(__file__, args.output_dir/'audit_release.py')
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)
    try:
        remote = RangeReader(URL)
        report = {'run_id': args.run_id, 'archive_url': URL, 'archive_bytes': remote.length,
                  'http_headers': remote.headers, 'scope': 'metadata equality only; no video bytes checked',
                  'members': [], 'all_selected_metadata_byte_equal': False}
        with zipfile.ZipFile(remote) as archive:
            report['archive_member_count'] = len(archive.infolist())
            for relative in ('splits/train.csv', 'splits/val.csv', 'splits/test.csv', 'use.txt'):
                candidates = [name for name in archive.namelist() if name == relative or name.endswith('/'+relative)]
                if len(candidates) != 1:
                    raise ValueError(f'Ambiguous/missing release member: {relative}')
                info = archive.getinfo(candidates[0])
                if info.file_size > 8*1024*1024 or info.compress_size > 8*1024*1024:
                    raise ValueError('Metadata member exceeds bounded size')
                data = archive.read(info)  # ZIP CRC is checked by zipfile.
                local = (args.local_root/relative).read_bytes()
                report['members'].append({'local_relative_path': relative, 'archive_member': info.filename,
                                          'file_bytes': info.file_size, 'compressed_bytes': info.compress_size,
                                          'zip_crc32': info.CRC, 'official_sha256': digest(data),
                                          'local_sha256': digest(local), 'byte_equal': data == local})
                save('release.json', report)
        report['all_selected_metadata_byte_equal'] = all(m['byte_equal'] for m in report['members'])
        report['requests'] = remote.requests
        report['downloaded_bytes'] = remote.downloaded
        save('release.json', report)
        status.update(status='completed', exit_code=0, output_sha256=digest((args.output_dir/'release.json').read_bytes()))
    except Exception as error:
        status.update(status='failed', exit_code=1, error=repr(error))
        (args.output_dir/'error.txt').write_text(traceback.format_exc())
    finally:
        signal.alarm(0)
        if remote is not None:
            status['downloaded_bytes'] = remote.downloaded
            save('ranges.json', remote.requests)
        status.update(end_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic()-started)
        save('status.json', status)
    print(json.dumps(status), flush=True)
    if status['status'] != 'completed':
        sys.exit(1)


if __name__ == '__main__':
    main()
