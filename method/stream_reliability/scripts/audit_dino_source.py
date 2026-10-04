"""Fetch only bounded, pinned official DINO architecture sources, without install."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'method/frozen_probe'))
sys.path.insert(0, str(ROOT))
from cache import file_sha
from run_context import RunContext


def fetch(url, cap):
    req = urllib.request.Request(url, headers={'User-Agent': 'SignRep-input-audit',
                                               'Accept-Encoding': 'identity'})
    with urllib.request.urlopen(req, timeout=20) as response:
        data = response.read(cap + 1)
    if len(data) > cap:
        raise ValueError('Registered response cap exceeded')
    return data


def main():
    parser = argparse.ArgumentParser()
    for field in ('config', 'output-dir'):
        parser.add_argument('--' + field, required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        commit = cfg['commit']
        if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
            raise ValueError('Exact Git commit required')
        tree_bytes = fetch(f'https://api.github.com/repos/facebookresearch/dinov2/git/trees/{commit}?recursive=1',
                           cfg['max_tree_bytes'])
        tree = json.loads(tree_bytes)
        if tree.get('truncated') or tree['sha'] != commit:
            raise ValueError('Complete pinned official tree required')
        (run.output / 'remote_tree.json').write_bytes(tree_bytes)
        selected = [row for row in tree['tree'] if row['type'] == 'blob' and
                    (row['path'] in cfg['exact_paths'] or
                     any(row['path'].startswith(p) and row['path'].endswith('.py')
                         for p in cfg['python_prefixes']))]
        if not selected or any(r['size'] > cfg['max_file_bytes'] for r in selected):
            raise ValueError('Architecture source member bounds invalid')
        total = sum(row['size'] for row in selected)
        if total > cfg['max_source_bytes'] or len(selected) > cfg['max_files']:
            raise ValueError('Registered aggregate source cap exceeded')

        def download(row):
            name = row['path']
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe source path')
            url = f'https://raw.githubusercontent.com/facebookresearch/dinov2/{commit}/{name}'
            data = fetch(url, row['size'])
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if len(data) != row['size'] or blob != row['sha']:
                raise ValueError('Pinned Git blob differs')
            target = run.output / 'official_source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(data)
            return dict(path=name, bytes=len(data), git_blob_sha=blob,
                        sha256=hashlib.sha256(data).hexdigest(), url=url)

        with ThreadPoolExecutor(max_workers=4) as pool:
            files = list(pool.map(download, selected))
        report = dict(run_id=args.run_id, repository='facebookresearch/dinov2', commit=commit,
                      files=files, source_bytes=total, tree_bytes=len(tree_bytes), GPU_hours=0,
                      scope='Official pinned architecture source only; author historical DINO commit unknown',
                      environment_mutation=False, weights_downloaded=False)
        run.save('source_manifest.json', report)
        run.status['output_hashes'] = {name: file_sha(run.output / name)
                                      for name in ('remote_tree.json', 'source_manifest.json')}


if __name__ == '__main__':
    main()
