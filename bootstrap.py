"""Read-only bounded inventory; creates artifacts only in the project."""
import argparse
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from signrepr.io import write_json

DATASETS = {
    'csl_daily': '/home/shared_data/sign_language/CSLDaily',
    'how2sign': '/home/shared_data/sign_language/How2Sign',
    'how2sign_alt': '/home/dongvk/datasets/How2Sign',
    'asl_citizen': '/home/dongvk/datasets/ASL_Citizen',
    'ms_asl': '/home/dongvk/datasets/MS-ASL',
    'wlasl2000': '/home/dongvk/datasets/WLASL2000',
    'youtube_asl': '/home/dongvk/datasets/youtubeASL',
    'phoenix14t': '/home/dongvk/datasets/phoenix14T',
    'csl_daily_sentence_crop': '/home/dongvk/datasets/CSL_Daily_Sentence_Crop',
    'places365': '/home/dongvk/datasets/places365',
    'local_exploration_scripts': '/home/dongvk/datasets/script_data_exploring',
    'local_download_scripts': '/home/dongvk/datasets/src_for_download',
}


def scan(base, entry_limit=5000, depth_limit=3):
    queue = deque([(Path(base), 0)])
    counts, metadata, errors, links = Counter(), [], [], []
    count = 0
    truncated = False
    while queue and count < entry_limit:
        directory, depth = queue.popleft()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if count >= entry_limit:
                        truncated = True
                        break
                    count += 1
                    if entry.is_symlink():
                        links.append({'path': entry.path, 'target': os.readlink(entry.path)})
                    elif entry.is_dir(follow_symlinks=False):
                        if depth < depth_limit:
                            queue.append((Path(entry.path), depth + 1))
                        else:
                            truncated = True
                    elif entry.is_file(follow_symlinks=False):
                        suffix = Path(entry.name).suffix.lower()
                        counts[suffix] += 1
                        if suffix in {'.csv', '.tsv', '.json', '.jsonl', '.pkl', '.eaf', '.xml', '.txt', '.gzip'} or entry.name.startswith(('labels.', 'README')):
                            metadata.append({'path': entry.path, 'bytes': entry.stat().st_size})
        except OSError as error:
            errors.append({'path': str(directory), 'error': str(error)})
    return {'entries_examined': count, 'entry_limit': entry_limit,
            'depth_limit': depth_limit, 'truncated': truncated or bool(queue),
            'extension_counts_are_partial': dict(counts),
            'metadata_files_observed': metadata, 'errors': errors, 'symlinks': links}


def command(args):
    if not shutil.which(args[0]):
        return {'available': False}
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {'available': True, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    except subprocess.TimeoutExpired:
        return {'available': True, 'error': 'timeout'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--entry-limit', type=int, default=5000)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    for source in [Path('/home/shared_data/sign_language'), Path('/home/dongvk/datasets')]:
        if root == source.resolve() or source.resolve() in root.parents:
            raise ValueError('Project must be outside source roots')
    if args.dry_run:
        print({'root': str(root), 'source_roots_read_only': True, 'entry_limit_per_dataset': args.entry_limit})
        return
    for name in ['configs/experiments', 'scripts', 'src', 'tests', 'third_party', 'patches',
                 'evidence/code_changes', 'evidence/papers', 'provenance', 'data/manifests',
                 'data/splits', 'data/derived', 'data/external', 'features', 'checkpoints',
                 'runs', 'results/predictions', 'results/tables', 'reports', 'state', 'logs']:
        (root / name).mkdir(parents=True, exist_ok=True)
    inventory = []
    for dataset_id, raw_path in DATASETS.items():
        path = Path(raw_path)
        item = {'dataset_id': dataset_id, 'path': raw_path, 'resolved_path': str(path.resolve()),
                'exists': path.exists(), 'readable': os.access(path, os.R_OK | os.X_OK), 'download_allowed': False}
        if item['readable']:
            item['children'] = sorted(p.name for p in path.iterdir())
            item['scan'] = scan(path, args.entry_limit)
        inventory.append(item)
    packages = {}
    for name in ['torch', 'numpy', 'pandas', 'PyYAML', 'opencv-python', 'transformers', 'timm', 'scipy', 'scikit-learn']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    free = shutil.disk_usage(root).free
    report = {'utc_time': datetime.now(timezone.utc).isoformat(), 'hostname': platform.node(),
              'python': sys.version, 'python_executable': sys.executable, 'packages_observed_not_smoke_tested': packages,
              'disk_free_bytes': free, 'disk_free_min_bytes': 20 * 1024 ** 3,
              'heavy_write_gate': 'PASS' if free >= 20 * 1024 ** 3 else 'BLOCKED_COMPUTE',
              'dataset_roots': [str(p) for p in [Path('/home/shared_data/sign_language'), Path('/home/dongvk/datasets')]],
              'datasets': inventory, 'gpu': command(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,utilization.gpu', '--format=csv']),
              'gpu_allocation': {'index': 0, 'max_workers': 1, 'source': 'explicit user reply 2026-10-01'},
              'ffprobe': command(['ffprobe', '-version']), 'git': command(['git', '--version'])}
    write_json(root / 'evidence/bootstrap_inventory.json', report)
    print({'artifact': 'evidence/bootstrap_inventory.json', 'datasets_observed': len(inventory), 'heavy_write_gate': report['heavy_write_gate']})


if __name__ == '__main__':
    main()
