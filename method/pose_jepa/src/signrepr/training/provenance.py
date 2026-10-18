"""Scientific source/data/config provenance shared by every training workflow."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import torch
from ..config import config_hash
from ..data.schema import read_manifest, sha256_file
from .distributed import world_size


def source_hash():
    root = Path(__file__).parents[1]
    h = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def provenance(config, manifest):
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    rows = read_manifest(manifest)
    if len({r.get('score_profile', 'raw_probability') for r in rows}) != 1:
        raise ValueError('Do not mix detector score conversion profiles in one training run')
    caches = []
    for row in rows:
        digest = sha256_file(row['pose_path'])
        if digest != row['pose_sha256']:
            raise ValueError(f"Pose checksum mismatch: {row['sample_id']}")
        caches.append((row['sample_id'], digest))
    source_lock = Path(__file__).resolve().parents[3]/'docs/sources.lock.json'
    if not source_lock.is_file():
        source_lock = Path(__file__).resolve().parents[1]/'sources.lock.json'
    if not source_lock.is_file():
        raise ValueError('Missing pinned source lock in the installed package')
    return {'stage': 'ssl', 'config_sha256': config_hash(config), 'manifest_sha256': sha256_file(manifest),
            'sources_lock_sha256': sha256_file(source_lock),
            'pose_inventory_sha256': hashlib.sha256(json.dumps(caches, sort_keys=True).encode()).hexdigest(),
            'score_profiles': sorted({r.get('score_profile', 'raw_probability') for r in rows}),
            'code_commit': commit, 'code_sha256': source_hash(), 'python': platform.python_version(),
            'torch': str(torch.__version__), 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'world_size': world_size(), 'best_validation_metric': None}
