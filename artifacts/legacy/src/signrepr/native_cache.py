"""Integrity checks for resumable native preprocessing, independent of GPU packages."""
import json
from pathlib import Path

from .io import sha256


def verified_preprocessing(audit_path, row, identity):
    audit_path = Path(audit_path)
    audit = json.loads(audit_path.read_text())
    if audit.get('status') != 'PASS_NATIVE_PREPROCESSING':
        raise ValueError('Existing failed clip needs an explicit reviewed recovery: ' + str(audit_path))
    if audit['cache_identity'] != identity or audit['sample_id'] != row['sample_id']:
        raise ValueError('Native cache identity changed')
    source_path = (Path(row['source_root']) / row['relative_path']).resolve()
    if audit['source_path'] != str(source_path) or sha256(source_path) != audit['source_sha256']:
        raise ValueError('Native input content changed')
    for name, text in {**audit['streams'], 'landmarks': audit['landmarks_path']}.items():
        path = Path(text).resolve()
        if not path.is_relative_to(audit_path.parent.resolve()):
            raise ValueError('Native cached intermediate escapes its clip folder')
        if sha256(path) != audit['stream_sha256'][name]:
            raise ValueError('Native cached intermediate checksum mismatch: ' + name)
    return audit
