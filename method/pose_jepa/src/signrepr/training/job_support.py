"""Small atomic files and checkpoint rotation for long-running, single-GPU jobs."""
import json
import os
from pathlib import Path
import tempfile
from .checkpoints import save_training, load_training
from ..data.schema import sha256_file


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def durable_checkpoint(trainer, stream, root, metadata):
    """Retain a second complete checkpoint; each .pt replacement is atomic."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    last, previous = root/'last.pt', root/'previous.pt'
    receipt = root/'checkpoint.json'
    # Never overwrite a valid previous checkpoint with a corrupted latest file
    # after resume_latest recovered from the previous copy.
    rotate = last.exists() and receipt.exists() and json.loads(receipt.read_text()).get('sha256') == sha256_file(last)
    if rotate:
        temp = root/'.previous.link'
        if temp.exists(): temp.unlink()
        os.link(last, temp); os.replace(temp, previous)
    save_training(trainer, last, stream.state_dict(), metadata)
    atomic_json(receipt, {'step': trainer.step, 'sha256': sha256_file(last),
                                        'last': str(last), 'previous': str(previous) if previous.exists() else None})


def resume_latest(trainer, stream, root, metadata):
    """Fall back for unreadable files, but never bypass scientific/config mismatch."""
    import torch
    import pickle
    root = Path(root)
    errors = []
    for name in ('last.pt', 'previous.pt'):
        path = root/name
        if not path.exists(): continue
        try:
            state = torch.load(path, map_location='cpu', weights_only=False)
        except (OSError, EOFError, RuntimeError, pickle.UnpicklingError) as exc:
            errors.append(f'{name}: {exc}'); continue
        previous = state['metadata']
        for key in ('manifest_sha256', 'pose_inventory_sha256', 'code_sha256', 'sources_lock_sha256', 'selection_protocol_sha256',
                    'selection_pose_inventory_sha256'):
            if previous.get(key) != metadata.get(key):
                raise ValueError(f'Resume provenance mismatch: {key}')
        del state
        cursor, restored = load_training(trainer, path)
        stream.load_state_dict(cursor)
        return restored, {'checkpoint': str(path), 'step': trainer.step, 'fallback_errors': errors}
    if errors:
        raise RuntimeError('No readable recovery checkpoint: '+'; '.join(errors))
    return metadata, None
