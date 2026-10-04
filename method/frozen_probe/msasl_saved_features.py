"""Read complete, immutable pooled-feature shards without scoring or refitting."""
import hashlib
import json
from pathlib import Path
import numpy as np

from bounded_artifacts import verify_store
from cache import file_sha


def load_complete_features(root, items):
    root = Path(root)
    c = json.loads((root / 'config.json').read_text())
    s = json.loads((root / 'status.json').read_text())
    if s['status'] != 'completed' or s['exit_code'] != 0:
        raise ValueError('Incomplete feature run; no partial-cohort scoring')
    checked = verify_store(root)
    if file_sha(root / 'config.json') != s['config_sha256']:
        raise ValueError('Feature config identity differs')
    manifest = json.loads((root / 'artifact_manifest.json').read_text())
    if manifest['plan']['file_caps'] != {**c['artifact_file_caps'], 'artifact_manifest.json': c['artifact_manifest_cap']}:
        raise ValueError('Feature store plan differs from registered config')
    for name, digest in s['output_hashes'].items():
        if file_sha(root / name) != digest:
            raise ValueError('Feature output binding differs')
    ids = [x['sample_id'] for x in items]
    if ids != c['sample_ids'] or len(ids) != 1037 or ids != sorted(set(ids)):
        raise ValueError('Feature cohort differs from complete frozen dictionary')
    values, records, next_start = [], [], 0
    new_windows, reused_windows, reused_ids = 0, 0, []
    for shard in c['shards']:
        if shard['start'] != next_start:
            raise ValueError('Noncontiguous feature shard')
        a = np.load(root / shard['array_name'], allow_pickle=False)
        rows = json.loads((root / shard['records_name']).read_text())
        part = items[shard['start']:shard['stop']]
        if a.dtype != np.float64 or a.shape != (len(part), 768) or len(rows) != len(part):
            raise ValueError('Shard shape/dtype/record count mismatch')
        if not np.isfinite(a).all() or np.max(np.abs(np.linalg.norm(a, axis=1) - 1)) > 1e-12:
            raise ValueError('Invalid pooled vectors')
        for row, item, vector in zip(rows, part, a):
            if row['sample_id'] != item['sample_id'] or row['windows'] != item['complete_stride2_windows']:
                raise ValueError('Shard ID/window mismatch')
            for field in ('source_sha256', 'clock_pts_ticks_sha256', 'all_resized_RGB_sha256', 'first_native16_RGB_sha256'):
                if row['input'][field] != item[field]:
                    raise ValueError('Saved native input hash differs')
            if not row['input']['source_identity_join'] or row['input']['native_frames'] != item['selected_native_frames']:
                raise ValueError('Saved source/native count differs')
            if row['pooled_vector_sha256'] != hashlib.sha256(vector.tobytes()).hexdigest():
                raise ValueError('Pooled row hash differs')
            if row['reused_from'] is None:
                if row['new_encoder_windows'] != row['windows'] or row['independent_pool_max_abs_error'] > 1e-12:
                    raise ValueError('New extraction/pooling contract differs')
                new_windows += row['windows']
            else:
                if row['reused_from'] != c['cost_runtime'] or row['new_encoder_windows'] != 0:
                    raise ValueError('Unregistered feature reuse')
                reused_windows += row['windows']; reused_ids.append(row['sample_id'])
        values.append(a); records.extend(rows); next_start = shard['stop']
    summary = json.loads((root / 'extraction_summary.json').read_text())
    cost_rows = json.loads((Path(c['cost_runtime']) / 'feature_cost_records.json').read_text())
    if (next_start != 1037 or len(c['shards']) != 42 or new_windows != 36028 or reused_windows != 396
            or sorted(reused_ids) != sorted(x['sample_id'] for x in cost_rows)
            or summary['clips'] != 1037 or summary['new_encoder_windows'] != new_windows
            or summary['reused_encoder_windows'] != reused_windows
            or summary['new_encoder_forward_calls'] != new_windows + 1):
        raise ValueError('Incomplete full extraction accounting')
    return np.concatenate(values), records, dict(store=checked, clips=1037, shards=42,
        new_encoder_windows=new_windows, reused_windows=reused_windows, reused_ids=sorted(reused_ids),
        scope='Saved hash/count/shape/provenance checks; no new decode, raw-new-window replay or task quality')
