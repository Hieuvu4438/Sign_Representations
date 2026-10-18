"""Manifest integrity, source/hash contamination and auditable pose duplicate checks."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
from .schema import read_manifest, load_pose, sha256_file, write_manifest
from .normalization import canonicalize_sample


def pose_signature(pose):
    sample = canonicalize_sample(pose)
    if sample['status'] != 'ok':
        return None
    # Fixed 16 temporal bins, ordered global selected joints and valid support.
    indices = np.linspace(0, sample['length']-1, 16).round().astype(int)
    xy = np.concatenate([s['global_xy'].numpy()[indices] for s in sample['parts'].values()], 1)
    valid = np.concatenate([s['joint_valid'].numpy()[indices] for s in sample['parts'].values()], 1)
    return xy, valid


def validate_manifest(path, near_duplicate_threshold=.01):
    errors, hashes, ids, source_groups, signatures = [], {}, {}, {}, []
    path = Path(path).resolve()
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip(): continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict): raise ValueError('Manifest row must be a JSON object')
            if isinstance(row.get('pose_path'), str):
                row['pose_path'] = str((path.parent/row['pose_path']).resolve())
            rows.append(row)
        except (ValueError, TypeError) as exc:
            errors.append({'line': line_number, 'reason': str(exc)})
    coverage = Counter()
    near_duplicates = []
    required = {'sample_id', 'corpus', 'language', 'split', 'source_video_id', 'pose_path', 'num_frames',
                'joint_layout', 'coordinate_system', 'extractor_profile', 'timing_source', 'pose_sha256'}
    for row in rows:
        sid = row.get('sample_id', '<missing>')
        try:
            if required-row.keys():
                raise ValueError(f'Missing manifest fields: {sorted(required-row.keys())}')
            for key in ('sample_id', 'corpus', 'language', 'split', 'source_video_id', 'pose_path', 'pose_sha256'):
                if not isinstance(row[key], str) or not row[key]:
                    raise ValueError(f'Manifest field {key} must be a nonempty string')
            if sid in ids:
                raise ValueError('Duplicate sample ID')
            ids[sid] = row
            pose = load_pose(row['pose_path'])
            if row['num_frames'] != len(pose['xy']):
                raise ValueError('Manifest/cache frame count mismatch')
            digest = sha256_file(row['pose_path'])
            if digest != row['pose_sha256']:
                raise ValueError('Pose checksum mismatch')
            if row['joint_layout'] != 'coco_wholebody_133' or row['coordinate_system'] != 'viewport_normalized':
                raise ValueError('Invalid pose convention/profile')
            if digest in hashes:
                errors.append({'sample_id': sid, 'reason': 'exact_pose_duplicate', 'other': hashes[digest]})
            hashes[digest] = sid
            key = (row['corpus'], row['source_video_id'])
            source_groups.setdefault(key, []).append(row)
            signature = pose_signature(pose)
            # Exact brute-force audit for small/pilot manifests; report candidates.
            if signature is not None:
                for other, sig in signatures:
                    support = signature[1] & sig[1]
                    if support.mean() >= .5:
                        rms = float(np.sqrt(np.mean((signature[0][support]-sig[0][support])**2)))
                        if rms < near_duplicate_threshold:
                            near_duplicates.append({'sample_id': sid, 'other': other, 'rms_canonical_xy': rms})
                signatures.append((sid, signature))
            canonical = canonicalize_sample(pose, row)
            coverage['samples'] += 1
            coverage['known_timing'] += int(bool(pose['timing_known']))
            coverage['normalization_failed'] += int(canonical['status'] != 'ok')
            for part, stream in canonical['parts'].items():
                coverage[f'{part}_observed_frames'] += int(stream['frame_valid'].sum())
            coverage['frames'] += len(pose['xy'])
        except (ValueError, OSError, KeyError) as exc:
            errors.append({'sample_id': sid, 'reason': str(exc)})
    for group in source_groups.values():
        if len({row['split'] for row in group}) > 1:
            errors.append({'reason': 'source_video_cross_split', 'samples': [row['sample_id'] for row in group]})
    return {'valid': not errors, 'errors': errors, 'near_duplicate_candidates': near_duplicates,
            'near_duplicate_policy': 'review or exclude training candidates before real runs',
            'coverage': dict(coverage), 'manifest_sha256': sha256_file(path)}


def build_protocol(config, output):
    """Freeze official splits, remove identified training contaminants, preserve eval denominator."""
    policy = config['protocol']
    if policy not in ('in_domain_ssl', 'cross_corpus_transfer'):
        raise ValueError('Unknown training protocol')
    rows = read_manifest(config['manifest'])
    target = set(config.get('target_corpora', []))
    if policy == 'cross_corpus_transfer' and not target:
        raise ValueError('Cross-corpus protocol requires target corpus IDs')
    protected = [r for r in rows if r['split'] in ('val', 'validation', 'test') or (policy == 'cross_corpus_transfer' and r['corpus'] in target)]
    keys = {(r['corpus'], r['source_video_id']) for r in protected}
    hashes = {r['pose_sha256'] for r in protected}
    video_hashes = {r.get('video_sha256') for r in protected if r.get('video_sha256')}
    urls = {r.get('source_url') for r in protected if r.get('source_url')}
    exclude_ids = set(config.get('near_duplicate_exclusions', []))
    train, excluded = [], []
    for r in rows:
        if r['split'] not in ('train', 'pretrain'):
            continue
        reasons = []
        if policy == 'cross_corpus_transfer' and r['corpus'] in target:
            reasons.append('target_corpus')
        if (r['corpus'], r['source_video_id']) in keys or r['pose_sha256'] in hashes:
            reasons.append('protected_source_or_pose_hash')
        if r.get('video_sha256') in video_hashes or r.get('source_url') in urls:
            reasons.append('protected_video_hash_or_url')
        if r['sample_id'] in exclude_ids:
            reasons.append('reviewed_near_duplicate')
        if reasons:
            excluded.append({'sample_id': r['sample_id'], 'reasons': reasons})
        else:
            train.append(r)
    if not train:
        raise ValueError('No eligible SSL training samples after contamination screen')
    output = Path(output)
    write_manifest(output/'ssl_train.jsonl', train)
    for split, filename in [('train', 'gallery'), ('val', 'validation_queries'), ('test', 'query')]:
        write_manifest(output/f'{filename}.jsonl', [r for r in rows if r['split'] == split and r['corpus'] in target])
    report = {'protocol': policy, 'model_selection_scope': config['model_selection_scope'], 'excluded': excluded,
              'ssl_samples': len(train), 'source_manifest_sha256': sha256_file(config['manifest'])}
    (output/'protocol.json').write_text(json.dumps(report, indent=2))
    return report
