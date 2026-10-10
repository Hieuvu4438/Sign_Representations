"""Metadata-only joins and prospective group partitions; no feature-array loads."""
from collections import Counter
import hashlib
from pathlib import Path


def unique(rows, key='sample_id'):
    result = {r[key]: r for r in rows}
    if not rows or len(result) != len(rows):
        raise ValueError('Empty or duplicate sample metadata')
    return result


def validate_rows(rows):
    unique(rows)
    for r in rows:
        if (r['split'] not in ('train', 'val') or type(r['class_id']) is not int
                or r['class_id'] < 0 or not r['signer'] or not r['source_group']):
            raise ValueError('Invalid split/class/group metadata; no test access')
    for field in ('signer', 'source_group', 'source_sha256'):
        if {r[field] for r in rows if r['split'] == 'train'} & {
                r[field] for r in rows if r['split'] == 'val'}:
            raise ValueError('Train/dev leakage in ' + field)


def asl_rows(cohort, manifest, cache_root, role):
    if not cohort['passed'] or manifest['status'] != 'completed':
        raise ValueError('Uncompleted cohort/cache')
    samples, entries = unique(cohort['samples']), unique(manifest['samples'])
    if set(samples) != set(entries):
        raise ValueError('Cache/cohort ID coverage differs')
    rows, files = [], set()
    for key in sorted(samples):
        s, e = samples[key], entries[key]
        for field in ('sample_id', 'class_id', 'participant_id', 'split', 'source_sha256'):
            if s[field] != e[field]:
                raise ValueError('ASL cache metadata join differs: ' + field)
        if (e['status'] != 'completed' or len(e['shape']) != 2 or e['shape'][0] < 1
                or e['shape'][1] != 768 or not 0 <= s['class_id'] < len(cohort['classes'])
                or cohort['classes'][s['class_id']] != s['gloss']):
            raise ValueError('ASL class/feature contract differs')
        name = e['file']
        if Path(name).name != name or not name.endswith('.npz') or name in files:
            raise ValueError('Unsafe or duplicate cache reference')
        files.add(name)
        rows.append(dict(role=role, sample_id=key, split=s['split'], class_id=s['class_id'],
            class_name=s['gloss'], signer=s['participant_id'], source_group=s['source_sha256'],
            source_sha256=s['source_sha256'], windows=e['shape'][0],
            feature=dict(kind='native_window_cache', path=str(Path(cache_root)/name),
                         sha256=e['sha256'], bytes=e['bytes'], shape=e['shape'])))
    if sum(r['feature']['bytes'] for r in rows) != manifest['payload_bytes']:
        raise ValueError('ASL payload accounting differs')
    validate_rows(rows)
    return rows


def msasl_rows(items, records, shards, role='MSASL100'):
    source, saved = unique(items), unique(records)
    if set(source) != set(saved) or len(shards) != len(items):
        raise ValueError('MS-ASL metadata/shard coverage differs')
    rows = []
    for key in sorted(source):
        s, e = source[key], saved[key]
        if (s['input_contract'] != 'MSASL_ESTCLK_v1' or
                e['windows'] != s['complete_stride2_windows'] or
                not e['input']['source_identity_join'] or
                e['input']['native_frames'] != s['selected_native_frames']):
            raise ValueError('MS-ASL native input join differs')
        for field in ('source_sha256', 'clock_pts_ticks_sha256', 'all_resized_RGB_sha256',
                      'first_native16_RGB_sha256'):
            if e['input'][field] != s[field]:
                raise ValueError('MS-ASL input digest differs')
        rows.append(dict(role=role, sample_id=key, split=s['split'], class_id=s['class_id'],
            class_name=None, signer=str(s['signer_id']), source_group=s['parent_id'],
            source_sha256=s['source_sha256'], windows=s['complete_stride2_windows'],
            feature=dict(kind='pooled_clip_shard', **shards[key],
                         row_sha256=e['pooled_vector_sha256'], shape=[768])))
    validate_rows(rows)
    return rows


def counts(rows, classes):
    result = dict(clips=len(rows), signers=len({r['signer'] for r in rows}),
                  sources=len({r['source_group'] for r in rows}),
                  class_counts={str(k): 0 for k in classes})
    c = Counter(r['class_id'] for r in rows)
    if set(c) - set(classes):
        raise ValueError('Out-of-vocabulary metadata')
    result['class_counts'].update({str(k): v for k, v in c.items()})
    result['missing_classes'] = [k for k in classes if not c[k]]
    return result


def hash_order(items, prefix):
    return sorted(items, key=lambda x: (hashlib.sha256((prefix+'|'+str(x)).encode()).hexdigest(), str(x)))


def calibration_partition(rows, signer_ids):
    validate_rows(rows)
    train = [r for r in rows if r['split'] == 'train']
    selected = set(signer_ids)
    available = {r['signer'] for r in train}
    if not selected or not selected < available:
        raise ValueError('Calibration groups must be a strict train-only subset')
    calibration = [r for r in train if r['signer'] in selected]
    gallery_pool = [r for r in train if r['signer'] not in selected]
    for field in ('sample_id', 'signer', 'source_group', 'source_sha256'):
        if {r[field] for r in calibration} & {r[field] for r in gallery_pool}:
            raise ValueError('Calibration/gallery leakage in ' + field)
    if sorted(r['sample_id'] for r in calibration+gallery_pool) != sorted(r['sample_id'] for r in train):
        raise ValueError('Partition discarded train clips')
    return calibration, gallery_pool


def choose_support(pool, classes, prefix):
    by_class = {k: [r['sample_id'] for r in pool if r['class_id'] == k] for k in classes}
    missing = [k for k in classes if not by_class[k]]
    # Report infeasibility rather than selecting a smaller passing vocabulary.
    if missing:
        return dict(feasible=False, missing_classes=missing, sample_ids=[])
    return dict(feasible=True, missing_classes=[],
                sample_ids=[hash_order(by_class[k],prefix+'|'+str(k))[0] for k in classes])


def nested_task(rows, support_ids, classes, class_order, sizes):
    indexed = unique(rows)
    if (len(class_order) != len(classes) or set(class_order) != set(classes)
            or len(support_ids) != len(classes) or len(set(support_ids)) != len(support_ids)
            or [indexed[k]['class_id'] for k in support_ids] != classes
            or any(indexed[k]['split'] != 'train' for k in support_ids)
            or not sizes or sizes != sorted(set(sizes)) or not all(0 < n <= len(classes) for n in sizes)):
        raise ValueError('Gallery/order/split contract differs')
    supports = dict(zip(classes, support_ids))
    return {str(n): dict(class_ids=sorted(class_order[:n]),
                        support_ids=[supports[k] for k in sorted(class_order[:n])]) for n in sizes}


def membership(rows, gallery_classes, anchor_classes, persistent_absent_classes):
    g, anchor, absent = set(gallery_classes), set(anchor_classes), set(persistent_absent_classes)
    if anchor & absent or not anchor <= g or absent & g:
        raise ValueError('Common-query membership changed')
    return dict(total=len(rows), known=sum(r['class_id'] in g for r in rows),
                unknown=sum(r['class_id'] not in g for r in rows),
                common_known_anchor=sum(r['class_id'] in anchor for r in rows),
                common_unknown=sum(r['class_id'] in absent for r in rows))
