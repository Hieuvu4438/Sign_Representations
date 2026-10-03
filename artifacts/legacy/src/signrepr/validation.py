import math
from collections import Counter, defaultdict
from pathlib import Path

from signrepr.manifests import FIELDS


def validate_rows(rows):
    errors, warnings, counts = [], [], Counter()
    seen = set()
    for number, row in enumerate(rows, 1):
        sid = row.get('sample_id')
        for field in FIELDS:
            if field not in row:
                errors.append({'line': number, 'sample_id': sid, 'reason': 'MISSING_FIELD', 'field': field})
        if not isinstance(sid, str) or not sid:
            errors.append({'line': number, 'reason': 'INVALID_ID'})
        if sid in seen:
            errors.append({'line': number, 'sample_id': sid, 'reason': 'DUPLICATE_ID'})
        seen.add(sid)
        counts['rows'] += 1
        try:
            root = Path(row['source_root']).resolve()
            relative = Path(row['relative_path'])
            source = (root / relative).resolve()
            if relative.is_absolute() or not source.is_relative_to(root):
                errors.append({'sample_id': sid, 'reason': 'PATH_ESCAPES_SOURCE_ROOT'})
            elif not source.exists():
                counts['missing_source'] += 1
            elif row.get('input_kind') == 'frames_directory' and not source.is_dir():
                errors.append({'sample_id': sid, 'reason': 'EXPECTED_FRAMES_DIRECTORY'})
            elif row.get('input_kind') == 'video' and not source.is_file():
                errors.append({'sample_id': sid, 'reason': 'EXPECTED_VIDEO_FILE'})
        except (KeyError, TypeError, OSError) as error:
            errors.append({'sample_id': sid, 'reason': 'INVALID_PATH', 'error': str(error)})
        start, end, duration = row.get('start_sec'), row.get('end_sec'), row.get('duration_sec')
        if (start is None) != (end is None):
            errors.append({'sample_id': sid, 'reason': 'PARTIAL_INTERVAL'})
        elif start is not None:
            if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in [start, end]) or not 0 <= start < end:
                errors.append({'sample_id': sid, 'reason': 'INVALID_INTERVAL'})
            elif duration is not None and end > duration + 1e-6:
                errors.append({'sample_id': sid, 'reason': 'INTERVAL_OUTSIDE_DURATION'})
        if duration is not None and (not math.isfinite(duration) or duration <= 0):
            errors.append({'sample_id': sid, 'reason': 'INVALID_DURATION'})
        if row.get('official_split') not in {'train', 'val', 'test'}:
            errors.append({'sample_id': sid, 'reason': 'INVALID_OFFICIAL_SPLIT'})
        if row.get('annotation_quality') not in {'GOLD_EXISTING_LEXICAL_OR_SENTENCE', 'GOLD_EXISTING', 'HUMAN_VERIFIED', 'WEAK_TEXT_DERIVED', 'UNKNOWN'}:
            errors.append({'sample_id': sid, 'reason': 'INVALID_ANNOTATION_STATUS'})
        if row.get('grouping_status') == 'SOURCE_RECORDING_UNKNOWN':
            counts['unresolved_recording_group'] += 1
        if row.get('signer_id') is None:
            counts['unknown_signer'] += 1
        if row.get('decode_status') == 'NOT_CHECKED':
            counts['decode_not_checked'] += 1
    for key in ['missing_source', 'unresolved_recording_group', 'unknown_signer', 'decode_not_checked']:
        if counts[key]:
            warnings.append({'reason': key, 'count': counts[key]})
    return {'status': 'FAIL' if errors else 'PASS_SCHEMA', 'counts': dict(counts), 'errors': errors, 'warnings': warnings,
            'scope': 'Schema/source existence checks; linguistic validity and full decode are not established.'}


def audit_rows(rows):
    groups, signers, hashes = defaultdict(lambda: defaultdict(list)), defaultdict(set), defaultdict(set)
    counts, unresolved = Counter(), Counter()
    for row in rows:
        split = row.get('research_split') or row['official_split']
        counts[(row['dataset_id'], split)] += 1
        groups[row['recording_group']][split].append(row['sample_id'])
        if row.get('signer_id') is not None:
            signers[(row['dataset_id'], str(row['signer_id']))].add(split)
        if row.get('content_hash'):
            hashes[row['content_hash']].add(split)
        if row.get('grouping_status') == 'SOURCE_RECORDING_UNKNOWN':
            unresolved[row['dataset_id']] += 1
    overlaps = [{'recording_group': group, 'samples_by_split': dict(splits)} for group, splits in groups.items() if len(splits) > 1]
    signer_overlap = [{'dataset_id': dataset, 'signer_id': signer, 'splits': sorted(splits)} for (dataset, signer), splits in signers.items() if len(splits) > 1]
    return {'status': 'FAIL_RECORDING_OVERLAP' if overlaps else 'UNVERIFIED_GROUPS' if unresolved else 'PASS_METADATA_GROUPING',
            'sample_counts': [{'dataset_id': dataset, 'split': split, 'rows': count} for (dataset, split), count in sorted(counts.items())],
            'recording_group_count': len(groups), 'recording_overlaps': overlaps,
            'signer_count': len(signers), 'signer_overlaps': signer_overlap,
            'unresolved_recording_counts': dict(unresolved),
            'exact_hash_overlaps': [value for value, splits in hashes.items() if len(splits) > 1],
            'content_hash_coverage': sum(bool(row.get('content_hash')) for row in rows),
            'perceptual_duplicate_audit': 'NOT_RUN', 'pretraining_overlap': 'UNKNOWN',
            'scope': 'Metadata group audit; absent metadata overlap does not establish absent pretraining or visual duplication.'}
