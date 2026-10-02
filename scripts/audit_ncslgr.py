"""Audit public historical XML mirrors against the official video index."""
import argparse
import csv
import io
import json
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from _common import ROOT
from signrepr.io import sha256, write_json, write_jsonl
from signrepr.ncslgr import parse_database, event_counter


def run(first, second, index, output):
    with zipfile.ZipFile(index) as z:
        name = 'video_index-20120129/files_by_xml_file.csv'
        media_rows = list(csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig'))))
    by_key = defaultdict(list)
    for row in media_rows:
        by_key[(row['XML file'], row['Occurs in utterance id'])].append(row)
    audits, manifest = [], []
    filenames_a = {p.name for p in first.glob('*.xml')}
    filenames_b = {p.name for p in second.glob('*.xml')}
    if filenames_a != filenames_b:
        raise ValueError('Mirror file inventories differ')
    for filename in sorted(filenames_a):
        p, q = first / filename, second / filename
        a, b = parse_database(p), parse_database(q)
        ca, cb = event_counter(a), event_counter(b)
        fa, fb = event_counter(a, True), event_counter(b, True)
        only_a, only_b = ca - cb, cb - ca
        metadata_keys = ('root_attributes', 'fields', 'participants', 'media')
        meta_equal = all(a[k] == b[k] for k in metadata_keys)
        ua = sorted(((u['attributes'], u['segments'], u['media_refs']) for u in a['utterances']), key=lambda u: json.dumps(u, sort_keys=True))
        ub = sorted(((u['attributes'], u['segments'], u['media_refs']) for u in b['utterances']), key=lambda u: json.dumps(u, sort_keys=True))
        # Different annotations are preserved, never merged into invented gold.
        audit = {'xml_file': filename, 'sha256_first': sha256(p), 'sha256_second': sha256(q),
                 'byte_equal': p.read_bytes() == q.read_bytes(),
                 'newline_normalized_equal': p.read_bytes().replace(b'\r\n', b'\n') == q.read_bytes().replace(b'\r\n', b'\n'),
                 'metadata_equal': meta_equal, 'utterance_metadata_equal': ua == ub,
                 'all_annotation_events_equal': ca == cb,
                 'functional_annotation_events_equal': fa == fb,
                 'first_only_events': sum(only_a.values()),
                 'second_only_events': sum(only_b.values()),
                 'differing_fields': dict(Counter(json.loads(k)['field_name'] or 'UNKNOWN'
                                                 for c in (only_a, only_b) for k in c.elements())),
                 'event_differences': {'first_only': [json.loads(k) for k in only_a.elements()],
                                       'second_only': [json.loads(k) for k in only_b.elements()]},
                 'issues_first': a['issues'], 'issues_second': b['issues'],
                 'utterances': len(a['utterances']),
                 'functional_label_utterance_counts': dict(Counter(
                     label for u in a['utterances'] for label in u['recorded_functional_labels'])),
                 'participant_names': sorted({p['NAME'] for p in a['participants'].values()})}
        audits.append(audit)
        for u in a['utterances']:
            candidates = by_key.get((filename, u['utterance_id']), [])
            names = {m['LEGACY-PATH'].rsplit(':', 1)[-1] for m in u['media_refs']}
            mapped = [r for r in candidates if r['Video file name in XML file'] in names]
            u['official_index_rows'] = mapped
            u['index_mapping_status'] = 'EXACT_FILENAME_UTTERANCE_AND_MEDIA_NAME' if mapped else 'UNRESOLVED'
            u['functional_mirror_agreement'] = fa == fb and ua == ub and meta_equal
            u['annotation_source'] = str(p.relative_to(ROOT))
            u['annotation_sha256'] = sha256(p)
            u['public_bu_video_candidate'] = any('www.bu.edu' in r['Compressed MOV file'] for r in mapped)
            u['split'] = 'UNASSIGNED'
            manifest.append(u)
    summary = {
        'status': 'AUDIT_COMPLETE_MAIN_TRACK_NOT_OPEN', 'utc': datetime.now(timezone.utc).isoformat(),
        'first_mirror': str(first.relative_to(ROOT)), 'second_mirror': str(second.relative_to(ROOT)),
        'official_index_sha256': sha256(index), 'files': len(audits), 'utterances': len(manifest),
        'byte_equal_files': sum(a['byte_equal'] for a in audits),
        'newline_equal_files': sum(a['newline_normalized_equal'] for a in audits),
        'all_events_equal_files': sum(a['all_annotation_events_equal'] for a in audits),
        'functional_events_equal_files': sum(a['functional_annotation_events_equal'] for a in audits),
        'mapped_utterances': sum(bool(u['official_index_rows']) for u in manifest),
        'public_bu_candidate_utterances': sum(u['public_bu_video_candidate'] for u in manifest),
        'functional_label_utterance_counts': dict(Counter(l for u in manifest for l in u['recorded_functional_labels'])),
        'public_bu_functional_counts_by_recording': {
            filename: dict(Counter(l for u in manifest if u['xml_file'] == filename for l in u['recorded_functional_labels']))
            for filename in sorted({u['xml_file'] for u in manifest if u['public_bu_video_candidate']})},
        'issue_counts': dict(Counter(i['kind'] for a in audits for i in a['issues_first'])),
        'negative_label_policy': 'No confirmed absent labels from this audit. Coding scheme presence or empty tier alone does not assert absence.',
        'scope_policy': 'Keep original core intervals, HOLD/ONSET/OFFSET separately. Do not extend, clip or infer interval extents.',
        'gate_limitations': ['Historical mirror release differs in some tiers; exact official XML byte verification unavailable.',
                             'Media timing/speed and view still require decode inspection.',
                             'Public BU candidate NEG/WH/COND examples concentrated in a single recording; no independent train/val/test gate.',
                             'No matched/mismatched gold semantic candidate pairs constructed from translations.'],
        'file_audits': audits}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'report.json', summary)
    write_jsonl(output / 'utterances.jsonl', manifest)
    print(json.dumps({k: v for k, v in summary.items() if k != 'file_audits'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=Path, default=ROOT / 'third_party/asl-prosody/ncslgr-xml')
    parser.add_argument('--second', type=Path, default=ROOT / 'third_party/ASL_CCG/ncslgr-xml')
    parser.add_argument('--index', type=Path, default=ROOT / 'data/external/ncslgr_video_index.zip')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/ncslgr_annotation_audit')
    args = parser.parse_args()
    run(args.first, args.second, args.index, args.output)
