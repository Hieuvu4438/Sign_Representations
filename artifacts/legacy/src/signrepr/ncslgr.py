"""Read historical SignStream XML as annotations, without inferring negatives.

A-element timecodes are relative to UTTERANCE.S. The distributed Vogler
parser's Token constructor adds that offset. Field IDs vary between files;
only each file's coding scheme determines the meaning of an annotation.
"""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from defusedxml import ElementTree as ET

FUNCTIONAL_FIELDS = {
    'negative', 'wh question', 'yes-no question', 'conditional/when',
    'rhetorical question', 'topic/focus', 'relative clause', 'role shift',
}
PHASE_VALUES = {'400000': 'HOLD', '400001': 'ONSET', '400002': 'OFFSET'}


def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    ensure_ascii=False).encode()).hexdigest()


def inclusive_frame_bounds(start_ms, end_ms, fps):
    """Match the official parser: round endpoints and include the end frame.

    Return a Python [start, stop) frame interval. No source-boundary clipping.
    Negative relative intervals must be offset to source time before calling.
    """
    if not math.isfinite(fps) or fps <= 0 or start_ms < 0 or end_ms < start_ms:
        raise ValueError('Invalid interval or observed FPS')
    return round(start_ms / 1000.0 * fps), round(end_ms / 1000.0 * fps) + 1


def parse_database(path):
    path = Path(path)
    if path.stat().st_size > 20_000_000:
        raise ValueError('XML exceeds registered parser limit')
    root = ET.parse(path, forbid_dtd=True, forbid_entities=True,
                    forbid_external=True).getroot()
    if root.tag != 'SIGNSTREAM-DATABASE':
        raise ValueError('Not a SignStream database')
    fields = {}
    issues = []
    for f in root.findall('./CODING-SCHEME/FIELD'):
        fid = f.attrib['ID']
        if fid in fields:
            raise ValueError('Duplicate field ID')
        fields[fid] = {'attributes': dict(f.attrib),
                       'values': {v.attrib['ID']: dict(v.attrib)
                                  for v in f.findall('VALUE')}}
    participants = {p.attrib['ID']: dict(p.attrib)
                    for p in root.findall('./PARTICIPANTS/PARTICIPANT')}
    media = {m.attrib['ID']: dict(m.attrib)
             for m in root.findall('./MEDIA-FILES/MEDIA-FILE')}
    utterances = []
    seen = set()
    for u in root.findall('./UTTERANCES/UTTERANCE'):
        uid = u.attrib['ID']
        if uid in seen:
            raise ValueError('Duplicate utterance ID')
        seen.add(uid)
        start, end = int(u.attrib['S']), int(u.attrib['E'])
        if start < 0 or end <= start:
            raise ValueError('Invalid utterance interval')
        refs = [m.attrib['ID'] for m in u.findall('./MEDIA-REF')]
        events, segments = [], []
        for si, seg in enumerate(u.findall('./SEGMENT')):
            participant = participants.get(seg.attrib['PARTICIPANT-ID'])
            if participant is None:
                raise ValueError('Unknown participant ID')
            segments.append(dict(seg.attrib))
            for ti, tr in enumerate(seg.findall('./TRACK')):
                fid = tr.attrib['FID']
                field = fields.get(fid)
                if field is None:
                    issues.append({'utterance_id': uid, 'kind': 'unknown_field',
                                   'fid': fid, 'event_count': len(tr.findall('A'))})
                name = field['attributes']['NAME'] if field else None
                for ai, a in enumerate(tr.findall('A')):
                    s, e = int(a.attrib['S']), int(a.attrib['E'])
                    vid = a.attrib.get('VID')
                    value = field['values'].get(vid) if field else None
                    if vid is not None and field is not None and value is None:
                        issues.append({'utterance_id': uid, 'kind': 'unknown_value',
                                       'fid': fid, 'vid': vid})
                    event = {
                        'event_id': f'{path.stem}:{uid}:{si}:{ti}:{ai}',
                        'segment_index': si, 'participant_id': participant['ID'],
                        'participant_name': participant['NAME'], 'fid': fid,
                        'field_name': name, 'vid': vid,
                        'value_name': value.get('NAME') if value else None,
                        'value_label': value.get('LABEL') if value else None,
                        'text': ''.join(a.itertext()), 'relative_start_ms': s,
                        'relative_end_ms': e, 'absolute_start_ms': start + s,
                        'absolute_end_ms': start + e,
                        'is_functional': name in FUNCTIONAL_FIELDS,
                        'phase': PHASE_VALUES.get(vid),
                        'interval_attributes': dict(a.attrib),
                    }
                    events.append(event)
                    if e < s or start + s < 0:
                        issues.append({'utterance_id': uid, 'kind': 'invalid_event_interval',
                                       'event_id': event['event_id']})
                    if s < 0:
                        issues.append({'utterance_id': uid, 'kind': 'event_before_utterance',
                                       'event_id': event['event_id'], 'excess_ms': -s,
                                       'is_functional': event['is_functional']})
                    if e > end - start:
                        issues.append({'utterance_id': uid, 'kind': 'event_past_utterance',
                                       'event_id': event['event_id'],
                                       'excess_ms': e - (end - start),
                                       'is_functional': event['is_functional']})
        known_media = []
        for mid in refs:
            if mid not in media:
                raise ValueError('Unknown media ID')
            known_media.append(media[mid])
        positive = [e for e in events if e['is_functional'] and e['phase'] is None
                    and (e['value_name'] or e['text'].strip())]
        utterances.append({
            'sample_id': f'ncslgr:{path.stem}:{uid}', 'xml_file': path.name,
            'utterance_id': uid, 'start_ms': start, 'end_ms': end,
            'attributes': dict(u.attrib), 'segments': segments,
            'participant_names': sorted({e['participant_name'] for e in events}),
            'media_refs': known_media, 'events': events,
            'functional_core_intervals': positive,
            'recorded_functional_labels': sorted({
                f"{e['field_name']}::{e['value_name'] or e['text'].strip()}"
                for e in positive}),
            'negative_labels': None,
            'absence_status': 'NOT_CONFIRMED_BY_ANNOTATION_AUDIT',
            'recording_group': f'ncslgr:xml:{path.stem}',
            'recording_group_status': 'CONSERVATIVE_XML_FILE_GROUP_PENDING_MEDIA_AUDIT',
        })
    return {'root_attributes': dict(root.attrib), 'fields': fields,
            'participants': participants, 'media': media,
            'utterances': utterances, 'issues': issues}


def event_counter(database, functional_only=False):
    """Compare all annotation content, ignoring track/list order only."""
    result = Counter()
    for u in database['utterances']:
        for event in u['events']:
            if functional_only and not event['is_functional']:
                continue
            # Index-derived IDs change when a mirror inserts a token. All
            # meaning-bearing source values and timings remain in this key.
            content = {k: v for k, v in event.items() if k != 'event_id'}
            result[json.dumps({'utterance_id': u['utterance_id'], **content},
                              sort_keys=True, ensure_ascii=False)] += 1
    return result
