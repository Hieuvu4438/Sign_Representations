"""Conservative source-ID parsing and annotation-only contracts."""
import math
import re
from urllib.parse import parse_qs, urlparse


def youtube_id(url):
    if not isinstance(url, str): return None
    try:
        p = urlparse(url)
        if p.scheme not in ('http', 'https') or p.username or p.password or p.port:
            return None
        if p.hostname in ('youtube.com', 'www.youtube.com', 'm.youtube.com') and p.path == '/watch':
            values = parse_qs(p.query, keep_blank_values=True).get('v', [])
            key = values[0] if len(values) == 1 else ''
        elif p.hostname == 'youtu.be' and len(p.path.split('/')) == 2:
            key = p.path[1:]
        else: return None
        return key if re.fullmatch(r'[A-Za-z0-9_-]{11}', key) else None
    except ValueError:
        return None


def issues(row, class_count):
    if not isinstance(row, dict): return ['not_object']
    errors = []
    if type(row.get('label')) is not int or not 0 <= row['label'] < class_count:
        errors.append('label_range_or_type')
    if type(row.get('signer_id')) is not int or row['signer_id'] < 0:
        errors.append('signer_id_range_or_type')
    if type(row.get('signer')) is not int or row['signer'] < -1:
        errors.append('legacy_signer_range_or_type')
    if youtube_id(row.get('url')) is None: errors.append('source_url_not_accepted')
    def finite_number(v): return type(v) in (int, float) and math.isfinite(v)
    start, end, fps = row.get('start_time'), row.get('end_time'), row.get('fps')
    if not all(finite_number(v) for v in (start, end, fps)) or start < 0 or end <= start or fps <= 0:
        errors.append('time_or_FPS_contract')
    if type(row.get('start')) is not int or type(row.get('end')) is not int or row['start'] < 0 or row['end'] <= row['start']:
        errors.append('frame_interval_contract')
    box = row.get('box')
    if not isinstance(box, list) or len(box) != 4 or not all(finite_number(v) for v in box) or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        errors.append('normalized_box_contract')
    if not all(finite_number(row.get(k)) and row[k] > 0 for k in ('width', 'height')):
        errors.append('dimension_contract')
    return errors
