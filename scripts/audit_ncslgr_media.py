"""Fully decode the bounded readiness cohort and audit source timing/views.

This verifies container timing and paired frame counts. It does not certify
linguistic video alignment, human identity, camera semantics or playback speed
from movement. Those limitations stay explicit in the report.
"""
import concurrent.futures
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from _common import ROOT
from signrepr.io import read_jsonl, sha256, write_json, write_jsonl
from signrepr.ncslgr import inclusive_frame_bounds


def main():
    cohort = ROOT / 'data/external/ncslgr/readiness_cohort.jsonl'
    rows = [r for _, r in read_jsonl(cohort)]
    heads = json.loads((ROOT / 'evidence/ncslgr/cohort_heads.json').read_text())
    registry = json.loads((ROOT / 'provenance/assets.json').read_text())
    if heads['cohort_sha256'] != sha256(cohort):
        raise ValueError('HEAD/cohort mismatch')
    def decode(key):
        r = heads['results'][key]
        asset = registry['assets'].get(r['asset_id'])
        if asset is None:
            return {'status': 'FAIL', 'reason': 'ASSET_NOT_REGISTERED'}
        path = ROOT / asset['local_path']
        try:
            provenance = json.loads((ROOT / 'provenance' / (r['asset_id'] + '.json')).read_text())
            actual_hash = sha256(path)
            if actual_hash != provenance['sha256'] or path.stat().st_size != asset['expected_bytes']:
                raise ValueError('Source bytes differ from acquisition provenance')
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError('Decode open failed')
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            count = 0
            rgb_hash = hashlib.sha256()
            try:
                while True:
                    valid, bgr = cap.read()
                    if not valid:
                        break
                    if bgr.shape != (height, width, 3):
                        raise ValueError('Frame geometry changed')
                    rgb_hash.update(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes())
                    count += 1
            finally:
                cap.release()
            if not np.isfinite(fps) or fps <= 0 or count != declared or count == 0:
                raise ValueError('Invalid FPS, empty or incomplete decode')
            packets = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                'stream=r_frame_rate,avg_frame_rate,time_base,duration,nb_frames:packet=pts_time,duration_time',
                '-of', 'json', str(path)]))
            pts = np.asarray([float(p['pts_time']) for p in packets['packets']])
            if len(pts) != count:
                raise ValueError('Packet/frame count mismatch; RPZA clock cannot be assumed')
            nominal_residual = float(np.max(np.abs(pts - np.arange(count) / 30)))
            nominal_clock_verified = nominal_residual <= 1e-6
            return {'status': 'PASS', 'asset_id': r['asset_id'], 'path': asset['local_path'],
                    'source_sha256': actual_hash, 'decoded_rgb_sha256': rgb_hash.hexdigest(),
                    'fps': fps, 'frames': count, 'width': width, 'height': height,
                    'duration_sec': float(packets['streams'][0]['duration']),
                    'nominal_30fps_pts_verified': nominal_clock_verified,
                    'max_packet_pts_residual_sec': nominal_residual,
                    'packet_pts_sha256': hashlib.sha256(pts.tobytes()).hexdigest(),
                    'ffprobe_stream': packets['streams'][0],
                    'annotation_frame_clock_fps': 30.0 if nominal_clock_verified else None}
        except Exception as error:
            return {'status': 'FAIL', 'asset_id': r['asset_id'],
                    'reason': type(error).__name__ + ': ' + str(error)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        decoded = dict(zip(heads['results'], pool.map(decode, heads['results'])))
    samples, failures = [], []
    for row in rows:
        views = []
        for view in row['views']:
            key = hashlib.sha256(view['Compressed MOV file'].encode()).hexdigest()
            views.append({'view_index': view['Perspective/Camera id'], **decoded[key]})
        reasons = []
        if any(v['status'] != 'PASS' for v in views):
            reasons.append('DECODE_FAILURE')
        else:
            # Do not silently rescale half-speed video using a guessed factor.
            if not all(v['nominal_30fps_pts_verified'] for v in views):
                reasons.append('PACKET_PTS_DO_NOT_ESTABLISH_NOMINAL_30FPS_CLOCK')
            if len({v['frames'] for v in views}) != 1 or len({v['packet_pts_sha256'] for v in views}) != 1:
                reasons.append('PAIRED_STREAM_CLOCK_MISMATCH')
            for v in views:
                if row['end_ms'] / 1000 > v['duration_sec'] + 1e-3:
                    reasons.append('UTTERANCE_AFTER_SOURCE_END')
                for event in row['functional_core_intervals']:
                    start, stop = inclusive_frame_bounds(event['absolute_start_ms'],
                                                        event['absolute_end_ms'], 30.0)
                    if start < 0 or stop > v['frames']:
                        reasons.append('CORE_INTERVAL_OUTSIDE_DECODED_FRAMES')
        sample = {**row, 'decoded_views': views,
                  'media_audit_status': 'PASS_METADATA_TIMING' if not reasons else 'NEEDS_REVIEW',
                  'media_audit_reasons': sorted(set(reasons)),
                  'linguistic_alignment_human_review': 'NOT_PERFORMED',
                  'view_semantics': 'PENDING_SESSION_SPECIFIC_VISUAL_AUDIT'}
        samples.append(sample)
        if reasons:
            failures.append({'sample_id': row['sample_id'], 'reasons': sorted(set(reasons))})
    by_hash = defaultdict(set)
    by_xml = defaultdict(set)
    for row in samples:
        by_xml[row['recording_group']].add(row['split'])
        for view in row['decoded_views']:
            if view['status'] == 'PASS':
                by_hash[view['decoded_rgb_sha256']].add(row['split'])
    hash_overlaps = {k: sorted(v) for k, v in by_hash.items() if len(v) > 1}
    xml_overlaps = {k: sorted(v) for k, v in by_xml.items() if len(v) > 1}
    report = {'status': 'PASS_METADATA_WITH_LIMITS' if not failures and not hash_overlaps else 'NEEDS_REVIEW',
              'cohort_sha256': sha256(cohort), 'utterances': len(rows),
              'unique_videos': len(decoded), 'decoded_pass': sum(v['status'] == 'PASS' for v in decoded.values()),
              'metadata_timing_pass': len(samples) - len(failures), 'failures': failures,
              'fps_counts': dict(Counter(str(v.get('fps')) for v in decoded.values())),
              'nominal_30fps_packet_pts_verified_videos': sum(v.get('nominal_30fps_pts_verified', False) for v in decoded.values()),
              'clock_policy': 'Verify every packet PTS against index/30 seconds. Average FPS slightly exceeds30 when only the final packet duration is shorter; no playback-speed rescaling is applied. Frame annotation clock is30 only after exact packet-PTS verification.',
              'frame_geometry_counts': dict(Counter(f"{v.get('width')}x{v.get('height')}" for v in decoded.values())),
              'exact_decoded_video_cross_split_overlaps': hash_overlaps,
              'conservative_xml_group_cross_split_overlaps': xml_overlaps,
              'limitations': ['Frame-rate/endpoint checks do not certify playback speed or semantic alignment.',
                              'Mirror annotation release remains historical, not certified as2011 updated release.',
                              'If retaining XML-file grouping, shared XML groups must be excluded before protocol lock.',
                              'Camera IDs need session-specific view validation. No inference or metric selection performed.'],
              'videos': decoded}
    write_json(ROOT / 'reports/ncslgr_media_audit.json', report)
    write_jsonl(ROOT / 'data/external/ncslgr/readiness_decoded.jsonl', samples)
    print(json.dumps({k: v for k, v in report.items() if k != 'videos'}, indent=2))


if __name__ == '__main__':
    main()
