"""Complete a failed P004 input audit without inference, exclusions or padding."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import file_sha
from confirmatory import query_map
from run_context import RunContext
from video_contract import dhash64, fingerprint_indices, summarize_timeline
from scripts.screen_duplicates import compare_pair


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import cv2
        import numpy as np
        cv2.setNumThreads(1)
        for name, digest in config['input_hashes'].items():
            if file_sha(REPO / name) != digest:
                raise ValueError('Bound failed-attempt/input artifact changed')
        test = json.loads((REPO / config['test_manifest']).read_text())
        cohort = json.loads((REPO / config['cohort_manifest']).read_text())
        known = json.loads((REPO / config['video_audit']).read_text())
        indexed = query_map(cohort, test)
        if len(indexed) != config['expected_queries']:
            raise ValueError('Complete selected test census required')
        logical = sum(s['source_bytes'] * 3 for s in indexed.values())
        if logical > config['max_logical_source_read_bytes']:
            raise ValueError('Registered CPU input-census read limit exceeded')
        audits, violations = [], []
        for number, key in enumerate(sorted(indexed), 1):
            source = indexed[key]
            path = Path(source['input_path'])
            if path.stat().st_size != source['source_bytes'] or file_sha(path) != source['source_sha256']:
                raise ValueError('Source bytes changed after original failed audit')
            response = subprocess.run(['ffprobe', '-v', 'error', '-threads', '1', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,avg_frame_rate,r_frame_rate,duration:frame=best_effort_timestamp_time',
                '-of', 'json', str(path)], capture_output=True, text=True, check=True, timeout=30)
            if response.stderr.strip():
                raise ValueError('ffprobe decode error')
            native = json.loads(response.stdout)
            pts = [float(f['best_effort_timestamp_time']) for f in native['frames']]
            if (not pts or len(pts) > config['max_frames_per_video']
                    or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0)):
                raise ValueError('Invalid native timestamps or registered source audit ceiling exceeded')
            stream = native['streams'][0]
            indices = fingerprint_indices(len(pts), [0, .25, .5, .75, 1])
            cap = cv2.VideoCapture(str(path))
            fingerprints, count = {}, 0
            try:
                if not cap.isOpened():
                    raise ValueError('OpenCV source unavailable')
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame.shape != (stream['height'], stream['width'], 3):
                        raise ValueError('Native dimensions changed')
                    if count in indices:
                        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
                        fingerprints[count] = dict(dhash64=dhash64(gray), gray_std=float(gray.std()),
                                                   gray_range=int(gray.max())-int(gray.min()))
                    count += 1
            finally:
                cap.release()
            if count != len(pts):
                raise ValueError('Decoded frame count/PTS mismatch')
            eligible = count >= 16
            if eligible:
                timeline = summarize_timeline(pts, count, 16, 2)
            else:
                timeline = dict(frames=count, windows=0,
                                duration_s=float(stream.get('duration', pts[-1]-pts[0])),
                                incomplete_tail_frames=count)
                violations.append(dict(sample_id=key, frames=count, reason='fewer than16nativeframes; no complete frozen window'))
            audits.append({k: source[k] for k in ('sample_id', 'split', 'class_id', 'participant_id', 'source_sha256')} |
                          timeline | dict(pts_s=pts, stream=stream, fingerprint_frame_indices=indices,
                          fingerprints=[fingerprints[i] for i in indices],
                          source_integrity_passed=True, native_recipe_eligible=eligible))
            if number % 100 == 0 or number == len(indexed):
                run.save('progress.json', dict(audited=number, expected=len(indexed), native_contract_violations=violations,
                                              predictions_computed=False))
                print(json.dumps(dict(audited=number, expected=len(indexed), contract_violations=len(violations))), flush=True)
        # Short rows still count as selected queries; zero windows is a contract fault,
        # never a zero vector, prediction, test exclusion, or an accuracy denominator.
        candidates, pairs, eligible_pairs = [], 0, 0
        for source in audits:
            for previous in known['samples']:
                pairs += 1
                if len(source['fingerprints']) != 5 or len(previous['fingerprints']) != 5:
                    continue
                pair = compare_pair(source, previous)
                if pair is None:
                    continue
                eligible_pairs += 1
                if pair['candidate']:
                    candidates.append(dict(test_id=source['sample_id'], known_id=previous['sample_id'], **pair))
        run.save('source_census.json', dict(samples=audits, N_expected=len(indexed), N_audited=len(audits),
                 source_integrity_passed=True, native_recipe_eligible=not violations,
                 contract_violations=violations, frames=sum(s['frames'] for s in audits),
                 complete_native_windows=sum(s['windows'] for s in audits),
                 participant_counts=dict(Counter(s['participant_id'] for s in audits)),
                 selected_query_exclusions=0, predictions_computed=False, GPU_hours=0,
                 scope='post-failure input-contract census only; no valid all-query P004 result or changed recipe',
                 all_source_read_bytes_planned=logical))
        run.save('duplicate_screen.json', dict(cross_split_pairs_tested=pairs, eligible_pairs=eligible_pairs,
                 candidates=candidates, fewer_than_five_fingerprint_sources=[s['sample_id'] for s in audits if len(s['fingerprints']) != 5],
                 scope='bounded thumbnail/length screen, not exhaustive uniqueness/session/pretraining proof'))
        run.status['output_hashes'] = {name:file_sha(run.output/name) for name in ('source_census.json','duplicate_screen.json')}


if __name__ == '__main__':
    main()
