"""First selected-test access after P004 code/config/analysis freeze; CPU only."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import file_sha
from confirmatory import query_map
from replication_test import check_test_freeze as check_freeze
from run_context import RunContext
from scripts.screen_duplicates import compare_pair
from video_contract import dhash64, fingerprint_indices, summarize_timeline


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'freeze', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    freeze = check_freeze(args.freeze, args.config)
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import cv2
        cv2.setNumThreads(1)
        cohort = json.loads((REPO / config['cohort_manifest']).read_text())
        old_audit = json.loads((REPO / config['video_audit']).read_text())
        classes = {gloss: c for c, gloss in enumerate(cohort['classes'])}
        metadata = {}
        for split in ('train', 'val', 'test'):
            path = Path(config['dataset']) / 'splits' / f'{split}.csv'
            if file_sha(path) != config['split_hashes'][split]:
                raise ValueError('Official split CSV changed')
            with path.open(newline='') as handle:
                metadata[split] = list(csv.DictReader(handle))
        test_rows = [r for r in metadata['test'] if r['Gloss'] in classes]
        if not 1 <= len(test_rows) <= config['max_queries']:
            raise ValueError('All selected test rows exceed registered audit ceiling')
        all_known_ids = {r['Video file'] for split in ('train', 'val') for r in metadata[split]}
        known_people = {r['Participant ID'] for split in ('train', 'val') for r in metadata[split]}
        samples = []
        planned = 0
        for row in sorted(test_rows, key=lambda r: r['Video file']):
            key = row['Video file']
            if Path(key).name != key or key in all_known_ids or row['Participant ID'] in known_people:
                raise ValueError('Official test file/participant leakage or unsafe input path')
            path = Path(config['dataset']) / 'videos' / key
            planned += 3 * path.stat().st_size
            if planned > config['max_logical_source_read_bytes']:
                raise ValueError('Registered full audit source-read ceiling exceeded')
            samples.append(dict(sample_id=key, gloss=row['Gloss'], class_id=classes[row['Gloss']], split='test',
                participant_id=row['Participant ID'], input_path=str(path), source_bytes=path.stat().st_size,
                source_sha256=file_sha(path)))
        test = dict(classes=cohort['classes'], samples=samples, predicate='every official test row with exact Gloss in COHORT002 classes',
            run_id=args.run_id, freeze_sha256=file_sha(args.freeze), frozen_utc=freeze['frozen_utc'],
            selected_metadata_access_utc=run.status['start_utc'], split_hashes=config['split_hashes'],
            class_map_origin='COHORT002 train only; no test resampling', automatic_exclusions=0,
            participant_counts=dict(Counter(s['participant_id'] for s in samples)),
            class_counts=dict(Counter(s['gloss'] for s in samples)))
        query_map(cohort, test)
        run.save('test_manifest.json', test)
        audits = []
        for number, source in enumerate(samples, 1):
            cap = None
            try:
                response = subprocess.run(['ffprobe', '-v', 'error', '-threads', '1', '-select_streams', 'v:0',
                    '-show_entries', 'stream=width,height,avg_frame_rate,r_frame_rate,duration:frame=best_effort_timestamp_time',
                    '-of', 'json', source['input_path']], capture_output=True, text=True, check=True, timeout=30)
                if response.stderr.strip():
                    raise ValueError('ffprobe reported a decode error')
                native = json.loads(response.stdout)
                pts = [float(f['best_effort_timestamp_time']) for f in native['frames']]
                if not 16 <= len(pts) <= config['max_frames_per_video']:
                    raise ValueError('Native length outside frozen contract; no automatic exclusion')
                stream = native['streams'][0]
                indices = fingerprint_indices(len(pts), [0, .25, .5, .75, 1])
                fingerprints = {}
                cap = cv2.VideoCapture(source['input_path'])
                if not cap.isOpened():
                    raise ValueError('OpenCV cannot open held-out source')
                count = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame.shape != (stream['height'], stream['width'], 3):
                        raise ValueError('Frame dimensions changed')
                    if count in indices:
                        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
                        fingerprints[count] = dict(dhash64=dhash64(gray), gray_std=float(gray.std()),
                                                   gray_range=int(gray.max())-int(gray.min()))
                    count += 1
                timeline = summarize_timeline(pts, count, 16, 2)
                audits.append({k: source[k] for k in ('sample_id', 'split', 'class_id', 'participant_id', 'source_sha256')} |
                    timeline | dict(pts_s=pts, stream=stream, fingerprint_frame_indices=indices,
                                    fingerprints=[fingerprints[i] for i in indices], passed=True))
            except BaseException:
                run.save('audit_progress.json', dict(samples=audits, failed_source=source['sample_id'], passed=False))
                raise
            finally:
                if cap is not None:
                    cap.release()
            if number % 50 == 0 or number == len(samples):
                run.save('audit_progress.json', dict(samples=audits, expected=len(samples), passed=False))
                print(json.dumps(dict(audited=number, expected=len(samples))), flush=True)
        candidates, tested, eligible = [], 0, 0
        for source in audits:
            for known in old_audit['samples']:
                tested += 1
                pair = compare_pair(source, known)
                if pair is None:
                    continue
                eligible += 1
                if pair['candidate']:
                    candidates.append(dict(test_id=source['sample_id'], known_id=known['sample_id'], known_split=known['split'], **pair))
        run.save('duplicate_screen.json', dict(cross_split_pairs_tested=tested, eligible_pairs=eligible,
            candidates=candidates, automatic_exclusions=0,
            scope='same five-thumbnail/length heuristic as NEAR001; only selected train/dev versus selected test; not exhaustive uniqueness'))
        audit = dict(samples=audits, passed=True, automatic_exclusions=0, test_manifest_sha256=file_sha(run.output / 'test_manifest.json'),
                     frames=sum(s['frames'] for s in audits), dense_windows=sum(s['windows'] for s in audits),
                     sparse_windows=sum((s['windows'] + 3) // 4 for s in audits), candidate_count=len(candidates))
        run.save('video_audit.json', audit)
        # Representative dev timing rates frozen before test. Only native counts used here.
        rates = config['representative_rates']
        estimate = rates['setup_seconds'] + rates['safety_factor'] * (
            rates['encoder_seconds_per_window'] * (audit['dense_windows'] + 3 * audit['sparse_windows']) +
            rates['decode_preprocess_seconds_per_frame'] * audit['frames'] * 4 +
            rates['pool_search_seconds_per_query_recipe'] * len(samples) * 4)
        run.save('preflight_estimate.json', dict(actual_query_count=len(samples), actual_participant_count=len(test['participant_counts']),
            estimated_gpu_allocation_seconds=estimate, actual_encoder_windows_planned=audit['dense_windows'] + 3 * audit['sparse_windows'],
            rates=rates, gallery_source='reuse verified immutable train-only CACHE006; no test fit',
            planned_pooled_payload_bytes=len(samples) * 5 * 768 * 8, source_read_bytes_audit=planned,
            launch_eligible=len(candidates) == 0 and estimate <= config['prediction_allowance_seconds']))
        run.status['output_hashes'] = {name: file_sha(run.output / name) for name in
            ('test_manifest.json', 'video_audit.json', 'duplicate_screen.json', 'preflight_estimate.json')}


if __name__ == '__main__':
    main()
