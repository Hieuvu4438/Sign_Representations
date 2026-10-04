"""Seeded train/dev ffprobe audit. No model, test video, or old project code."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def has_material_jitter(gaps):
    # Exclude floating-point residue at the 1 ms reporting threshold. PTS
    # quantization can still cause jitter: this flag does not establish VFR.
    return bool(gaps) and max(gaps)-min(gaps) > .001 + 1e-9


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--dataset-root', type=Path, default=Path('/home/dongvk/datasets/ASL_Citizen'))
    p.add_argument('--seed', type=int, default=1701)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    status = {'run_id': a.run_id, 'status': 'running', 'pid': os.getpid(), 'host': platform.node(),
              'start_utc': datetime.now(timezone.utc).isoformat(), 'GPU_hours': 0, 'script_sha256': sha(__file__)}
    def save(name, data):
        tmp = a.output_dir/(name+'.tmp')
        tmp.write_text(json.dumps(data, indent=2)+'\n');tmp.replace(a.output_dir/name)
    save('status.json', status)
    (a.output_dir/'audit_video.py.snapshot').write_bytes(Path(__file__).read_bytes())
    try:
        rng = random.Random(a.seed)
        samples = []
        for split, n in [('train', 2), ('val', 3)]:
            csv_path = a.dataset_root/'splits'/f'{split}.csv'
            with csv_path.open(newline='') as f:
                rows = sorted(csv.DictReader(f), key=lambda r: r['Video file'])
            for row in rng.sample(rows, n):
                name = row['Video file']
                if Path(name).name != name:
                    raise ValueError('Invalid video basename')
                path = a.dataset_root/'videos'/name
                argv = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                        'stream=width,height,avg_frame_rate,r_frame_rate,duration:frame=best_effort_timestamp_time',
                        '-of', 'json', str(path)]
                probed = subprocess.run(argv, capture_output=True, text=True, timeout=30)
                if probed.returncode:
                    raise RuntimeError(probed.stderr)
                metadata = json.loads(probed.stdout)
                stream = metadata['streams'][0]
                frames = metadata.get('frames', [])
                pts = [float(f['best_effort_timestamp_time']) for f in frames]
                valid = bool(pts) and stream['width'] > 0 and stream['height'] > 0 and all(math.isfinite(t) for t in pts)
                valid = valid and all(b > t for t, b in zip(pts, pts[1:]))
                if not valid:
                    raise ValueError(f'{name}: invalid dimensions or timeline')
                gaps = [b-t for t,b in zip(pts,pts[1:])]
                windows = list(range(0, max(0, len(pts)-16+1), 2))
                samples.append({'sample_id': name, 'split': split, 'source_sha256': sha(path),
                                'split_csv_sha256': sha(csv_path), 'file_bytes': path.stat().st_size,
                                'stream': stream, 'decoded_frame_metadata_count': len(pts),
                                'pts_seconds': pts, 'min_delta_s': min(gaps) if gaps else None,
                                'max_delta_s': max(gaps) if gaps else None,
                                'pts_jitter_over_1ms': has_material_jitter(gaps),
                                'reference_window16_stride2_count': len(windows),
                                'full_video_covered_by_reference_windows': bool(windows) and windows[-1]+16 == len(pts),
                                'passed': True, 'ffprobe_command': argv})
        report = {'run_id': a.run_id, 'seed': a.seed, 'selection': 'sorted IDs; random.sample; train2,val3',
                  'ffprobe_version': subprocess.run(['ffprobe','-version'],capture_output=True,text=True,check=True).stdout.splitlines()[0],
                  'samples': samples, 'passed': all(x['passed'] for x in samples),
                  'scope': 'five video metadata/frame-timeline checks; not full pixel decode or feature parity',
                  'jitter_caveat': 'PTS quantization may cause jitter; flag alone does not prove variable-frame-rate'}
        save('video_audit.json', report)
        status.update(status='completed',exit_code=0,output_sha256=sha(a.output_dir/'video_audit.json'))
    except Exception as e:
        status.update(status='failed',exit_code=1,error=repr(e))
    status.update(wall_seconds=time.monotonic()-start,end_utc=datetime.now(timezone.utc).isoformat())
    save('status.json',status);print(json.dumps(status))
    if status['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
