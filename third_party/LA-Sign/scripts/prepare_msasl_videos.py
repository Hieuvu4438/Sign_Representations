"""Download cropped MSASL videos from the official manifest, with resumable output.

This produces videos only. Use RTMPose/RTMW to create corresponding 133-keypoint
pose pickles before launching LA-Sign. Failed samples remain in labels and are
reported; they are never silently excluded from the benchmark.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['train', 'val', 'test'], required=True)
    parser.add_argument('--classes', type=int, choices=[100, 200, 1000], default=1000)
    parser.add_argument('--limit', type=int, default=0, help='0 downloads the full selected split')
    parser.add_argument('--output-root', type=Path, default=ROOT)
    opts = parser.parse_args()
    rows = [json.loads(line) for line in (ROOT / f'data/MSASL/{opts.phase}_manifest.jsonl').read_text().splitlines()]
    official = json.loads((ROOT / f'assets/msasl_metadata/MS-ASL/MSASL_{opts.phase}.json').read_text())
    selected = [row for row, label in zip(rows, official) if label['label'] < opts.classes]
    if opts.limit:
        selected = selected[:opts.limit]
    if not shutil.which('ffmpeg') or not shutil.which('node'):
        parser.error('ffmpeg and Node >=22 are required')
    failures = []
    for row in selected:
        dest = opts.output_root / row['video_path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            print(f'Exists: {dest}', flush=True)
            continue
        url = row['url']
        if not url.startswith('http'):
            url = 'https://' + url
        raw = dest.with_suffix('.source.mp4')
        try:
            subprocess.run([sys.executable, '-m', 'yt_dlp', '--js-runtimes', 'node', '--no-playlist',
                            '--socket-timeout', '20', '--retries', '1', '--extractor-retries', '1',
                            '-f', 'bv[height<=360][ext=mp4]/bv[height<=360]', '-o', str(raw), url], check=True)
            # Local trimming avoids ffmpeg receiving expiring/signed YouTube URLs.
            subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-ss', str(row['start_time']), '-i', str(raw),
                            '-t', str(row['end_time'] - row['start_time']), '-an', '-c:v', 'libx264',
                            '-preset', 'fast', '-crf', '18', str(dest)], check=True)
            subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                            'stream=width,height', str(dest)], check=True)
            raw.unlink()
        except subprocess.CalledProcessError as exc:
            dest.unlink(missing_ok=True)
            failures.append({'name': row['name'], 'url': url, 'error': str(exc)})
            print(f'Failed: {row["name"]}', flush=True)
    report = opts.output_root / f'msasl_{opts.phase}_download_report.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({'selected': len(selected), 'failures': failures}, indent=2), encoding='utf-8')
    sys.exit(1 if failures else 0)
