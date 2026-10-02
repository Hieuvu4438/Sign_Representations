"""Bounded NCSLGR readiness download: only URLs in the official public index.

No inference/training. HTTP transport is upgraded to HTTPS without changing
the published path. Individual and total byte limits precede any download.
"""
import argparse
import concurrent.futures
import csv
import io
import json
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone

from _common import ROOT
from signrepr.io import sha256, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--reuse-heads', action='store_true')
    args = parser.parse_args()
    cohort = ROOT / 'data/external/ncslgr/readiness_cohort.jsonl'
    rows = [r for _, r in read_jsonl(cohort)]
    if len(rows) > 300:
        raise ValueError('Readiness collection exceeds 300-utterance bound')
    index = ROOT / 'data/external/ncslgr_video_index.zip'
    with zipfile.ZipFile(index) as z:
        original = list(csv.DictReader(io.StringIO(z.read(
            'video_index-20120129/files_by_xml_file.csv').decode('utf-8-sig'))))
    published = {r['Compressed MOV file'] for r in original}
    selected = {}
    for row in rows:
        for view in row['views']:
            u = view['Compressed MOV file']
            if u not in published or ';' in u:
                raise ValueError('URL not an individual published index entry')
            key = __import__('hashlib').sha256(u.encode()).hexdigest()
            selected[key] = u
    heads_path = ROOT / 'evidence/ncslgr/cohort_heads.json'
    if args.reuse_heads:
        heads = json.loads(heads_path.read_text())
        if heads['cohort_sha256'] != sha256(cohort) or set(heads['results']) != set(selected):
            raise ValueError('Stored HEAD cohort does not match')
    else:
        def head(item):
            key, u = item
            result = {'asset_id': 'NCSLGR_COHORT_' + key[:16],
                      'published_url': u, 'request_url': u.replace('http:', 'https:', 1)}
            try:
                request = urllib.request.Request(result['request_url'], method='HEAD',
                    headers={'User-Agent': 'SignRepresentationResearch/1.0'})
                with urllib.request.urlopen(request, timeout=30) as response:
                    result.update(http_status=response.status, final_url=response.url,
                                  expected_bytes=int(response.headers.get('Content-Length', 0)),
                                  content_type=response.headers.get('Content-Type'))
                result['status'] = 'READY' if (
                    result['http_status'] == 200 and result['content_type'] == 'video/quicktime'
                    and 0 < result['expected_bytes'] <= 10_000_000) else 'NOT_READY'
            except Exception as error:
                result.update(status='FAIL', error=type(error).__name__ + ': ' + str(error))
            print(json.dumps({'head': key, 'status': result['status']}), flush=True)
            return key, result
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = dict(pool.map(head, selected.items()))
        heads = {'utc': datetime.now(timezone.utc).isoformat(), 'cohort_sha256': sha256(cohort),
                 'official_index_sha256': sha256(index), 'results': results}
        write_json(heads_path, heads)
    ready = [r for r in heads['results'].values() if r['status'] == 'READY']
    total = sum(r['expected_bytes'] for r in ready)
    if total > 1024 ** 3:
        raise ValueError('Targeted metadata-selected collection exceeds 1 GiB bound')
    print(json.dumps({'ready_urls': len(ready), 'selected_urls': len(selected),
                      'registered_bytes': total, 'download': args.download}), flush=True)
    if not args.download:
        return
    registry_path = ROOT / 'provenance/assets.json'
    registry = json.loads(registry_path.read_text())
    for r in ready:
        registry['assets'][r['asset_id']] = {
            'kind': 'video', 'url': r['request_url'], 'expected_bytes': r['expected_bytes'],
            'published_url': r['published_url'], 'url_source': 'D02_INDEX',
            'revision': '20120129 official index; historical annotation readiness cohort',
            'local_path': 'data/external/ncslgr/cohort_videos/' + r['asset_id'].lower() + '.mov',
            'license_access_note': 'Targeted research use of explicitly public-indexed URL; no DAI login/authentication bypass; no redistribution.'}
    write_json(registry_path, registry)
    def fetch(r):
        key = r['asset_id']
        process = subprocess.run([sys.executable, str(ROOT / 'scripts/fetch_asset.py'),
                                  '--asset', key], capture_output=True, text=True)
        (ROOT / 'logs' / ('fetch_' + key.lower() + '.log')).write_text(
            process.stdout + process.stderr)
        result = {'asset_id': key, 'status': 'PASS' if process.returncode == 0 else 'FAIL',
                  'exit_code': process.returncode}
        print(json.dumps(result), flush=True)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, ready))
    failures = [r for r in results if r['status'] != 'PASS']
    write_json(ROOT / 'reports/ncslgr_cohort_download.json', {
        'status': 'PASS' if not failures and len(ready) == len(selected) else 'PARTIAL',
        'cohort_sha256': sha256(cohort), 'selected_urls': len(selected),
        'ready_urls': len(ready), 'registered_bytes': total,
        'downloaded_urls': len(results) - len(failures), 'failures': failures,
        'head_failures': [r for r in heads['results'].values() if r['status'] != 'READY'],
        'inference_or_training_performed': False})


if __name__ == '__main__':
    main()
