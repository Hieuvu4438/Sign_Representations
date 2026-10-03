"""Resolve a bounded ASL-MTP metadata subset from public pages; no media downloads."""
import argparse
import csv
import hashlib
import json
import urllib.request
from pathlib import Path

from _common import ROOT
from signrepr.dai import candidates_from_html, resolve_row
from signrepr.io import sha256, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT / 'third_party/SL-Models-Analysis/data/asl-mtp.csv')
    parser.add_argument('--limit-pages', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.limit_pages <= 20:
        raise ValueError('Public-page pilot must be bounded to 1..20 pages')
    with args.csv.open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    urls = sorted({row['file'] for row in rows if '/utterancesearchresult' in row['file']},
                  key=lambda url: hashlib.sha256((str(args.seed) + url).encode()).hexdigest())[:args.limit_pages]
    selected = [(index, row) for index, row in enumerate(rows) if row['file'] in urls]
    if args.dry_run:
        print(json.dumps({'selected_pages': len(urls), 'selected_pairs': len(selected), 'media_downloads': False, 'role': 'external_mapping_only'}))
        return
    cache = ROOT / 'evidence/dai_pages'
    cache.mkdir(parents=True, exist_ok=True)
    parsed, errors = {}, []
    for url in urls:
        key = hashlib.sha256(url.encode()).hexdigest()
        path = cache / (key + '.html')
        try:
            if not path.exists():
                with urllib.request.urlopen(url, timeout=20) as response:
                    if 'text/html' not in response.headers.get('Content-Type', ''):
                        raise ValueError('DAI response is not HTML')
                    content = response.read(2000001)
                    if len(content) > 2000000:
                        raise ValueError('Page exceeds pilot cap')
                path.write_bytes(content)
            candidates, login_required = candidates_from_html(path.read_text(), url)
            parsed[url] = (candidates, path, login_required)
        except Exception as error:
            errors.append({'url': url, 'error': str(error)})
    outputs = []
    for index, row in selected:
        base = {'pair_row_index': index, 'source_csv_sha256': sha256(args.csv), 'phenomenon': row['phenomenon'],
                'raw_video_id': row['video_id'], 'page_url': row['file'], 'role': 'external_test_only'}
        if row['file'] not in parsed:
            result = {'mapping_status': 'UNRESOLVED', 'reason': 'PAGE_ACCESS_FAILED'}
        else:
            candidates, path, login_required = parsed[row['file']]
            result = {**resolve_row(row, candidates), 'page_sha256': sha256(path), 'page_path': str(path.relative_to(ROOT)),
                      'official_download_login_required': login_required}
        outputs.append({**base, **result})
        print(json.dumps({'row': index, 'mapping_status': result['mapping_status'], 'reason': result.get('reason')}), flush=True)
    write_jsonl(ROOT / 'data/external/asl_mtp_resolution_pilot.jsonl', outputs)
    write_json(ROOT / 'reports/asl_mtp_resolution_pilot.json', {'pages_attempted': len(urls), 'pairs': len(outputs),
               'metadata_resolved_review_pending': sum(row['mapping_status'] == 'RESOLVED_METADATA_REVIEW_PENDING' for row in outputs),
               'errors': errors, 'scope': 'Public-page mapping only; no media or gated annotation download; not linguistic human validation.'})


if __name__ == '__main__':
    main()
