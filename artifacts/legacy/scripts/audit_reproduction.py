"""Audit pinned source/weight availability without loading models or downloading assets."""
import argparse
import json
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from _common import ROOT
from signrepr.io import sha256, write_json


def source_inventory():
    registered = json.loads((ROOT / 'provenance/source_commits.json').read_text())
    observed = {}
    for name in ('SignRep', 'SHuBERT', 'dinov2'):
        path = ROOT / 'third_party' / name
        try:
            head = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.PIPE).strip()
            dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
            observed[name] = {'expected_commit': registered[name]['commit'], 'observed_commit': head,
                              'tracked_changes': dirty, 'status': 'PASS' if head == registered[name]['commit'] and not dirty else 'FAIL'}
        except subprocess.CalledProcessError:
            observed[name] = {'status': 'MISSING_OR_INVALID_CHECKOUT', 'expected_commit': registered[name]['commit']}
    return observed


def remote_inventory():
    urls = {
        'signrep_releases': 'https://api.github.com/repos/ryanwongsa/SignRep/releases',
        'signrep_head': 'https://api.github.com/repos/ryanwongsa/SignRep/commits/main',
        'shubert_head': 'https://api.github.com/repos/ShesterG/SHuBERT/commits/main',
        'shubert_models': 'https://huggingface.co/api/models/ShesterG/SHuBERT/tree/main/models?recursive=true&limit=1000',
    }
    results = {}
    for name, url in urls.items():
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'SignRepresentations-audit'})
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.load(response)
                if response.headers.get('Link') and 'rel="next"' in response.headers['Link']:
                    raise ValueError('Paginated listing requires further inspection')
            if name.endswith('_head'):
                data = {'sha': data['sha']}
            elif name.endswith('_releases'):
                data = [{'tag': r['tag_name'], 'assets': [{'name': a['name'], 'size': a['size'], 'url': a['browser_download_url']} for a in r['assets']]} for r in data]
            else:
                data = [{'path': f['path'], 'bytes': f.get('size'),
                         'local_exists': (ROOT / 'checkpoints/shubert_author_hf' / f['path']).is_file()}
                        for f in data if f['type'] == 'file']
            results[name] = {'url': url, 'status': 'PASS_METADATA_RETRIEVAL', 'data': data}
        except (OSError, ValueError) as error:
            results[name] = {'url': url, 'status': 'UNVERIFIED_REMOTE', 'error': str(error)}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-upstream', action='store_true', help='Read current GitHub/HF metadata; never download weights.')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a fresh audit output; preserve previous observations')
    registry = json.loads((ROOT / 'provenance/assets.json').read_text())['assets']
    weights = {}
    for name, asset in registry.items():
        if not name.startswith(('W', 'C04_')):
            continue
        path = ROOT / asset['local_path']
        digest = sha256(path) if path.is_file() else None
        size = path.stat().st_size if path.is_file() else None
        valid = digest is not None and digest == asset.get('expected_sha256')
        if asset.get('expected_bytes') is not None:
            valid = valid and size == asset['expected_bytes']
        weights[name] = {'path': asset['local_path'], 'bytes': size, 'sha256': digest,
                         'expected_sha256': asset.get('expected_sha256'), 'status': 'PASS' if valid else 'MISSING_OR_MISMATCH'}
    sources = source_inventory()
    integrity = all(x['status'] == 'PASS' for x in [*weights.values(), *sources.values()])
    report = {'utc': datetime.now(timezone.utc).isoformat(),
              'local_registered_assets_and_sources': 'PASS' if integrity else 'FAIL',
              'full_paper_reproduction_ready': False,
              'scope': 'Byte/source integrity only; no new benchmark, video extraction, training or inference.',
              'weights_and_demo_sources': weights, 'upstream_sources': sources,
              'full_reproduction_blockers': [
                  'SignRep public checkout supplies feature inference, not a complete pretraining/downstream benchmark runner and protocol bundle.',
                  'SHuBERT upstream downstream task preparation and fine-tuning remain TODO; task-specific paper checkpoints are not mapped to the local weights.',
                  'SHuBERT pretraining config requires four per-sample k-means label files and the original training manifest/features; these are not registered here.',
                  'W08 is an author-demo translator; equivalence to paper benchmark models is unverified.',
                  'Training resume files for demo checkpoint-11625 and standalone initial ByT5 weights are not in the local inventory; they are not needed for strict-loaded W08 inference.',
                  'Existing lexical common cohort is a subset; it is not the full original dataset benchmark.',
                  'Project dependency environments and dataset paths require setup on a fresh machine; Git excludes weights, media and feature caches.',
              ]}
    if args.check_upstream:
        report['remote_metadata'] = remote_inventory()
    write_json(args.output, report)
    print(json.dumps({'output': str(args.output), 'local_registered_assets_and_sources': report['local_registered_assets_and_sources'],
                      'full_paper_reproduction_ready': False}))
    if not integrity or any(x['status'] != 'PASS_METADATA_RETRIEVAL' for x in report.get('remote_metadata', {}).values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
