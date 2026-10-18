"""Fetch only pinned reference files and record their actual checksums/status."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

SOURCES = {
    'Uni-Sign': ('ZechengLi19/Uni-Sign', 'eed438bcb49e30405cd6ccdfcccca330c134e830', ['datasets.py', 'models.py', 'demo/pose_extraction.py']),
    'SignRep': ('ryanwongsa/SignRep', '06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2', ['example_usage.py', 'README.md']),
    'MASA': ('sakura2233565548/MASA', '53bbb3b77bab7e98d33d7fcbd230e3e98c832ed1', ['moco/GCN_Transformer_mask.py', 'moco/st_gcn_encoder/st_gcn_single_frame.py']),
    'MAMP': ('maoyunyao/MAMP', '09e6252a5c7d061c226395d25e6aa47a69e7e58d', ['model_mamp/transformer.py', 'engine_pretrain.py']),
    'V-JEPA': ('facebookresearch/jepa', '51c59d518fc63c08464af6de585f78ac0c7ed4d5', ['app/vjepa/train.py', 'src/models/predictor.py']),
    'V-JEPA-2.1': ('facebookresearch/vjepa2', '204698b45b3712590f06245fbfba32d3be539812', ['app/vjepa_2_1/train.py', 'app/vjepa_2_1/models/predictor.py']),
}


def fetch(item):
    name, repository, revision, filename = item
    url = f'https://raw.githubusercontent.com/{repository}/{revision}/{filename}'
    result = {'source': name, 'repository': f'https://github.com/{repository}', 'revision': revision, 'file': filename, 'url': url}
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()
        result.update(status='downloaded_reference_only', bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    except Exception as exc:
        result.update(status='not_downloaded', error=str(exc))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(Path(__file__).parents[1]/'docs/sources.lock.json'))
    args = parser.parse_args()
    items = [(name, repo, rev, file) for name, (repo, rev, files) in SOURCES.items() for file in files]
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, items))
    report = {'method': 'pose_jepa_hybrid_v1', 'reference_files': records,
              'S-JEPA': {'source': 'https://sjepa.github.io/', 'paper': 'https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf',
                         'implementation': 'independent implementation from proposal14; no official source/checkpoint loaded'},
              'SignSeek': {'source': 'https://arxiv.org/abs/2609.03695', 'status': 'guide_reference_only; browser retrieval failed; no code/weights loaded'},
              'external_weights': [
                  {'source': 'ZechengLi19/Uni-Sign', 'revision': 'eab251b7fe7e8521afc0e67be98add670ea40a0d', 'status': 'not_loaded_into_pose_jepa'},
                  {'source': 'SignRep v0.0.1 ckpt.pt', 'status': 'not_loaded_into_pose_jepa; RGB baseline is a separate existing package'},
                  {'source': 'MAMP/MASA author share links', 'status': 'linked_by_author_not_loaded'},
                  {'source': 'google/mt5-base', 'revision': '2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f',
                   'status': 'weights_not_downloaded_for_pose_jepa'}],
              'copied_implementation_code': False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve actual model-config inspection records when repeating the code audit.
    if output.exists():
        previous = json.loads(output.read_text())
        old = {r['source']: r for r in previous.get('external_weights', [])}
        for i, entry in enumerate(report['external_weights']):
            if entry['source'] in old and old[entry['source']].get('config_sha256'):
                report['external_weights'][i] = old[entry['source']]
    output.write_text(json.dumps(report, indent=2))
    package_lock = Path(__file__).parents[1]/'src/signrepr/sources.lock.json'
    package_lock.write_bytes(output.read_bytes())
    print(json.dumps({'downloaded': sum(r['status'] == 'downloaded_reference_only' for r in records), 'total': len(records)}))


if __name__ == '__main__':
    main()
