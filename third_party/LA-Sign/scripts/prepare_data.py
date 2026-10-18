"""Fetch public inputs and prepare LA-Sign label pickles (Python 3.10+)."""
import argparse
import gzip
import json
import pickle
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
UNI_SIGN = 'https://raw.githubusercontent.com/ZechengLi19/Uni-Sign/eed438bcb49e30405cd6ccdfcccca330c134e830'


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    partial = path.with_suffix(path.suffix + '.part')
    print(f'Downloading {url} -> {path}', flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    partial.replace(path)


def dump_labels(path, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wb') as f:
        pickle.dump(labels, f)
    print(f'{path.relative_to(ROOT)}: {len(labels)} samples', flush=True)


def extract(zip_path, target):
    with zipfile.ZipFile(zip_path) as z:
        target.mkdir(parents=True, exist_ok=True)
        for entry in z.infolist():
            path = (target / entry.filename).resolve()
            if not path.is_relative_to(target.resolve()):
                raise ValueError(f'Unexpected archive path: {entry.filename}')
        z.extractall(target)


def wlasl():
    for phase in ['train', 'dev', 'test']:
        download(f'{UNI_SIGN}/data/WLASL/labels-2000.{phase}', ROOT / f'data/WLASL/labels-2000.{phase}')
    archive = ROOT / 'assets/wlasl_pose_format.zip'
    download('https://huggingface.co/ZechengLi19/Uni-Sign/resolve/main/wlasl_pose_format.zip?download=true', archive)
    pose_root = ROOT / 'dataset/WLASL/pose_format'
    with zipfile.ZipFile(archive) as z:
        complete = all((pose_root / i.filename).is_file() and (pose_root / i.filename).stat().st_size == i.file_size
                       for i in z.infolist() if not i.is_dir())
    if not complete:
        extract(archive, pose_root)
    metadata = ROOT / 'assets/WLASL_v0.3.json'
    download('https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json', metadata)
    classes = {row['gloss'] for row in json.loads(metadata.read_text())[:300]}
    for phase in ['train', 'dev', 'test']:
        with gzip.open(ROOT / f'data/WLASL/labels-2000.{phase}', 'rb') as f:
            labels = pickle.load(f)
        output_phase = 'val' if phase == 'dev' else phase
        dump_labels(ROOT / f'data/WLASL/labels-300.{output_phase}',
                    {key: sample for key, sample in labels.items() if sample['text'] in classes})


def msasl():
    archive = ROOT / 'assets/MS-ASL.zip'
    download('https://download.microsoft.com/download/3/c/a/3ca92c78-1c4a-4a91-a7ee-6980c1d242ec/MS-ASL.zip', archive)
    extract(archive, ROOT / 'assets/msasl_metadata')
    base = ROOT / 'assets/msasl_metadata/MS-ASL'
    for phase in ['train', 'val', 'test']:
        rows = json.loads((base / f'MSASL_{phase}.json').read_text())
        labels = {}
        manifest = []
        for index, row in enumerate(rows):
            # A clip has its own pose file, already cropped to start_time/end_time.
            # IDs are ours, NOT an assertion about unpublished LA-Sign pose names.
            key = f'{phase}_{index:05d}'
            labels[key] = {'name': key, 'text': row['text'], 'gloss': '',
                           'video_path': f'{key}.mp4', 'label': row['label']}
            manifest.append({'name': key, 'url': row['url'], 'start_time': row['start_time'],
                             'end_time': row['end_time'], 'text': row['text'],
                             'video_path': f'dataset/MSASL/rgb_format/{phase}/{key}.mp4',
                             'pose_path': f'dataset/MSASL/pose_format/{phase}/{key}.pkl'})
        for size in [100, 200, 1000]:
            dump_labels(ROOT / f'data/MSASL/labels-{size}.{phase}',
                        {key: sample for key, sample in labels.items() if sample['label'] < size})
        (ROOT / f'data/MSASL/{phase}_manifest.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in manifest))
    print('MSASL metadata is ready. The official archive contains NO videos or poses.')


def mt5():
    target = ROOT / 'pretrained_weight/mt5-base'
    if (target / 'pytorch_model.bin').exists():
        return
    sibling = ROOT.parent / 'Uni-Sign/pretrained_weight/mt5-base'
    if (sibling / 'pytorch_model.bin').exists() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(sibling, target_is_directory=True)
        return
    from huggingface_hub import snapshot_download
    snapshot_download('google/mt5-base', local_dir=str(target), allow_patterns=[
        'config.json', 'generation_config.json', 'pytorch_model.bin', 'spiece.model',
        'tokenizer_config.json', 'special_tokens_map.json'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['WLASL', 'MSASL', 'all'], default='all')
    parser.add_argument('--skip-mt5', action='store_true')
    args = parser.parse_args()
    if args.dataset in ['WLASL', 'all']:
        wlasl()
    if args.dataset in ['MSASL', 'all']:
        msasl()
    if not args.skip_mt5:
        mt5()
