"""Prepare a stratified, official-split real-data sanity corpus from existing caches.

Inputs are read-only. Variable-rate timing is read from source frame PTS, not an
assumed shared FPS. This small subset is not the ASL-Citizen full benchmark.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'src'))
from signrepr.data.import_unisign import import_unisign_pose
from signrepr.data.schema import write_manifest, sha256_file
from signrepr.data.splits import validate_manifest


def source_timing(path):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        times = [float(frame.pts*frame.time_base) if frame.pts is not None else np.nan for frame in container.decode(stream)]
        return {'timestamps_sec': times, 'image_width': stream.width, 'image_height': stream.height,
                'fps': float(stream.average_rate) if stream.average_rate else None,
                'timing_source': 'decoded_source_frame_pts'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--classes', type=int, default=4)
    p.add_argument('--train-per-class', type=int, default=4)
    p.add_argument('--val-per-class', type=int, default=2)
    p.add_argument('--test-per-class', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    grouped, counts = {}, {'train': args.train_per_class, 'val': args.val_per_class, 'test': args.test_per_class}
    for split in counts:
        grouped[split] = defaultdict(list)
        with (args.dataset_root/'splits'/f'{split}.csv').open(newline='') as f:
            for row in csv.DictReader(f):
                stem = Path(row['Video file']).stem
                if (args.dataset_root/'keypoints'/f'{stem}.pkl').is_file() and (args.dataset_root/'videos'/row['Video file']).is_file():
                    grouped[split][row['Gloss']].append(row)
    classes = sorted(set.intersection(*(set(g) for g in grouped.values())))
    classes = [label for label in classes if all(len(grouped[s][label]) >= counts[s] for s in counts)]
    rng = random.Random(args.seed); rng.shuffle(classes); classes = sorted(classes[:args.classes])
    if len(classes) != args.classes:
        raise ValueError('Insufficient classes with independent clips in all official splits')
    args.output.mkdir(parents=True, exist_ok=True)
    converted, audit = {}, []
    for split in counts:
        rows = []
        for label in classes:
            candidates = sorted(grouped[split][label], key=lambda row: row['Video file'])
            rng.shuffle(candidates)
            for source in candidates[:counts[split]]:
                stem = Path(source['Video file']).stem
                video = args.dataset_root/'videos'/source['Video file']
                cache = args.dataset_root/'keypoints'/f'{stem}.pkl'
                timing = source_timing(video)
                metadata = {'sample_id': f'asl_citizen/{split}/{stem}', 'corpus': 'asl_citizen', 'language': 'ASL',
                            'split': split, 'source_video_id': stem, 'signer_id': source['Participant ID'],
                            'video_path': str(video.resolve()), 'video_sha256': sha256_file(video),
                            'joint_layout': 'coco_wholebody_133', 'coordinate_system': 'viewport_normalized',
                            'extractor_profile': 'existing_133_joint_cache_extractor_weights_unverified',
                            'score_profile': 'simcc_positive_rational_v1',
                            'extractor_weights_sha256': None, 'gloss_id': label, 'gloss_namespace': 'ASL:asl_citizen',
                            'gloss_sequence': [label], 'translation': None, 'synthetic': False,
                            'official_split_sha256': sha256_file(args.dataset_root/'splits'/f'{split}.csv'), **timing}
                out = args.output/'poses'/split/f'{stem}.npz'
                row, tracking = import_unisign_pose(cache, metadata, out)
                if len(timing['timestamps_sec']) != row['num_frames']:
                    raise ValueError('Source-video/cache frame correspondence mismatch')
                rows.append(row)
                audit.append({'sample_id': row['sample_id'], 'tracking': tracking, 'frame_pts_count': len(timing['timestamps_sec'])})
        path = args.output/f'{split}.jsonl'
        write_manifest(path, rows)
        converted[split] = rows
        (args.output/f'{split}_audit.json').write_text(json.dumps(validate_manifest(path), indent=2))
    all_path = args.output/'all.jsonl'
    write_manifest(all_path, [row for rows in converted.values() for row in rows])
    all_audit = validate_manifest(all_path)
    (args.output/'all_audit.json').write_text(json.dumps(all_audit, indent=2))
    protocol = {'protocol': 'in_domain_ssl', 'model_selection_scope': 'target_validation', 'gallery_manifest': 'train.jsonl', 'query_manifest': 'val.jsonl',
                'classes': classes, 'seed': args.seed, 'counts': {s: len(r) for s, r in converted.items()},
                'scope': 'small stratified official-split sanity subset, not full ASL-Citizen benchmark',
                'timing_policy': 'verified monotonic source PTS with matching native cache frame count',
                'extractor_limitation': 'Existing detector profile/weight hash unavailable; no claim of exact RTMPose-x/RTMW reproduction',
                'score_profile': 'Explicit legacy input adaptation: q=s/(1+s), raw SimCC scores preserved for audit; not calibrated probabilities',
                'tracking_audit': audit}
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(json.dumps({'classes': classes, 'counts': protocol['counts'], 'audit_valid': all_audit['valid']}))


if __name__ == '__main__': main()
