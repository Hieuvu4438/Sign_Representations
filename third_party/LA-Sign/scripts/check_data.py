"""Validate every selected label against its actual pose; exit nonzero on gaps."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import utils
from config import train_label_paths, dev_label_paths, test_label_paths
from datasets import S2T_Dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='WLASL2000', choices=list(train_label_paths))
    parser.add_argument('--output', default=None)
    opts = parser.parse_args()
    args = utils.get_args_parser().parse_args(['--dataset', opts.dataset])
    report = {}
    for phase, paths in [('train', train_label_paths), ('dev' if 'WLASL' in opts.dataset else 'val', dev_label_paths), ('test', test_label_paths)]:
        bad = []
        if not Path(paths[opts.dataset]).is_file():
            report[phase] = {'error': f'Missing labels: {paths[opts.dataset]}'}
            continue
        dataset = S2T_Dataset(paths[opts.dataset], args, phase)
        for index, key in enumerate(dataset.list):
            try:
                sample = dataset[index]
                for part, count in [('body', 9), ('left', 21), ('right', 21), ('face_all', 18)]:
                    pose = sample[1][part]
                    if tuple(pose.shape[1:]) != (count, 3) or len(pose) == 0 or not np.isfinite(pose.numpy()).all():
                        raise ValueError(f'Invalid {part} pose: {tuple(pose.shape)}')
            except Exception as e:
                bad.append({'key': key, 'error': str(e)})
        report[phase] = {'samples': len(dataset), 'valid': len(dataset) - len(bad), 'invalid': len(bad), 'examples': bad[:10]}
        print(json.dumps({phase: report[phase]}), flush=True)
    if opts.output:
        path = Path(opts.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
    sys.exit(1 if any(row.get('error') or row.get('invalid', 0) for row in report.values()) else 0)
