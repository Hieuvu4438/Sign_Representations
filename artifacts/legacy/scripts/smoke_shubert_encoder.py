"""Strict load and deterministic interface smoke on CPU; no raw-video benchmark claim."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from _common import ROOT
from signrepr.io import sha256, write_json
from signrepr.shubert import NativeSHuBERT, load_native_dino


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/shubert.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dino', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior smoke attempts; select a fresh output')
    args.output.mkdir(parents=True)
    begun = time.perf_counter()
    config = yaml.safe_load(args.config.read_text())
    try:
        model = NativeSHuBERT(ROOT, config, device='cpu')
        write_json(args.output / 'load_audit.json', model.audit)
        rng = np.random.default_rng(42)
        maximum_difference = 0.
        shapes = {}
        for length in [15, 16]:
            streams = {key: rng.standard_normal((length, dimension)).astype(np.float32)
                       for key, dimension in config['stream_dimensions'].items()}
            a, b = model.forward(streams), model.forward(streams)
            difference = float(np.abs(a - b).max())
            maximum_difference = max(maximum_difference, difference)
            shapes[str(length)] = list(a.shape)
            if difference > config['deterministic_tolerance']:
                raise ValueError('Nondeterministic eval forward')
        # Reject temporal misalignment instead of silently cropping all streams.
        streams['face'] = streams['face'][:-1]
        try:
            model.forward(streams)
        except ValueError:
            alignment_rejected = True
        else:
            raise ValueError('Misaligned native streams were accepted')
        dino_audits = []
        if args.dino:
            import torch
            for asset_id in ['W03', 'W04']:
                dino, audit = load_native_dino(ROOT, config, asset_id)
                with torch.inference_mode():
                    output = dino(torch.zeros((1, 3, 224, 224)))
                if output.shape != (1, 384) or not torch.isfinite(output).all():
                    raise ValueError('DINO interface smoke failed')
                audit['fixture_output_shape'] = list(output.shape)
                dino_audits.append(audit)
                del dino
        report = {'status': 'PASS_ENCODER_INTERFACE_ONLY', 'device': 'cpu', 'shapes': shapes,
                  'determinism_max_abs_difference': maximum_difference, 'misaligned_streams_rejected': alignment_rejected,
                  'raw_video_preprocessing_smoke': 'PENDING', 'dino_audits': dino_audits,
                  'fixture_note': 'Deterministic synthetic stream inputs test the real pretrained checkpoint interface; these are not dataset features or performance measurements.',
                  'elapsed_seconds': time.perf_counter() - begun, 'config_sha256': sha256(args.config),
                  'implementation_sha256': sha256(ROOT / 'src/signrepr/shubert.py')}
    except Exception as error:
        report = {'status': 'FAIL', 'failure_reason': f'{type(error).__name__}: {error}', 'elapsed_seconds': time.perf_counter() - begun}
        write_json(args.output / 'report.json', report)
        raise
    write_json(args.output / 'report.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
