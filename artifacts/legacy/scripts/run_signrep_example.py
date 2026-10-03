"""Run the pinned upstream example with local paths and a selectable device."""
import argparse
import ast
import json
import os
import sys
from pathlib import Path

from _common import ROOT


class ExampleSettings(ast.NodeTransformer):
    def __init__(self, checkpoint, video, device):
        self.values = {'ckpt_dir': str(checkpoint), 'video_dir': str(video)}
        self.device = device

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in self.values:
                node.value = ast.Constant(self.values[name])
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda' and not node.args and not node.keywords:
            node.func.attr = 'to'
            node.args = [ast.Constant(self.device)]
        return node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda:0'])
    args = parser.parse_args()
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    import numpy as np
    import torch
    import yaml
    from signrepr.io import sha256
    from signrepr.shubert import verify_source

    config = yaml.safe_load((ROOT / 'configs/signrep.yaml').read_text())
    source = verify_source(ROOT, config['upstream_path'], config['upstream_commit'])
    checkpoint = ROOT / config['checkpoint_path']
    if sha256(checkpoint) != config['checkpoint_sha256']:
        raise ValueError('SignRep checkpoint checksum mismatch')
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    sys.path.insert(0, str(source))
    example = source / 'example_usage.py'
    tree = ast.parse(example.read_text(), filename=str(example))
    tree = ast.fix_missing_locations(ExampleSettings(checkpoint, args.video.resolve(), args.device).visit(tree))
    namespace = {'__name__': '__main__', '__file__': str(example)}
    exec(compile(tree, str(example), 'exec'), namespace)
    features, latents = namespace['list_of_features'], namespace['list_of_latents']
    if not features:
        raise ValueError('Upstream example requires at least 16 decoded video frames')
    features, latents = np.stack(features), np.stack(latents)
    if not np.isfinite(features).all() or not np.isfinite(latents).all():
        raise ValueError('Non-finite upstream output')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        np.savez_compressed(handle, features=features, latent=latents)
    print(json.dumps({'status': 'PASS_UPSTREAM_SIGNREP_EXAMPLE', 'device': args.device,
                      'features_shape': list(features.shape), 'latent_shape': list(latents.shape),
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
