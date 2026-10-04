"""CPU strict coverage of published face/hand weights against pinned official DINO."""
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'method/frozen_probe'))
sys.path.insert(0, str(ROOT))
from cache import file_sha
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    for field in ('config', 'output-dir'):
        parser.add_argument('--' + field, required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        for producer_name, filename, digest in cfg['dependencies']:
            producer = REPO / producer_name
            status = json.loads((producer / 'status.json').read_text())
            if status['status'] != 'completed' or file_sha(producer / filename) != digest:
                raise ValueError('Completed, immutable dependency required')
            if status['output_hashes'][filename] != digest:
                raise ValueError('Dependency output ledger differs')
        source_run = REPO / cfg['source_run']
        manifest = json.loads((source_run / 'source_manifest.json').read_text())
        for row in manifest['files']:
            if file_sha(source_run / 'official_source' / row['path']) != row['sha256']:
                raise ValueError('Pinned official DINO source changed')
        author_source = REPO / cfg['author_source']
        if file_sha(author_source) != cfg['author_source_sha256']:
            raise ValueError('Pinned author preprocessing changed')
        sys.path.insert(0, str(source_run / 'official_source'))
        import numpy as np
        import torch
        from PIL import Image
        from torchvision import transforms
        from dinov2.hub.backbones import dinov2_vits14_reg
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        if torch.cuda.is_initialized():
            raise ValueError('CPU-only audit unexpectedly initialized CUDA')
        spec = importlib.util.spec_from_file_location('author_dino_reference', author_source)
        author = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(author)  # main guarded; no loader/hub/video function executed
        pixels = np.random.default_rng(args.seed).integers(0, 256, (80, 112, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
        own_input = transform(image).unsqueeze(0)
        reference = author.preprocess_image(image)
        preprocessing_equal = torch.equal(own_input, reference)
        if not preprocessing_equal:
            raise ValueError('Author PIL helper preprocessing parity failed')
        equality = json.loads((REPO / cfg['asset_run'] / 'asset_equality.json').read_text())
        if not equality['all_equal']:
            raise ValueError('Published author bytes required')
        report = dict(run_id=args.run_id, GPU_hours=0, dino_commit=manifest['commit'],
                      author_historical_dino_commit='unknown',
                      preprocessing_equal=preprocessing_equal, input_shape=list(reference.shape),
                      input_semantics='synthetic RGB/PIL Resize helper; not decord encoded-crop video path',
                      state_recipe='drop dino_head; strip leading backbone.; replace pos_embed with (1,257,384)',
                      scope='CPU software/load/preprocessing helper only; no real crops or benchmark', models=[])
        run.save('dino_load_audit.json', report)
        arrays = {'synthetic_rgb': pixels, 'normalized_input': own_input.numpy()}
        for asset in equality['assets'][1:]:
            path = REPO / asset['local_path']
            if file_sha(path) != asset['local_sha256']:
                raise ValueError('Published author DINO bytes changed')
            model = dinov2_vits14_reg(pretrained=False)
            model.pos_embed = torch.nn.Parameter(torch.zeros(1, 257, 384))
            container = torch.load(path, map_location='cpu', weights_only=False)
            teacher = container['teacher']
            heads = sorted(k for k in teacher if 'dino_head' in k)
            state = {}
            for name, value in teacher.items():
                if 'dino_head' in name:
                    continue
                if not name.startswith('backbone.'):
                    raise ValueError('Unexpected non-backbone/non-head teacher key')
                state[name.removeprefix('backbone.')] = value
            expected = model.state_dict()
            missing = sorted(set(expected) - set(state))
            unexpected = sorted(set(state) - set(expected))
            mismatches = {k: dict(expected=list(expected[k].shape), actual=list(state[k].shape))
                          for k in set(expected) & set(state) if expected[k].shape != state[k].shape}
            row = dict(asset=asset['local_path'], sha256=asset['local_sha256'],
                       teacher_keys=len(teacher), omitted_teacher_head_keys=heads,
                       expected_keys=len(expected), loaded_keys=len(state), missing_keys=missing,
                       unexpected_keys=unexpected, shape_mismatches=mismatches)
            report['models'].append(row)
            run.save('dino_load_audit.json', report)
            if missing or unexpected or mismatches:
                raise ValueError('Pinned official DINO architecture differs; no partial load')
            model.load_state_dict(state, strict=True)
            model.cpu().eval()
            with torch.no_grad():
                output = model(reference)
            if output.shape != (1, 384) or not torch.isfinite(output).all():
                raise ValueError('Synthetic DINO output contract failed')
            row.update(strict_load_completed=True, eval_mode=not model.training,
                       output_shape=list(output.shape), finite=True)
            arrays[path.stem] = output.numpy()
            del model, container, teacher, state, expected
        np.savez(run.output / 'synthetic_dino_reference.npz', **arrays)
        report.update(peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      CUDA_initialized=torch.cuda.is_initialized(), all_strict_pass=True)
        if report['CUDA_initialized']:
            raise ValueError('CPU-only audit initialized CUDA')
        run.save('dino_load_audit.json', report)
        run.status['output_hashes'] = {name: file_sha(run.output / name) for name in
                                      ('dino_load_audit.json', 'synthetic_dino_reference.npz')}


if __name__ == '__main__':
    main()
