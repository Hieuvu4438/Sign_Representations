"""Granted-GPU numerical interface parity, entirely synthetic and predeclared."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'method/frozen_probe'))
sys.path.insert(0, str(ROOT))
from cache import file_sha
from gpu_accounting import inspect_effective_budget
from resources import gpu_preflight
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
        for name, digest in cfg['input_hashes'].items():
            if file_sha(REPO / name) != digest:
                raise ValueError('Frozen software/reference identity changed')
        for producer in cfg['completed_dependencies']:
            if json.loads((REPO / producer / 'status.json').read_text())['status'] != 'completed':
                raise ValueError('Completed strict CPU producer required')
        source_root = REPO / cfg['dino_source_run']
        manifest = json.loads((source_root / 'source_manifest.json').read_text())
        for row in manifest['files']:
            if file_sha(source_root / 'official_source' / row['path']) != row['sha256']:
                raise ValueError('Pinned official architecture changed')
        budget = inspect_effective_budget(cfg)
        physical_rows = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                       check=True, capture_output=True, text=True, timeout=10).stdout.splitlines()
        if physical_rows != [cfg['granted_uuid']]:
            raise ValueError('This native compatibility smoke requires one verified physical GPU')
        observation = gpu_preflight()
        if observation['uuid'] != cfg['granted_uuid']:
            raise ValueError('Granted UUID differs')
        run.start_gpu(budget)
        import numpy as np
        import torch
        if torch.cuda.device_count() != 1:
            raise ValueError('Exactly one scheduler-visible granted device required; visibility preserved')
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(8192 / observation['total_mib'], 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        sys.path.insert(0, str(REPO / 'third_party/SHuBERT/fairseq'))
        sys.path.insert(0, str(source_root / 'official_source'))
        from examples.shubert.models.shubert import SHubertConfig, SHubertModel
        from dinov2.hub.backbones import dinov2_vits14_reg
        report = dict(run_id=args.run_id, hardware=observation, torch_version=torch.__version__,
                      atol=cfg['atol'], rtol=cfg['rtol'], dtype='float32', TF32=False, models=[],
                      scope='Synthetic CPU/GPU numerical interface only; no real video/crop/mask, scientific pilot or benchmark')
        run.save('gpu_parity.json', report)

        def compare(name, tensor, reference):
            array = tensor.detach().cpu().numpy()
            if array.shape != reference.shape or not np.isfinite(array).all():
                raise ValueError('Synthetic output shape/finite contract failed')
            equal = bool(np.allclose(array, reference, atol=cfg['atol'], rtol=cfg['rtol']))
            row = dict(name=name, shape=list(array.shape), finite=True, parity=equal,
                       max_absolute_error=float(np.max(np.abs(array - reference))))
            report['models'].append(row)
            run.save('gpu_parity.json', report)
            if not equal:
                raise ValueError('Registered numerical tolerance failed; no tolerance widening')

        main_ref = np.load(REPO / cfg['main_reference'], allow_pickle=False)
        main = SHubertModel(SHubertConfig())
        checkpoint = torch.load(REPO / cfg['main_checkpoint'], map_location='cpu', weights_only=False)
        main.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        main.cuda().eval()
        source = {name.removeprefix('input_'): torch.from_numpy(main_ref[name]).cuda()
                  for name in main_ref.files if name.startswith('input_')}
        with torch.no_grad():
            output = main.extract_features([source], padding_mask=None, kmeans_labels=None, mask=False)
        compare('SHuBERT_author_layer_last', torch.stack([row[-1].squeeze(1) for row in output['layer_results']]),
                main_ref['layer_last'])
        compare('SHuBERT_final_x_separate', output['x'], main_ref['final_x'])
        del main, output, source
        main_ref.close()
        dino_ref = np.load(REPO / cfg['dino_reference'], allow_pickle=False)
        for name, checkpoint_path in cfg['dino_checkpoints'].items():
            model = dinov2_vits14_reg(pretrained=False)
            model.pos_embed = torch.nn.Parameter(torch.zeros(1, 257, 384))
            teacher = torch.load(REPO / checkpoint_path, map_location='cpu', weights_only=False)['teacher']
            model.load_state_dict({k.removeprefix('backbone.'): v for k, v in teacher.items()
                                   if 'dino_head' not in k}, strict=True)
            del teacher
            model.cuda().eval()
            with torch.no_grad():
                tensor = model(torch.from_numpy(dino_ref['normalized_input']).cuda())
            compare(name, tensor, dino_ref[name])
            del model, tensor
        dino_ref.close()
        torch.cuda.synchronize()
        report.update(all_pass=True, peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_torch_reserved_bytes=torch.cuda.max_memory_reserved())
        run.save('gpu_parity.json', report)
        run.status['output_hashes'] = {'gpu_parity.json': file_sha(run.output / 'gpu_parity.json')}


if __name__ == '__main__':
    main()
