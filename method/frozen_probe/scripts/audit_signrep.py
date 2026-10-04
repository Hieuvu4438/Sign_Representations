"""CPU-only strict loading audit against the upstream inference example."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time

REPO = Path(__file__).resolve().parents[3]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(4*1024*1024), b''):
            h.update(data)
    return h.hexdigest()


class IntegerProducts(ast.NodeTransformer):
    """The example has shape expressions such as 22 * 2; allow only int products."""
    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.op, ast.Mult) and isinstance(node.left, ast.Constant)
                and isinstance(node.right, ast.Constant)
                and type(node.left.value) is int and type(node.right.value) is int):
            return ast.copy_location(ast.Constant(node.left.value * node.right.value), node)
        raise ValueError('Unsupported expression in upstream configuration')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=REPO/'checkpoints/signrep/ckpt.pt')
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {'run_id': a.run_id, 'pid': os.getpid(), 'host': platform.node(),
              'status': 'running', 'start_utc': datetime.now(timezone.utc).isoformat(),
              'script_sha256': sha(__file__), 'GPU_hours': 0}
    def save(name, data):
        tmp = a.output_dir/(name+'.tmp')
        tmp.write_text(json.dumps(data, indent=2)+'\n')
        tmp.replace(a.output_dir/name)
    save('status.json', status)
    try:
        import torch
        torch.set_num_threads(1)
        upstream = REPO/'third_party/SignRep'
        example = upstream/'example_usage.py'
        assignments = [n for n in ast.walk(ast.parse(example.read_text()))
                       if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'model_params' for t in n.targets)]
        if len(assignments) != 1:
            raise ValueError('Ambiguous model configuration in upstream example')
        params = ast.literal_eval(IntegerProducts().visit(assignments[0].value))
        sys.path.insert(0, str(upstream))
        model_class = importlib.import_module('models.final_models.FINAL_hiera_latent_model_head_v25_active').Model
        model = model_class(**params)
        if not all(hasattr(model, name) for name in ['cls_head', 'fc', 'active_head']):
            raise RuntimeError('Upstream prior head creation was silently skipped')
        checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
        weights = checkpoint['model']
        expected = model.state_dict()
        missing, unexpected = sorted(set(expected)-set(weights)), sorted(set(weights)-set(expected))
        mismatched = {k: {'expected': list(expected[k].shape), 'actual': list(weights[k].shape)}
                      for k in set(expected)&set(weights) if expected[k].shape != weights[k].shape}
        covered = sum(v.numel() for k,v in expected.items() if k in weights and v.shape == weights[k].shape)
        report = {'run_id': a.run_id, 'torch': torch.__version__, 'device': 'cpu',
                  'checkpoint': str(a.checkpoint), 'checkpoint_sha256': sha(a.checkpoint),
                  'example_sha256': sha(example), 'model_params': params,
                  'model_file_sha256': sha(upstream/'models/final_models/FINAL_hiera_latent_model_head_v25_active.py'),
                  'missing_keys': missing, 'unexpected_keys': unexpected, 'mismatched_shapes': mismatched,
                  'state_tensor_numel_coverage': covered / sum(v.numel() for v in expected.values()),
                  'strict_load_passed': False, 'inference_parity': 'not_run',
                  'scope': 'loading only, not feature/benchmark parity'}
        save('coverage.json', report)
        model.load_state_dict(weights, strict=True)
        model.eval()
        report['strict_load_passed'] = True
        report['eval_mode'] = not model.training
        save('coverage.json', report)
        status.update(status='completed', exit_code=0, output_sha256=sha(a.output_dir/'coverage.json'))
    except Exception as e:
        status.update(status='failed', exit_code=1, error=repr(e))
    status.update(end_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic()-started)
    save('status.json', status)
    print(json.dumps(status))
    if status['status'] != 'completed':
        sys.exit(1)


if __name__ == '__main__':
    main()
