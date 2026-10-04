"""Bounded one-device property inspection to diagnose UUID formatting."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from budget import inspect_budget
from cache import file_sha
from resources import gpu_preflight
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    budget = inspect_budget([REPO / p for p in config['budget_runtime_roots']])
    observation = gpu_preflight(max_utilization=100)
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import torch
        if torch.cuda.device_count() != 1: raise ValueError('Diagnostic restricted to one visible GPU')
        run.start_gpu(budget)
        properties = torch.cuda.get_device_properties(0)
        run.save('report.json', dict(run_id=args.run_id, nvidia_smi=observation,
                 torch_uuid_string=str(properties.uuid), torch_uuid_repr=repr(properties.uuid),
                 torch_uuid_type=type(properties.uuid).__name__, torch_device_name=properties.name,
                 torch_total_memory=properties.total_memory, torch_version=torch.__version__,
                 scope='device property/UUID inspection only; no encoder or scores'))
        run.status['output_sha256'] = file_sha(run.output / 'report.json')


if __name__ == '__main__': main()
