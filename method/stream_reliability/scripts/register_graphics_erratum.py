"""Bind an observed incidental OpenGL context to a conservative full-wall charge."""
import argparse
import json
from pathlib import Path
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
        path = REPO / cfg['original_status_path']
        if file_sha(path) != cfg['original_status_sha256']:
            raise ValueError('Original completed detector status changed')
        status = json.loads(path.read_text())
        if status['status'] != 'completed' or status['H_id'] != 'H003':
            raise ValueError('Expected completed H003 detector run')
        report = dict(accounting_run_id=args.run_id, charged_run_id=status['run_id'], H_id=status['H_id'],
                      original_status_path=cfg['original_status_path'], original_status_sha256=file_sha(path),
                      original_GPU_hours=status['GPU_hours'], conservative_GPU_seconds=status['wall_seconds'],
                      observation=cfg['observed_console'],
                      observation_provenance='Actual unified execution session90929 stderr, captured in tool transcript; not a saved native stderr file',
                      reason='CPU TFLite delegate still created NVIDIA OpenGL contexts; torch CUDA_initialized=False does not exclude graphics resource use',
                      accounting='Replace cost for the same actual LANDMARK001 run, not a second GPU job; whole wall is an upper bound',
                      original_artifacts_preserved=True)
        run.save('graphics_accounting.json', report)
        run.status['output_hashes'] = {'graphics_accounting.json': file_sha(run.output / 'graphics_accounting.json')}


if __name__ == '__main__':
    main()
