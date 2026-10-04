"""Build a CPU-only evidence index from immutable runs, without scoring predictions."""
import argparse
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import file_sha
from run_context import RunContext


def inventory(run_path, single_output=None):
    status_path = run_path / 'status.json'
    status = json.loads(status_path.read_text())
    if status['status'] != 'completed':
        raise ValueError('Reproduction evidence must be a completed run')
    if file_sha(run_path / 'config.json') != status['config_sha256']:
        raise ValueError('Run config changed')
    outputs = dict(status.get('output_hashes', {}))
    if 'output_sha256' in status:
        if single_output is None:
            raise ValueError('Single-output digest needs an explicit filename')
        outputs[single_output] = status['output_sha256']
    for name, digest in outputs.items():
        if file_sha(run_path / name) != digest:
            raise ValueError('Recorded run output changed')
    if (run_path / 'environment.json').exists():
        environment = json.loads((run_path / 'environment.json').read_text())
    else:
        report = json.loads((run_path / 'report.json').read_text())
        environment = {k: report[k] for k in ('python', 'numpy', 'code_hashes')}
    hashes = environment['code_hashes']
    for name, digest in hashes.items():
        if file_sha(run_path / 'source_snapshot' / name) != digest:
            raise ValueError('Source snapshot changed')
    return dict(run_id=status['run_id'], protocol=status['protocol'], status=status['status'],
        path=str(run_path.relative_to(REPO)), status_sha256=file_sha(status_path),
        config_sha256=status['config_sha256'], output_hashes=outputs, code_hashes=hashes,
        source_snapshot=str((run_path / 'source_snapshot').relative_to(REPO)),
        recorded_process_command=shlex.join(status['command']), environment=environment,
        GPU_hours=status['GPU_hours'], wall_seconds=status['wall_seconds'], seed=status['seed'])


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        entries = [inventory(REPO / item['path'], item.get('single_output')) for item in config['completed_runs']]
        if len({entry['run_id'] for entry in entries}) != len(entries):
            raise ValueError('Duplicate reproduction entry')
        documents = {name: file_sha(REPO / name) for name in config['binding_documents']}
        statuses = []
        for root in config['budget_runtime_roots']:
            for path in sorted((REPO / root).glob('*/status.json')):
                status = json.loads(path.read_text())
                if status.get('GPU_allocated', False) or status['GPU_hours'] > 0:
                    statuses.append(dict(run_id=status['run_id'], H_id=status.get('H_id'), status=status['status'],
                        GPU_hours=status['GPU_hours'], status_sha256=file_sha(path), path=str(path.relative_to(REPO))))
        terminal = [s for s in statuses if s['status'] in ('completed', 'failed', 'interrupted', 'invalidated')]
        live = [s for s in statuses if s not in terminal]
        result = dict(completed_runs=entries, binding_documents=documents, all_owned_GPU_attempts=statuses,
            terminal_GPU_seconds=sum(s['GPU_hours'] * 3600 for s in terminal),
            live_attempts=live, live_cost='unreconciled; GPU_hours0 in running status is not zero spend',
            scope='provenance/reproduction index only; no test metric read or scorer invoked',
            storage='large raw train features remain in source-referenced tmpfs; ranks/manifests in ignored runtime',
            replay='commands are actual historical argv. Never rerun into completed outputs. Use matching source snapshots, verify input assets/gates and choose a fresh registered output/run ID.')
        run.save('reproduction.json', result)
        lines = ['# Reproduction index', '', 'Generated from completed statuses/configs/output digests/source snapshots; no task scores read.', '',
            '| Run | Protocol | Seed | GPU seconds | Wall seconds | Status and exact command |',
            '| --- | --- | ---: | ---: | ---: | --- |']
        for entry in entries:
            path = REPO / entry['path']
            lines.append(f"| {entry['run_id']} | {entry['protocol']} | {entry['seed']} | {entry['GPU_hours']*3600:.6f} | {entry['wall_seconds']:.6f} | [status]({path}/status.json) |")
        lines += ['', 'Actual commands below are historical evidence; completed output directories are immutable.',
                  'Replay needs the recorded source snapshot/config and a fresh registered output/run ID.',
                  'Thread environment and scheduler CUDA visibility are stored in each environment record.']
        for entry in entries:
            lines += ['', f"**{entry['run_id']}**", '', '```bash', entry['recorded_process_command'], '```']
        lines += ['', f"Terminal owned GPU cost: {result['terminal_GPU_seconds']:.6f}s; live attempts: {len(live)} (cost unreconciled).", '',
                  'This index verifies digests and snapshots. It does not establish scientific validity, test completion, population generalization or PAPER_READY.']
        (run.output / 'reproduction.md').write_text('\n'.join(lines) + '\n')
        run.status['output_hashes'] = {name: file_sha(run.output / name) for name in ('reproduction.json', 'reproduction.md')}


if __name__ == '__main__':
    main()
