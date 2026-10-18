"""Recheck delivered files, executed test evidence, real runs and inference bundles.

This proves the implementation handoff and reported experiments. It deliberately
does not certify full-benchmark performance or missing real CSLR/SLT experiments.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from signrepr.config import config_hash, load_config
from signrepr.training.provenance import source_hash
from signrepr.data.schema import sha256_file
from signrepr.data.splits import validate_manifest
from signrepr.export.api import SignRepresentationModel


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/evidence/final_audit.json')
    args = parser.parse_args()
    evidence, runtime = ROOT/'docs/evidence', ROOT/'runtime'
    required = [
        'pyproject.toml', 'README.md', 'IMPLEMENTATION_STATUS.md', 'requirements-tested.lock.txt',
        'docs/method_provenance.md', 'docs/codebase_review.md', 'docs/sources.lock.json',
        'docs/data_protocol.md', 'docs/model_card.md', 'docs/requirements_audit.md',
        'docs/COMPLETION_REPORT_VI.md', 'examples/inference_cpu.py',
        *[f'configs/{name}.yaml' for name in ['ssl_pose_jepa', 'metric_gloss', 'islr_frozen',
                                            'islr_finetune', 'cslr_frozen', 'cslr_finetune', 'slt_frozen']],
    ]
    required += [str(path.relative_to(ROOT)) for path in (ROOT/'src/signrepr').rglob('*.py')]
    require(all((ROOT/path).is_file() for path in required), 'Missing implementation deliverable')
    for path in (ROOT/'configs').rglob('*.yaml'):
        if path.name != 'data_protocol.yaml': load_config(path)

    tree = ET.parse(evidence/'tests_torch251.xml')
    tests = tree.findall('.//testcase')
    require(tests and not tree.findall('.//failure') and not tree.findall('.//error')
            and not tree.findall('.//skipped'), 'Final test evidence has failures, errors or unexecuted cases')
    test_report = json.loads((evidence/'test_report.json').read_text())
    require(test_report['source_sha256'] == source_hash(), 'Final tests do not match current source')
    require(test_report['tests'] == len(tests) and test_report['exit_code'] == 0, 'Test report/JUnit mismatch')
    wheel = json.loads((evidence/'wheel_install.json').read_text())
    require(wheel['verified'], 'Wheel installation was not verified')
    require(wheel['bundled_sources_lock_sha256'] == sha256_file(ROOT/'docs/sources.lock.json')
            == sha256_file(ROOT/'src/signrepr/sources.lock.json'), 'Packaged source lock mismatch')
    lock = json.loads((ROOT/'docs/sources.lock.json').read_text())
    require(len(lock['reference_files']) == 13 and all(r['status'] == 'downloaded_reference_only'
            and len(r['sha256']) == 64 for r in lock['reference_files']), 'Pinned reference audit incomplete')

    data = validate_manifest(runtime/'aslc_sanity/all.jsonl')
    require(data['valid'] and data['coverage']['samples'] == 32 and not data['near_duplicate_candidates'],
            'Real sanity manifest audit failed')
    visual = json.loads((evidence/'real_visual_audit.json').read_text())
    require(visual['samples'] == 20 and len(visual['visual_review']['reviewed_pages']) == 4,
            'Twenty-video visual audit evidence is missing')
    terminals = json.loads((runtime/'pilot_controls/results.json').read_text())
    require(len(terminals) == 5 and all(r['status'] == 'complete' and r['exit_code'] == 0 for r in terminals),
            'Pilot process evidence is incomplete')
    pilot_summary = json.loads((evidence/'pilot_summary.json').read_text())
    pilot_runs = []
    for record in pilot_summary['runs']:
        name = record['run']; root = runtime/f'pilot5k_{name}'
        updates = [json.loads(line) for line in (root/'train.jsonl').read_text().splitlines()]
        successful = [row for row in updates if row['successful']]
        require(len(successful) == 5000 and successful[-1]['step'] == 5000, f'Pilot budget mismatch: {name}')
        state = torch.load(root/'last.pt', map_location='cpu', weights_only=False)
        require(state['step'] == 5000 and state['config']['model']['encoder_layers'] == 8
                and state['config']['model']['predictor_layers'] == 4, f'Pilot architecture/state mismatch: {name}')
        require(all(torch.isfinite(t).all() for key in ('system', 'teacher', 'criterion')
                    for t in state[key].values()), f'Nonfinite trained state: {name}')
        require(config_hash(state['config']) == state['metadata']['config_sha256'], f'Config hash mismatch: {name}')
        require(sha256_file(root/'best_validation.pt') == record['checkpoint_sha256'], f'Checkpoint/report mismatch: {name}')
        snapshot_sha = state['metadata']['code_sha256']
        snapshot = runtime/'source_snapshots'/snapshot_sha/'signrepr'
        digest = hashlib.sha256()
        for path in sorted(snapshot.rglob('*.py')):
            digest.update(str(path.relative_to(snapshot)).encode()); digest.update(path.read_bytes())
        require(digest.hexdigest() == snapshot_sha, f'Missing/mismatched source snapshot: {name}')
        pilot_runs.append({'run': name, 'updates': state['step'], 'source_snapshot_sha256': snapshot_sha})
        del state

    overfit = json.loads((evidence/'real_overfit_train_sanity.json').read_text())['metrics']
    require(overfit['num_samples'] == 16 and overfit['Top1_micro'] == 1., 'Real train overfit sanity not proven')
    bundles = []
    for folder, report in [('selected_pose_encoder', 'selected_bundle_cpu.json'),
                           ('hybrid_pose_encoder', 'hybrid_bundle_cpu.json')]:
        model = SignRepresentationModel(runtime/folder, 'cpu')
        parity = json.loads((evidence/report).read_text())
        require(parity['parity_verified'] and parity['bundle_sha256'] == model.metadata['bundle_sha256'],
                f'Bundle parity evidence mismatch: {folder}')
        bundles.append({'path': str(runtime/folder), 'method': model.metadata['method'],
                        'bundle_sha256': model.metadata['bundle_sha256']})
    metric_model = SignRepresentationModel(runtime/'metric_pose_encoder')
    require(metric_model.metric_head is not None and metric_model.metadata['selected_encoder'] == 'online',
            'Matched trained metric bundle is missing')
    smoke = torch.load(runtime/'final_verified_smoke/last.pt', map_location='cpu', weights_only=False)
    require(smoke['step'] == 20 and smoke['metadata']['code_sha256'] == source_hash(),
            'Final source smoke checkpoint does not match delivered implementation')
    require(smoke['metadata']['sources_lock_sha256'] == sha256_file(ROOT/'docs/sources.lock.json'),
            'Final checkpoint lacks current source-lock provenance')
    for report in ['selected_raw_test_retrieval.json', 'hybrid_raw_test_retrieval.json']:
        require(json.loads((evidence/report).read_text())['metrics']['num_queries'] == 8,
                'Frozen retrieval fixed denominator mismatch')
    report = {'implementation_handoff_verified': True, 'source_sha256': source_hash(),
              'tests': len(tests), 'test_scope': 'all repository correctness/workflow tests; CPU Gloo and single-GPU AMP',
              'wheel_verified': True, 'real_data_samples': data['coverage']['samples'],
              'visual_audit_videos': visual['samples'], 'pilots': pilot_runs, 'bundles': bundles,
              'required_artifact_checksums': {path: sha256_file(ROOT/path) for path in sorted(required)},
              'conditional_experiments_not_claimed': ['full benchmark/SOTA', 'full-budget Stage 2',
                                                     'real continuous CSLR/SLT', 'annotated minimal pairs',
                                                     'calibrated detector reproduction', 'multi-GPU NCCL'],
              'requirements_review': 'docs/requirements_audit.md'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({'implementation_handoff_verified': True, 'tests': len(tests), 'pilot_updates': 25000}))


if __name__ == '__main__': main()
