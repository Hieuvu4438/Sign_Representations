"""Close this direction's actual run/source/asset records, without new inference."""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'method/frozen_probe'))
sys.path.insert(0, str(ROOT))
from cache import file_sha
from gpu_accounting import inspect_effective_budget
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
        for name, digest in cfg.get('input_hashes', {}).items():
            if file_sha(REPO / name) != digest:
                raise ValueError('Registered closure input changed')
        entries = []
        for binding in cfg['runs']:
            path = REPO / binding['path']
            status = json.loads((path / 'status.json').read_text())
            if status['status'] not in ('completed', 'failed'):
                raise ValueError('Predecessor not terminal')
            if file_sha(path / 'config.json') != status['config_sha256']:
                raise ValueError('Original run config changed')
            environment = json.loads((path / 'environment.json').read_text())
            for name, digest in environment['code_hashes'].items():
                if file_sha(path / 'source_snapshot' / name) != digest:
                    raise ValueError('Original source snapshot changed')
            for name, digest in status.get('output_hashes', {}).items():
                if file_sha(path / name) != digest:
                    raise ValueError('Original output digest changed')
            if status['status'] == 'failed' and not (path / 'error.txt').is_file():
                raise ValueError('Failed attempt error not retained')
            alias = binding['record_alias']
            copy = REPO / 'method/research_lab/records/experiments' / (alias + '_status.json')
            if copy.read_bytes() != (path / 'status.json').read_bytes():
                raise ValueError('Common status copy differs')
            entries.append(dict(run_id=status['run_id'], path=binding['path'], status=status['status'],
                                status_sha256=file_sha(path / 'status.json'),
                                config_sha256=status['config_sha256'],
                                source_files=len(environment['code_hashes']),
                                output_hashes=status.get('output_hashes', {})))
        original_bindings = 0
        for name in cfg['frozen_gates']:
            gate = json.loads((REPO / name).read_text())
            for source, digest in {**gate.get('config_hashes', {}), **gate['file_hashes']}.items():
                if file_sha(REPO / source) != digest:
                    raise ValueError('Original scientific gate file changed')
                original_bindings += 1
        source_run = REPO / cfg['dino_source_run']
        manifest = json.loads((source_run / 'source_manifest.json').read_text())
        for member in manifest['files']:
            if file_sha(source_run / 'official_source' / member['path']) != member['sha256']:
                raise ValueError('Pinned architecture changed')
        landmark = json.loads((REPO / cfg['landmark_report']).read_text())
        for sample in landmark['samples']:
            if file_sha(sample['raw_path']) != sample['raw_sha256']:
                raise ValueError('Raw observation store missing/changed')
        native_checks = []
        if 'native_cuda_report' in cfg:
            import numpy as np
            native = json.loads((REPO / cfg['native_cuda_report']).read_text())
            inputs = json.loads((REPO / cfg['aligned_input_report']).read_text())
            if not native['completed'] or not native['all_bitwise_native_parity'] or native['original_CPU_GPU_gate_pass']:
                raise ValueError('Native compatibility cannot rescue original cross-device gate')
            if len(native['samples']) != 4 or len(inputs['samples']) != 4:
                raise ValueError('Fixed four-source coverage required')
            for feature, sample in zip(native['samples'], inputs['samples']):
                if feature['sample_id'] != sample['sample_id'] or feature['frames'] != sample['frames']:
                    raise ValueError('Source alignment differs')
                for field in ('feature', 'native_reference'):
                    if file_sha(feature[field + '_path']) != feature[field + '_sha256']:
                        raise ValueError('Native compatibility payload changed')
                if file_sha(sample['metadata_path']) != sample['metadata_sha256']:
                    raise ValueError('Aligned source metadata changed')
                metadata = json.loads(Path(sample['metadata_path']).read_text())
                with np.load(feature['feature_path'], allow_pickle=False) as arrays:
                    reference = np.load(feature['native_reference_path'], allow_pickle=False)
                    expected_mask = np.asarray([[row['streams'][name]['observed'] for name in
                        ('face', 'left_hand', 'right_hand', 'body_posture')] for row in metadata], dtype=bool)
                    checks = [np.array_equal(arrays['author_layer_last'], reference),
                              arrays['author_layer_last'].shape == (12, sample['frames'], 768),
                              np.isfinite(arrays['author_layer_last']).all(),
                              np.array_equal(arrays['observed_mask'], expected_mask),
                              np.array_equal(arrays['pts_start_s'], [row['pts_start_s'] for row in metadata]),
                              np.array_equal(arrays['same_detection'], [row['same_detection'] for row in metadata])]
                    if not all(checks):
                        raise ValueError('Native arrays or retained masks/PTS/duplicate flags differ')
                native_checks.append(dict(sample_id=sample['sample_id'], frames=sample['frames'],
                                          recomputed_bitwise_parity=True, metadata_fields_exact=True))
        ast_files = []
        for source in sorted(ROOT.rglob('*.py')):
            if 'runtime' in source.relative_to(ROOT).parts or '__pycache__' in source.parts:
                continue
            ast.parse(source.read_text(), filename=str(source))
            ast_files.append(str(source.relative_to(REPO)))
        budget_config = json.loads((REPO / cfg['budget_config']).read_text())
        effective = inspect_effective_budget(budget_config)
        if abs(effective['project_used_seconds'] - cfg['expected_program_seconds']) > 1e-6:
            raise ValueError('Unexpected cumulative resource spend')
        report = dict(audited_utc=datetime.now(timezone.utc).isoformat(),
                      scope='Same-agent prerequisite artifact integrity, not science or external audit',
                      runs=entries, original_frozen_file_checks=original_bindings,
                      dino_source_files=len(manifest['files']), raw_landmark_files=len(landmark['samples']),
                      raw_landmark_bytes=landmark['raw_bytes'], own_AST_files=ast_files,
                      effective_budget=effective,
                      document_hashes={name: file_sha(REPO / name) for name in cfg['documents']},
                      all_predecessors_terminal=True, ready_real_four_streams=False,
                      native_CUDA_four_clip_compatibility_verified=bool(native_checks),
                      native_payload_checks=native_checks, original_real_CPU_GPU_gate_pass=False,
                      natural_failure_ground_truth_verified=False, PAPER_READY=False)
        run.save('artifact_audit.json', report)
        record = REPO / cfg['record_output']
        with record.open('xb') as stream:
            stream.write((run.output / 'artifact_audit.json').read_bytes())
        run.status['output_hashes'] = {'artifact_audit.json': file_sha(run.output / 'artifact_audit.json')}


if __name__ == '__main__':
    main()
