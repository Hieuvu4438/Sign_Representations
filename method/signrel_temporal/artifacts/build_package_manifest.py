"""Build traceable resource/coverage and reproduction artifacts from actual files."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from signrel.utils import read_jsonl, sha256, atomic_json, code_hash
from signrel.config import ROOT
from signrel.data.cache import read_cache

matrix = json.loads(Path('artifacts/matrix_status.json').read_text())
assert matrix.get('test_evaluated') is True, 'Wait for actual test evaluation'
frozen = json.loads(Path('artifacts/code_freeze_pilot.json').read_text())['code_sha256']
run_paths = ['runs/pilot_a1_seed17', 'runs/pilot_a4_seed17'] + [r['run_dir'] for r in matrix['rows']]
compute = []
for directory in run_paths:
    path = Path(directory, 'run_manifest.json')
    if not path.exists():
        item = next(r for r in matrix['rows'] if r['run_dir'] == directory)
        compute.append({'run_dir': directory, 'status': item['state'], 'variant': item['variant'],
                        'seed': item['seed'], 'successful_updates': None,
                        'training_seconds': None, 'mean_eligible_relation_pairs_per_sample': None})
        continue
    m = json.loads(path.read_text())
    assert m['source_sha256'] == frozen, 'Scientific runs must share frozen training code'
    assert sha256(Path(directory, 'checkpoints/last.pt')) == m['checkpoint_sha256']
    counts, denominators, hand_denominators = [], [], []
    lags = Counter()
    norms = []
    events = 0
    with Path(directory, 'metrics.jsonl').open() as file:
        for line in file:
            event = json.loads(line)
            events += 1
            norms.append(event['grad_norm'])
            for micro in event['microbatches']:
                stats = micro['relation_stats']
                counts.append(stats.get('count', 0))
                denominators.append(micro['relation_denominator'])
                hand_denominators.append(micro['hand_denominators'])
                lags.update(stats.get('lag_counts', {}))
    assert events == m['successful_updates']
    compute.append({'run_dir': directory, 'status': m['status'], 'variant': m['variant'],
                    'seed': m['seed'], 'successful_updates': m['successful_updates'],
                    'effective_batch': m['effective_batch'], 'precision': m['precision'],
                    'training_seconds': m['elapsed_seconds'],
                    'seconds_per_successful_update': m['elapsed_seconds'] / max(1, m['successful_updates']),
                    'peak_allocated_vram_bytes': m['peak_vram_bytes'],
                    'mean_eligible_relation_pairs_per_sample': float(np.mean(counts) / 8),
                    'mean_relation_weight_denominator_per_microbatch': float(np.mean(denominators)),
                    'mean_hand_weight_denominators_per_microbatch': np.mean(hand_denominators, axis=0).tolist(),
                    'eligible_relation_lag_counts_all_microbatches': dict(sorted(lags.items())),
                    'gradient_norm_finite': bool(np.isfinite(norms).all()),
                    'init_checkpoint_sha256': m['init_checkpoint_sha256'],
                    'window_mask_plan_sha256': sha256(Path(directory, 'window_plan.jsonl')),
                    'source_sha256': m['source_sha256'], 'checkpoint_sha256': m['checkpoint_sha256'],
                    'source_archive_sha256': sha256(Path(directory, 'source.tar.gz'))})

coverage = []
cache_files = {}
for manifest in ('data/cache/cache_manifest.jsonl', 'data/cache/ssl_dev_cache_manifest.jsonl',
                 'data/cache/eval_cache_manifest.jsonl'):
    rows = read_jsonl(manifest)
    clips = []
    for row in rows:
        arrays, meta = read_cache(row['cache_path'])
        assert meta['preprocess_fingerprint'] == 'e8b618580dd6913574e24c053128c9e2e314b78e8b95d388f74c5e23f19865e1'
        assert sha256(row['cache_path']) == row['cache_sha256']
        cache_files[row['cache_path']] = row['cache_sha256']
        cache_files[str(Path(row['cache_path']).with_suffix('.json'))] = sha256(Path(row['cache_path']).with_suffix('.json'))
        clips.append({'video_id': row['video_id'], 'split': row.get('split'),
                      'frames': len(arrays['frame_valid']),
                      'LH': float(arrays['app_valid'][:, 0].mean()),
                      'RH': float(arrays['app_valid'][:, 1].mean()),
                      'BODY': float(arrays['body_valid'].mean()),
                      'extraction_seconds': meta['extraction_seconds']})
    coverage.append({'manifest': manifest, 'manifest_sha256': sha256(manifest),
                     'videos': len(clips), 'canonical_frames': sum(r['frames'] for r in clips),
                     'mean_hand_valid_frame_fraction': [float(np.mean([r[k] for r in clips])) for k in ('LH', 'RH')],
                     'mean_body_valid_frame_fraction': float(np.mean([r['BODY'] for r in clips])),
                     'actual_extraction_seconds': sum(r['extraction_seconds'] for r in clips), 'per_video': clips})
ledger = json.loads(Path('artifacts/budget_ledger.json').read_text())
resource = {'status': 'measured', 'training_runs': compute, 'cache_coverage': coverage,
            'global_budget_ledger_sha256': sha256('artifacts/budget_ledger.json'),
            'global_caps': ledger['global_caps'], 'actual_used': ledger['used'],
            'inference_profile': json.loads(Path('artifacts/inference_profile.json').read_text()),
            'component_profile': json.loads(Path('artifacts/component_profile.json').read_text()),
            'scope': 'GPU seconds are conservative process wall times while owning CUDA work, not utilization-integrated energy. Shared-GPU interference affects elapsed time; extraction is charged once and reused by controls. Frame-valid fractions are coverage, not annotated detector recall.'}
atomic_json('results/compute_coverage.json', resource)
text = ['# Measured compute and coverage', '',
        '| Run | Status | Updates | Train seconds | Seconds/update | Eligible relation pairs/sample | Peak allocated MB |',
        '|---|---|---:|---:|---:|---:|---:|']
for row in compute:
    if row['training_seconds'] is None:
        text.append(f"| {row['run_dir']} | {row['status']} | — | — | — | — | — |")
    else:
        text.append(f"| {row['run_dir']} | {row['status']} | {row['successful_updates']} | {row['training_seconds']:.2f} | {row['seconds_per_successful_update']:.3f} | {row['mean_eligible_relation_pairs_per_sample']:.2f} | {row['peak_allocated_vram_bytes']/1e6:.2f} |")
text += ['', 'Effective batch32; microbatch8, accumulation4; BF16 on CUDA. A1 performs no relation computation. Counts are actual eligible pairs; quality weighting denominators and signed lag buckets are in compute_coverage.json. Differing timing under shared-GPU contention is not a method speed claim.', '',
         '| Cache manifest | Videos | Frames | LH frame coverage | RH frame coverage | BODY frame coverage | Extraction seconds |',
         '|---|---:|---:|---:|---:|---:|---:|']
for row in coverage:
    text.append(f"| {row['manifest']} | {row['videos']} | {row['canonical_frames']} | {100*row['mean_hand_valid_frame_fraction'][0]:.2f}% | {100*row['mean_hand_valid_frame_fraction'][1]:.2f}% | {100*row['mean_body_valid_frame_fraction']:.2f}% | {row['actual_extraction_seconds']:.2f} |")
text += ['', resource['scope'], '', '## Global resource ledger', '',
         f"Measured GPU process time: {ledger['used']['gpu_seconds']/3600:.4f} hours / {ledger['global_caps']['gpu_hours']} hours; training wall: {ledger['used']['wall_training_seconds']/60:.2f} minutes / {ledger['global_caps']['wall_training_minutes']} minutes; downloads: {ledger['used']['download_bytes']/1e6:.2f} MB; cache ledger: {ledger['used']['new_cache_bytes']/1e6:.2f} MB.", '',
         '## Inference and ranking', '',
         f"On CPU/four threads, frozen A4 dev cached NPZ validation + encoder/window/pooling averaged {resource['inference_profile']['cached_export_mean_seconds_per_video']:.5f} seconds/video over120 clips; one-time restore {resource['inference_profile']['restore_model_seconds']:.5f} seconds. Saved-vector retrieval averaged {resource['inference_profile']['ranking_mean_seconds']:.5f} seconds for80 queries ×40 gallery clips over5 repetitions. Raw perception/teacher and output-file writing are excluded; see the profile receipt for exact scope.", '']
Path('results/compute_coverage.md').write_text('\n'.join(text))

# No circular phase-state/reproducibility hashes: final phase receipts reference
# this manifest; their own bytes are verified in the separate final receipt.
files = {}
for folder in ('src', 'tests', 'configs', 'docs'):
    for path in sorted(Path(folder).rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.yaml', '.md'):
            files[str(path)] = sha256(path)
for path in [Path('README.md'), Path('RUNBOOK.md'), Path('MODEL_CARD.md'),
             Path('requirements-core.lock.txt'), Path('requirements-perception.lock.txt'),
             Path('third_party/NOTICE.md')]:
    files[str(path)] = sha256(path)
for folder in ('artifacts', 'results', 'data/manifests', 'data/splits'):
    for path in sorted(Path(folder).rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and 'cache_qa' not in path.parts and path.suffix in ('.json', '.jsonl', '.csv', '.npy', '.pt', '.md', '.log', '.py'):
            if path.name not in ('phases.json', 'reproducibility_manifest.json', 'final_package_verification.json'):
                files[str(path)] = sha256(path)
for directory in run_paths:
    for path in sorted(Path(directory).rglob('*')):
        if path.is_file() and path.suffix in ('.pt', '.json', '.yaml', '.jsonl', '.gz', '.txt', '.patch'):
            files[str(path)] = sha256(path)
for path in ('data/cache/cache_manifest.jsonl', 'data/cache/ssl_dev_cache_manifest.jsonl', 'data/cache/eval_cache_manifest.jsonl'):
    files[path] = sha256(path)
files.update(cache_files)
atomic_json('artifacts/reproducibility_manifest.json', {
    'schema_version': 1, 'status': 'actual_artifacts_hashed',
    'current_package_source_sha256': code_hash(ROOT), 'scientific_training_source_sha256': frozen,
    'historical_training_source': 'Every real run has a source.tar.gz archive; resume must use its exact archived source/config environment. A later validation-only patch does not retroactively change the source of a run.',
    'files_sha256': files, 'scientific_run_count': sum(r['training_seconds'] is not None for r in compute),
    'matrix_status': matrix['status'],
    'restricted_assets': 'Dataset videos, numeric caches and previews stay local; original video hashes are in canonical manifests, local terms are separately hashed. This manifest is provenance, not a redistribution license.'})
print('Compute, coverage and reproducibility files generated from actual terminal state.')
