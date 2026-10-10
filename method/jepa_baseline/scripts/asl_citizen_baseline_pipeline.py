"""Resume-aware stage orchestration and evidence archival for a fixed ASL baseline."""
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def pin(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def corpus_check(folder):
    import numpy as np
    import torch
    from signrep_baseline.cache import load_standardized, clip_targets
    from signrep_baseline.preparation import validate_pretrain_audit
    torch.set_num_threads(2)
    audit = validate_pretrain_audit(folder / 'corpus/pretrain.jsonl')
    config = read(folder / 'run_config.json')
    manifest = rows(folder / 'corpus/pretrain.jsonl')
    assert len(manifest) == config['train_videos']
    assert [row['video_id'] for row in manifest] == [row['video_id'] for row in rows(folder / 'gallery.jsonl')]
    frames, valid_counts, all_counts = 0, {}, {}
    for row in manifest:
        assert pin(row['pose_cache']) == row['pose_cache_sha256']
        data, metadata = load_standardized(row['pose_cache'], row['bone_statistics_sha256'], row['video_sha256'])
        assert metadata['body_root_mode'] == 'canonical'
        frames += metadata['decoded_frames']
        observed = data['body_detected']
        starts = np.flatnonzero(np.convolve(observed.astype(int), np.ones(16, int), mode='valid') == 16)
        assert len(starts), row['video_id']
        _, masks = clip_targets(data, np.arange(starts[0], starts[0] + 16))
        for key, mask in masks.items():
            valid_counts[key] = valid_counts.get(key, 0) + int(mask.sum())
            all_counts[key] = all_counts.get(key, 0) + mask.numel()
    value = {'status': 'PASS', 'videos': len(manifest), 'decoded_training_frames': frames,
             'official_train_and_held_out_audit': True, 'body_root_mode': 'canonical',
             'corpus_bones_sha256': audit['bone_statistics_sha256'],
             'first_valid_clip_supervision_fraction': {key: valid_counts[key] / all_counts[key] for key in valid_counts}}
    save(folder / 'corpus_check.json', value)
    return value


def archive(folder):
    import numpy as np
    from signrep_baseline.cache import implementation_sha256
    config = read(folder / 'run_config.json')
    result = read(folder / 'RESULTS.json')
    verification = read(folder / 'VERIFICATION.json')
    assert result['status'] == 'COMPLETED' and verification['status'] == 'PASS'
    assert result['implementation_sha256'] == implementation_sha256()
    exp = ROOT / 'experiments' / config['experiment']
    contract = read(folder / 'train/run_contract.json')
    for name, expected in result['inventory']['manifest_sha256'].items():
        assert pin(folder / name) == expected
    assert pin(folder / 'initial.model.pt') == result['initial_sha256'] == contract['initial_checkpoint_sha256']
    assert pin(folder / 'final.model.pt') == result['final_sha256']
    phases = ['step0', f"step{config['total_steps']}"]
    provenance, source_pins, initial_starts = {}, {}, {}
    for phase, checkpoint in zip(phases, [result['initial_sha256'], result['final_sha256']]):
        windows = 0
        for part in ['gallery', 'queries']:
            records = rows(folder / 'retrieval' / phase / f'{part}_features.jsonl')
            original = rows(folder / f'{part}.jsonl')
            assert [row['video_id'] for row in records] == [row['video_id'] for row in original]
            for row in records:
                assert pin(row['features']) == row['features_sha256']
                if row['video_id'] not in source_pins:
                    source_pins[row['video_id']] = pin(row['video'])
                assert source_pins[row['video_id']] == row['video_sha256']
                with np.load(row['features'], allow_pickle=False) as data:
                    meta = json.loads(str(data['metadata_json']))
                    assert meta['encoder_checkpoint_sha256'] == checkpoint
                    assert meta['video_sha256'] == row['video_sha256']
                    assert meta['implementation_sha256'] == result['implementation_sha256']
                    assert meta['stride'] == config['retrieval_stride'] and meta['window_frames'] == 16 and meta['bbox'] is None
                    assert meta['inference_compute_dtype'] == 'bfloat16'
                    starts = np.arange(0, meta['decoded_frames'] - 15, config['retrieval_stride'])
                    assert np.array_equal(starts, data['window_starts'])
                    assert len(starts) == row['windows'] == len(data['features']) == len(data['activity'])
                    assert data['features'].dtype == np.float32 and data['activity'].dtype == np.float32
                    assert np.isfinite(data['features']).all() and np.isfinite(data['activity']).all()
                    if phase == 'step0':
                        initial_starts[row['video_id']] = starts
                    else:
                        assert np.array_equal(starts, initial_starts[row['video_id']])
                    windows += len(starts)
        provenance[phase] = {'status': 'PASS', 'videos': config['train_videos'] + config['val_queries'],
                             'windows': windows, 'checkpoint_sha256': checkpoint}
    assert provenance[phases[0]]['windows'] == provenance[phases[1]]['windows']
    comparison, changes, per_class = [], {}, []
    for mode in ['unweighted', 'activity_weighted']:
        before = read(folder / 'retrieval/step0' / f'{mode}_predictions.json')
        after = read(folder / 'retrieval' / phases[1] / f'{mode}_predictions.json')
        assert [row['query_id'] for row in before] == [row['query_id'] for row in after]
        changes[mode] = {'rank_improved': sum(b['rank'] < a['rank'] for a, b in zip(before, after)),
                         'rank_worsened': sum(b['rank'] > a['rank'] for a, b in zip(before, after)),
                         'rank_unchanged': sum(b['rank'] == a['rank'] for a, b in zip(before, after)),
                         'new_top5_hits': sum(a['rank'] > 5 and b['rank'] <= 5 for a, b in zip(before, after)),
                         'lost_top5_hits': sum(a['rank'] <= 5 and b['rank'] > 5 for a, b in zip(before, after))}
        for a, b in zip(before, after):
            comparison.append({'pooling': mode, 'query_id': a['query_id'], 'true_gloss': a['true_gloss'],
                               'step0_rank': a['rank'], 'final_rank': b['rank'],
                               'rank_improvement': a['rank'] - b['rank'],
                               'step0_nearest_gloss': a['nearest_gloss'], 'final_nearest_gloss': b['nearest_gloss']})
        for phase, predictions in zip(phases, [before, after]):
            for label in sorted({row['true_label'] for row in predictions}):
                selected = [row for row in predictions if row['true_label'] == label]
                ranks = np.array([row['rank'] for row in selected])
                per_class.append({'pooling': mode, 'phase': phase, 'label': label, 'gloss': selected[0]['true_gloss'],
                                  'queries': len(selected), 'top1_percent': float((ranks <= 1).mean() * 100),
                                  'top5_percent': float((ranks <= 5).mean() * 100),
                                  'MRR': float((1 / ranks).mean()), 'DCG': float((1 / np.log2(ranks + 1)).mean())})
    for name, records in [('query_comparison.csv', comparison), ('per_class_metrics.csv', per_class)]:
        with (exp / name).open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    save(exp / 'rank_changes.json', changes)
    for name in ['REPORT.md', 'RESULTS.json', 'VERIFICATION.json', 'INITIAL_VERIFICATION.json',
                 'run_config.json', 'inventory.json', 'corpus_check.json', 'pose_report.json']:
        shutil.copy2(folder / name, exp / name)
    for phase in phases:
        shutil.copy2(folder / 'retrieval' / phase / 'metrics.json', exp / f'{phase}_metrics.json')
    history = rows(folder / 'train/metrics.jsonl')
    save(exp / 'training_summary.json', {'steps': len(history), 'coverage': result['supervision_coverage'],
         'initial_100_mean_loss': float(np.mean([row['loss'] for row in history[:100]])),
         'final_100_mean_loss': float(np.mean([row['loss'] for row in history[-100:]])),
         'training_status': read(folder / 'training_status.json')})
    spec = importlib.util.spec_from_file_location('source_check', ROOT / 'scripts/verify_sources.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source_check = module.verify()
    save(exp / 'source_verification.json', source_check)
    review = f"""# {config['experiment']} post-execution review

Execution and numeric verification: PASS. All {config['total_steps']} updates completed, with finite losses/gradients and changed backbone/projection/prior tensors. Complete training state reaches the recorded budget. No released SignRep or previous pilot training weights were used; fresh Kinetics MAE initialization is pinned.

Data: {config['classes']} real classes; {config['train_videos']} official train/gallery videos; {config['val_queries']} official validation queries. Original signer/source split separation and full held-out exclusions are checked. Features in both phases have identical membership and window starts, and bind to their actual video/checkpoint/implementation hashes.

Arithmetic: independently implemented NumPy ranking and metrics match all recorded query ranks, top-five labels and nearest videos from separately replayed Torch float32 pooling/cosine scores. NumPy float64 pooling/cosine is a sensitivity audit; any differing ranks and metric values are explicitly recorded in the verification reports and certified against the measured score-rounding difference. Float32 pooling/cosine replay shares the original arithmetic backend and is not an independent implementation of those operations. Per-class scores and before/after query rankings are archived. These checks do not imply an independent full training rerun.

Limitations: one seed; reduced corpus, batch4 and {config['total_steps']}-step budget; BF16; corpus bone medians derived from selected train only; original full-frame inference. Activity valid-target coverage is {result['supervision_coverage']['activity']:.4%}. Step0 activity head is random. Any weighted score must be read with that coverage. Results are conditional on the fixed dictionary and cannot be compared directly with ASLC001 or full paper scores. No extra training or sample changes followed the metrics.
"""
    (exp / 'post_review.md').write_text(review)
    (exp / 'STATE.md').write_text(f"# {config['experiment']} execution state\n\nCOMPLETED: preparation, {config['total_steps']}-step training, fixed final retrieval and independent arithmetic verification completed.\n\nREPORT.md and RESULTS.json contain observed metrics. VERIFICATION.json,post_review.md,audit.json,per_class_metrics.csv and query_comparison.csv contain review evidence. Large artifacts remain under runtime/{config['experiment']}. This is a single-seed reduced-data/compute baseline; independent full training rerun absent.\n")
    audit = {'status': 'PASS', 'execution': 'completed', 'independent_full_training_rerun': False,
             'initialization': config['initialization_explanation'], 'initial_sha256': result['initial_sha256'],
             'final_sha256': result['final_sha256'], 'kinetics_sha256': pin(config['kinetics_checkpoint']),
             'implementation_sha256': result['implementation_sha256'], 'feature_cache_provenance': provenance,
             'raw_video_sha256': source_pins, 'rank_changes': changes, 'source_verification': source_check}
    files = list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'scripts').glob('*.py')) + list((ROOT / 'tests').rglob('*.py'))
    files += [path for path in exp.iterdir() if path.is_file() and path.name not in {'audit.json', 'pipeline.log'}]
    audit['reviewable_file_sha256'] = {str(path.relative_to(ROOT)): pin(path) for path in sorted(files)}
    audit['runtime_file_sha256'] = {str(path.relative_to(ROOT)): pin(path) for path in [folder / 'run_config.json',
          folder / 'train/run_contract.json', folder / 'train/metrics.jsonl', folder / 'train/last.pt',
          folder / 'corpus/pretrain.jsonl', folder / 'corpus/pretrain.jsonl.audit.json']}
    save(exp / 'audit.json', audit)
    print(json.dumps({'status': 'COMPLETED', 'report': str(exp / 'REPORT.md'), 'final': result['final']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['pipeline', 'corpus-check', 'archive'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true',
                        help='Prepare and audit the corpus, then stop before GPU retrieval/training')
    args = parser.parse_args()
    folder = args.output.resolve()
    config = read(folder / 'run_config.json')
    exp = ROOT / 'experiments' / config['experiment']
    if args.stage == 'corpus-check':
        print(json.dumps(corpus_check(folder), indent=2), flush=True)
        return
    if args.stage == 'archive':
        archive(folder)
        return
    # Hold one shared lock throughout every mutating pipeline invocation.
    lock = (folder / 'pipeline.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another pipeline invocation holds the corpus lock')
    started = time.monotonic()
    while not (folder / 'pose_report.json').exists():
        if time.monotonic() - started > config['stage_timeout_seconds']:
            raise TimeoutError('Pose stage has not produced its complete report within the stage budget')
        save(folder / 'pipeline_status.json', {'stage': 'waiting_for_pose_report', 'pid': os.getpid(), 'unix_time': time.time()})
        time.sleep(20)
    if not read(folder / 'pose_report.json')['all_usable']:
        raise ValueError('Resolve and record pose-invalid membership before observing retrieval metrics')
    pose_python = ROOT.parents[1] / '.venv-shubert-native/bin/python'
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    def run(stage, command, complete):
        if complete.exists():
            return
        save(folder / 'pipeline_status.json', {'stage': stage, 'pid': os.getpid(), 'unix_time': time.time()})
        try:
            with (exp / f'{stage}.log').open('a') as log:
                subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT,
                               timeout=config['stage_timeout_seconds'])
            if not complete.exists():
                raise ValueError(f'{stage} exited without required artifact {complete}')
        except Exception as error:
            save(folder / 'pipeline_status.json', {'stage': 'FAILED', 'failed_stage': stage,
                 'error': str(error), 'pid': os.getpid(), 'unix_time': time.time()})
            raise
    run('prepare', [sys.executable, '-m', 'signrep_baseline', 'prepare-corpus',
                   '--manifest', str(folder / 'source.jsonl'), '--held-out-manifest', str(folder / 'official_held_out.jsonl'),
                   '--training-reference-manifest', str(folder / 'official_train.jsonl'), '--output', str(folder / 'corpus'),
                   '--pose-python', str(pose_python), '--weights', str(ROOT / 'runtime/teacher/models')],
                   folder / 'corpus/pretrain.jsonl.audit.json')
    run('corpus_check', [sys.executable, __file__, 'corpus-check', '--output', str(folder)], folder / 'corpus_check.json')
    if args.prepare_only:
        save(folder / 'pipeline_status.json', {'stage': 'PREPARED_WAITING_FOR_GPU',
             'training_started': False, 'pid': os.getpid(), 'unix_time': time.time()})
        print('Corpus preparation and audit complete; GPU retrieval/training still pending.', flush=True)
        return
    import torch
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        save(folder / 'pipeline_status.json', {'stage': 'WAITING_FOR_GPU',
             'error': 'CUDA is unavailable in this execution environment',
             'pid': os.getpid(), 'unix_time': time.time()})
        raise RuntimeError('CUDA is unavailable; resume the same pipeline on a GPU-accessible environment')
    runner = ROOT / 'scripts/run_asl_citizen_baseline.py'
    verifier = ROOT / 'scripts/verify_asl_citizen_baseline.py'
    run('baseline', [sys.executable, str(runner), 'baseline', '--output', str(folder)], folder / 'retrieval/step0/metrics.json')
    run('initial_verification', [sys.executable, str(verifier), '--output', str(folder), '--initial-only'], folder / 'INITIAL_VERIFICATION.json')
    command = [sys.executable, str(runner), 'train', '--output', str(folder)]
    if (folder / 'train/last.pt').exists():
        command += ['--resume', str(folder / 'train/last.pt')]
    run('train', command, folder / 'RESULTS.json')
    run('verification', [sys.executable, str(verifier), '--output', str(folder)], folder / 'VERIFICATION.json')
    archive(folder)
    save(folder / 'pipeline_status.json', {'stage': 'COMPLETED', 'pid': os.getpid(), 'unix_time': time.time()})


if __name__ == '__main__':
    main()
