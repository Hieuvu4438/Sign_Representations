"""Fixed source-only head fit, full saved dictionary scoring for one init seed."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounded_artifacts import verify_store
from bounded_gpu_context import BoundedGPUContext
from limited_gpu_resources import configure_gpu_2gib
from resources import scratch_preflight, gpu_preflight
from rejection_heads import HEAD_NAMES, make_heads, balanced_logit_loss, selected_bce_indices, flat_parameters
from paired_rejection_episodes import paired_inputs
from msasl_saved_features import load_complete_features
from frozen_rejection import fit_source_known_threshold
from open_set_evaluation import JointOpenSet


def read(path): return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser()
    for name in ('config', 'run-id', 'output-dir'): p.add_argument('--' + name, required=True)
    p.add_argument('--seed', type=int, required=True); a = p.parse_args()
    with BoundedGPUContext(a.output_dir, a.run_id, a.config, a.seed) as run:
        c = run.config; resource = scratch_preflight('/dev/shm', 512 << 20)
        pool = Path(c['pool_runtime']); measure = Path(c['measured_runtime'])
        verify_store(pool); verify_store(measure)
        vectors = np.load(pool / 'fit_vectors.npy', allow_pickle=False)
        episodes = np.load(pool / 'training_plan.npy', allow_pickle=False)
        labels = np.array([r['class_id'] for r in read(pool / 'source_roster.json')])
        assert episodes.shape == (1000, 16, 52)
        mc = read(measure / 'config.json'); meta = read(measure / 'rejection_metadata.json')
        predictions = np.load(measure / 'predictions.npy', allow_pickle=False)
        ranks = np.load(measure / 'known_rank.npy', allow_pickle=False)
        base = np.load(measure / 'base_vectors.npy', allow_pickle=False)
        fc = read(Path(mc['feature_runtime']) / 'config.json'); items = read(fc['extraction_manifest'])
        ms, _, features_check = load_complete_features(mc['feature_runtime'], items)
        base_pos = {key: i for i, key in enumerate(meta['base_ids'])}
        ms_pos = {row['sample_id']: i for i, row in enumerate(items)}
        source_values = {'ASLCitizen_base': (base, base_pos), 'MSASL100': (ms, ms_pos)}
        query_values = np.empty((1412, 768), dtype=np.float32)
        for part in meta['parts']:
            values, positions = source_values[part['corpus']]
            query_values[part['start']:part['stop']] = values[[positions[key] for key in meta['query_ids'][part['start']:part['stop']]]].astype(np.float32)
        import torch
        torch.set_num_threads(1)
        sys.path.insert(0, 'method/stream_reliability'); from gpu_accounting import inspect_effective_budget
        budget = inspect_effective_budget(c); observation = gpu_preflight(c['min_free_mib'], c['max_gpu_utilization'])
        assert observation['uuid'] == c['granted_uuid']
        run.start_gpu(budget); device = configure_gpu_2gib(observation)
        heads = make_heads(a.seed, device)
        optimizers = [torch.optim.Adam(head.parameters(), lr=.001, weight_decay=.0001) for head in heads]
        losses = np.zeros((1000, 2), dtype=np.float64); counts = np.zeros((1000, 3), dtype=np.int16)
        torch.cuda.synchronize(); start = time.monotonic()
        for step, episode in enumerate(episodes):
            residual, similarity, targets = paired_inputs(vectors, labels, episode)
            x = torch.from_numpy(residual).to(device)
            sim = torch.from_numpy(similarity).to(device); target = torch.from_numpy(targets).to(device)
            selection = selected_bce_indices(sim, target)
            if selection is None: continue
            indices, binary = selection
            positive, negative = int(binary.sum().item()), int((binary == 0).sum().item())
            assert positive == negative and positive <= 16
            counts[step] = [positive, negative, 1]
            for hi, (head, optimizer) in enumerate(zip(heads, optimizers)):
                optimizer.zero_grad(set_to_none=True)
                loss = balanced_logit_loss(head(x), sim, target); assert loss is not None
                loss.backward(); optimizer.step(); losses[step, hi] = float(loss.detach().cpu())
            if (step + 1) % 100 == 0: print('source head fit', a.seed, step + 1, '/', 1000)
        torch.cuda.synchronize(); training_seconds = time.monotonic() - start
        scores = np.zeros((3, 3, 2, 1412), dtype=np.float32)
        y = np.asarray(meta['labels']); start = time.monotonic()
        with torch.inference_mode():
            for si, seed in enumerate(meta['support_seeds']):
                for ni, size in enumerate(meta['gallery_sizes']):
                    for part in meta['parts']:
                        entry = next(g for g in meta['gallery_metadata'] if g['part'] == part['name'] and g['seed'] == seed)
                        gallery = entry['galleries'][str(size)]; values, positions = source_values[part['corpus']]
                        support = values[[positions[key] for key in gallery['support_ids']]].astype(np.float32)
                        classes = gallery['class_ids']; sl = slice(part['start'], part['stop'])
                        columns = np.asarray([classes.index(int(label)) for label in predictions[si, ni, sl]])
                        residual = query_values[sl] - support[columns]
                        x = torch.from_numpy(residual).to(device)
                        for hi, head in enumerate(heads):
                            scores[si, ni, hi, sl] = head(x).cpu().numpy()
        torch.cuda.synchronize(); scoring_seconds = time.monotonic() - start
        assert np.isfinite(scores).all()
        flat, schema = flat_parameters(heads)
        assert flat.shape == (2411650,) and flat.dtype == np.float32
        run.save_array('head_weights.npy', flat, c['array_specs']['head_weights.npy'])
        run.save_array('head_scores.npy', scores, c['array_specs']['head_scores.npy'])
        run.save_array('train_losses.npy', losses, c['array_specs']['train_losses.npy'])
        run.save_array('training_counts.npy', counts, c['array_specs']['training_counts.npy'])
        calibration = {}; cal = meta['parts'][0]; cal_sl = slice(cal['start'], cal['stop'])
        for si, seed in enumerate(meta['support_seeds']):
            entry = next(g for g in meta['gallery_metadata'] if g['part'] == cal['name'] and g['seed'] == seed)
            for ni, size in enumerate(meta['gallery_sizes']):
                known = np.isin(y[cal_sl], entry['galleries'][str(size)]['class_ids'])
                for hi, method in enumerate(HEAD_NAMES):
                    calibration[f'{seed}:{size}:{method}'] = fit_source_known_threshold(scores[si, ni, hi, cal_sl], known, 19, 20)
        rows = []
        for si, seed in enumerate(meta['support_seeds']):
            for ni, size in enumerate(meta['gallery_sizes']):
                for part in meta['parts']:
                    sl = slice(part['start'], part['stop'])
                    entry = next(g for g in meta['gallery_metadata'] if g['part'] == part['name'] and g['seed'] == seed)
                    classes = entry['galleries'][str(size)]['class_ids']; known = np.isin(y[sl], classes)
                    kr = ranks[si, ni, sl][known].astype(np.float64)
                    retrieval = dict(Top1=float(np.mean(kr == 1)), Top5=float(np.mean(kr <= 5)),
                        DCG_FP64=float(np.mean(1 / np.log2(kr + 1))), MRR=float(np.mean(1 / kr)))
                    for hi, method in enumerate(HEAD_NAMES):
                        evaluator = JointOpenSet(meta['query_ids'][part['start']:part['stop']], y[sl],
                            predictions[si, ni, sl], scores[si, ni, hi, sl], classes,
                            meta['query_ids'][part['start']:part['stop']])
                        curve = evaluator.curve()
                        for policy in ('FROZEN50', 'MATCHED_SOURCE_SIZE'):
                            tau = calibration[f'{seed}:{50 if policy == "FROZEN50" else size}:{method}']['threshold']
                            point = evaluator.at_threshold(tau)
                            rows.append(dict(training_seed=a.seed, part=part['name'], support_seed=seed,
                                size=size, method=method, policy=policy, point=point,
                                known_retrieval=retrieval, adapted_grouped_trapezoid_AUC=curve['grouped_trapezoid_AUC']))
        assert len(rows) == 108
        memory = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        assert memory['peak_reserved_bytes'] <= 2 << 30
        run.save('head_evaluation.json', dict(training_seed=a.seed, score_names=list(HEAD_NAMES),
            calibration=calibration, metric_rows=rows, parameter_schema=schema,
            actual_episode_attempts=1000, actual_updates=int(counts[:, 2].sum()),
            training_seconds=training_seconds, scoring_seconds=scoring_seconds, memory=memory,
            features_check=features_check, resource=resource, torch_version=torch.__version__,
            device_observation=observation, new_encoder_calls=0,
            scope='One fixed initialization, all three support galleries/sizes; no combined primary gate or novelty claim. Additional sourcefit1027labels, sourcecal286labels; exposed MS validation exploratory.'))


if __name__ == '__main__': main()
