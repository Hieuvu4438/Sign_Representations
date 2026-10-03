"""Matched frozen heads for same-source cue interventions; exploratory A03."""
import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

from _common import ROOT
from signrepr.cue_ablation import CONDITIONS, pool_conditions
from signrepr.io import read_jsonl, sha256, write_json
from signrepr.statistics import grouped_bootstrap

CONFIG = ROOT / 'configs/protocol_ncslgr_branch_ablation_v3.json'
FEATURES = ROOT / 'features/ncslgr_branch_ablation_v3'
OUTPUT = ROOT / 'runs/ncslgr_branch_ablation_v3'
MANIFEST = ROOT / 'data/external/ncslgr/diagnostic_v1/manifest.jsonl'
SOURCE_MANIFEST = ROOT / 'data/manifests/ncslgr_native_body_v1.jsonl'
NATIVE = ROOT / 'features/shubert_native_ncslgr_v1'
PREP = ROOT / 'features/shubert_native_ncslgr_v1_preprocessing'
COMMON = ROOT / 'runs/fair_ncslgr_v1/coverage.json'


def identities():
    paths = [Path(__file__), ROOT / 'scripts/extract_ncslgr_branch_ablation.py',
             ROOT / 'src/signrepr/cue_ablation.py', ROOT / 'scripts/run_ncslgr_probes.py',
             ROOT / 'scripts/extract_shubert_pilot.py', ROOT / 'src/signrepr/shubert.py',
             ROOT / 'src/signrepr/native_cache.py', ROOT / 'src/signrepr/io.py',
             ROOT / 'src/signrepr/ncslgr.py', ROOT / 'src/signrepr/statistics.py',
             ROOT / 'configs/shubert.yaml', ROOT / 'configs/protocol_native_ncslgr_v1.json',
             ROOT / 'configs/protocol_fair_ncslgr_v1.json', MANIFEST, SOURCE_MANIFEST,
             MANIFEST.parent / 'protocol.lock.json', COMMON, NATIVE / 'index.jsonl',
             NATIVE / 'extraction_report.json', NATIVE / 'cohort.lock.json',
             PREP / 'report.json', PREP / 'cohort.lock.json',
             ROOT / 'reports/ncslgr_branch_reproduction_diagnostic_v1.json']
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def lock():
    if CONFIG.exists() or FEATURES.exists() or OUTPUT.exists():
        raise ValueError('Preserve existing protocol and attempts')
    prior = json.loads((MANIFEST.parent / 'protocol.lock.json').read_text())
    population = json.loads(COMMON.read_text())['eligible_ids']
    if len(population) != 175 or len(set(population)) != len(population):
        raise ValueError('Previously audited strict common population differs')
    p = dict(status='LOCKED_EXPLORATORY_SAME_SOURCE_CUE_ABLATION',
             created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), identity_sha256=identities(),
             eligible_ids=population, conditions={k: list(v) for k, v in CONDITIONS.items()},
             classes=prior['classes'], seeds=prior['training_seeds'], lr_grid=prior['lr_grid'],
             epochs=prior['epochs'], weight_decay=prior['weight_decay'], head_parameters=2307,
             device='cpu', extraction_torch_threads=4, head_torch_threads=1,
             primary='all_minus_without_face_macro_recall', secondary='All fixed conditions; paired all-minus-other contrasts descriptive, no best-condition selection',
             selection='Final epoch validation macro recall, first LR wins ties; same three LR choices and 100 epochs per condition/seed; save every selection and selected head before any test inference.',
             intervention='Replace declared absent stream features by zeros before the frozen native encoder. Crop/frame/context/observed masks unchanged. Observation and intervention availability are separate.',
             pooling='Identical all-four-observed frames within known utterance support for every condition; mean then L2; per-condition standardization fitted on train only. No functional interval, caption or candidate text input.',
             population='Exact existing 175-utterance strict common cohort, 65 train / 27 val / 83 test. No automatic exclusions or class redesign.',
             within_run_repetition_tolerance=1e-5, minimum_free_disk_gib=20,
             original_cache_equivalence='NOT_CERTIFIED: original 1e-5 absolute cross-cache guard failed on a training clip. Earlier failure retained. Compare conditions against freshly recomputed all-stream baseline within this run; original-cache differences are descriptive audit values only.',
             decision_evidence='reports/ncslgr_branch_reproduction_diagnostic_v1.json: DINO and encoder repeat differences zero on failed training clip; old-cache RMSE6.2e-7, changing CPU threads does not remove old-cache mismatch. No classification metric examined in this diagnosis.',
             prospective_scope='Earlier NCSLGR test results known. New controlled zeroing contrast locked before its extraction/metrics; exploratory, not an independent test.',
             limits=['Zeroed modalities are out-of-distribution interventions, not a causal estimate of naturally absent information.',
                     'The encoder remains frozen; readouts are re-trained on each fixed intervention with equal budget.',
                     'All-zero stream features can retain positional/duration information through encoder biases and positional encoding.',
                     'Original contextual inference can include author-imputed crops even when pooling uses fully observed frames.',
                     'Original native-cache numerical equivalence is not certified; matched comparison uses its own recomputed all-stream baseline. Cross-cache drift is recorded, not hidden or treated as an encoder improvement.',
                     'Eligibility conditioning, one test signer and shared XML archive groups limit generalization; CIs condition on fixed heads and observed recordings.',
                     'Positive historical recorded function only; no verified absence, adjudicated full scope, external semantics, representation adaptation or method novelty.'])
    write_json(CONFIG, p)
    print(json.dumps(dict(status=p['status'], config_sha256=sha256(CONFIG), utterances=len(population), conditions=len(CONDITIONS))))


def prepare(p):
    from signrepr.ncslgr import inclusive_frame_bounds
    if identities() != p['identity_sha256']:
        raise ValueError('Protocol source identity changed')
    state = json.loads((FEATURES / 'status.json').read_text())
    report = json.loads((FEATURES / 'extraction_report.json').read_text())
    if state['status'] != 'PASS_SAME_SOURCE_CUE_ABLATION_FEATURES' or report['status'] != state['status'] or report['config_sha256'] != sha256(CONFIG):
        raise ValueError('Full intervention extraction required before fitting or test inference')
    index_rows = [r for _, r in read_jsonl(FEATURES / 'index.jsonl')]
    index = {r['utterance_sample_id']: r for r in index_rows}
    if len(index) != len(index_rows) or set(index) != set(p['eligible_ids']):
        raise ValueError('Exact registered common population required')
    rows = [r for _, r in read_jsonl(MANIFEST) if r['view_role']=='body' and r['utterance_sample_id'] in index]
    rows.sort(key=lambda r: r['utterance_sample_id'])
    if len(rows) != len(index):
        raise ValueError('Locked label/source population differs')
    values, coverage = [], []
    for row in rows:
        entry = index[row['utterance_sample_id']]; path = Path(entry['shard'])
        metadata = json.loads(path.with_suffix('.json').read_text())
        if not path.resolve().is_relative_to(FEATURES.resolve()) or metadata != entry or sha256(path) != entry['shard_sha256'] or entry['config_sha256'] != sha256(CONFIG):
            raise ValueError('Immutable intervention shard provenance differs')
        start, stop = inclusive_frame_bounds(row['utterance_start_ms'], row['utterance_end_ms'], 30)
        with np.load(path, allow_pickle=False) as data:
            if data['embeddings'].shape != (len(CONDITIONS), entry['frames'], 768) or data['condition_names'].tolist() != list(CONDITIONS):
                raise ValueError('Registered condition axes differ')
            pooled, keep = pool_conditions(data['embeddings'], data['stream_observed_mask'], data['center_sec'], start/30, stop/30)
        values.append(pooled)
        coverage.append(dict(sample_id=row['utterance_sample_id'], split=row['split'], label=row['label'],
                             pooled_frames=int(keep.sum()), source_frames=entry['frames'],
                             source_sha256=entry['source_sha256'], all_reproduction_max_abs_difference=entry['all_reproduction_max_abs_difference']))
    labels = {name: i for i, name in enumerate(p['classes'])}
    gold = np.asarray([labels[r['label']] for r in rows])
    split = {s: np.asarray([i for i, r in enumerate(rows) if r['split']==s], dtype=int) for s in ['train', 'val', 'test']}
    counts = {s: np.bincount(gold[i], minlength=len(labels)).tolist() for s, i in split.items()}
    write_json(OUTPUT / 'coverage.json', dict(rows=coverage, class_counts=counts, eligible_ids=[r['utterance_sample_id'] for r in rows],
               feature_report_sha256=sha256(FEATURES / 'extraction_report.json'), feature_index_sha256=sha256(FEATURES / 'index.jsonl')))
    if any(min(v)==0 for v in counts.values()) or {s:len(i) for s,i in split.items()} != dict(train=65, val=27, test=83):
        raise ValueError('Registered split/class support differs; no automatic redesign')
    return rows, np.stack(values), gold, split


def run():
    import torch
    from run_ncslgr_probes import metrics
    if OUTPUT.exists():
        raise ValueError('Preserve prior ablation readout attempt')
    p = json.loads(CONFIG.read_text())
    if p['status'] != 'LOCKED_EXPLORATORY_SAME_SOURCE_CUE_ABLATION' or identities() != p['identity_sha256']:
        raise ValueError('Locked protocol changed')
    OUTPUT.mkdir(parents=True); started = time.perf_counter()
    state = dict(status='VALIDATING', pid=os.getpid(), device='cpu', config_sha256=sha256(CONFIG))
    try:
        rows, raw, gold, split = prepare(p)
        torch.set_num_threads(p['head_torch_threads'])
        train, val, test = [split[s] for s in ['train', 'val', 'test']]
        y = torch.from_numpy(gold).long(); selected = {}; fit_started = time.perf_counter()
        state['status']='FITTING_VALIDATION_ONLY'; write_json(OUTPUT / 'status.json', state)
        for ci, condition in enumerate(CONDITIONS):
            value = raw[:, ci]; mean, sd = value[train].mean(0), value[train].std(0)
            x = torch.from_numpy(((value-mean)/np.maximum(sd, 1e-5)).astype(np.float32))
            for seed in p['seeds']:
                best, candidates = None, []
                for lr in p['lr_grid']:
                    torch.manual_seed(seed); model = torch.nn.Linear(768, len(p['classes']))
                    if sum(v.numel() for v in model.parameters()) != p['head_parameters']:
                        raise ValueError('Readout capacity mismatch')
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=p['weight_decay'])
                    for _ in range(p['epochs']):
                        optimizer.zero_grad(); loss=torch.nn.functional.cross_entropy(model(x[train]), y[train])
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite training loss')
                        loss.backward(); optimizer.step()
                    model.eval()
                    with torch.inference_mode():
                        measured=metrics(gold[val], model(x[val]).argmax(1).numpy())
                    candidates.append(dict(lr=lr, validation=measured, final_train_loss=float(loss.detach())))
                    if best is None or measured['macro_recall']>best['validation']['macro_recall']:
                        best=dict(model=model, lr=lr, validation=measured)
                directory=OUTPUT/f'{condition}_seed{seed}'; directory.mkdir()
                selection=dict(seed=seed, condition=condition, selected_lr=best['lr'], validation=best['validation'],
                               candidates=candidates, test_used_for_selection=False, config_sha256=sha256(CONFIG))
                write_json(directory/'selection.json', selection)
                torch.save(dict(state_dict=best['model'].state_dict(), train_mean=mean, train_sd=sd, selection=selection), directory/'head.pt')
                selected[(condition, seed)]=(best['model'], x, selection, directory)
        fitting=time.perf_counter()-fit_started
        # Every condition/seed has been selected and persisted before test access.
        state['status']='EVALUATING_FIXED_HEADS'; write_json(OUTPUT / 'status.json', state)
        results, correct = {}, {}
        groups=[rows[i]['recording_group'] for i in test]
        prior=int(np.bincount(gold[train], minlength=len(p['classes'])).argmax())
        prior_correct=np.tile((gold[test]==prior).astype(float), (len(p['seeds']), 1))
        for condition in CONDITIONS:
            records, tables = [], []
            for seed in p['seeds']:
                model,x,selection,directory=selected[(condition,seed)]
                with torch.inference_mode():
                    probability=model(x[test]).softmax(1).numpy(); prediction=probability.argmax(1)
                measured=metrics(gold[test], prediction);write_json(directory/'metrics.json', measured)
                predictions=[dict(sample_id=rows[i]['utterance_sample_id'], split='test', gold=int(gold[i]), prediction=int(prediction[j]),
                    correct=int(prediction[j]==gold[i]), recording_group=rows[i]['recording_group'], signer=rows[i]['signer'],
                    conservative_xml_group=rows[i]['conservative_xml_group'],
                    **{f'probability_{name}':float(probability[j,c]) for c,name in enumerate(p['classes'])}) for j,i in enumerate(test)]
                with (directory/'predictions.csv').open('w', newline='') as f:
                    writer=csv.DictWriter(f, fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
                records.append(dict(seed=seed, selection=selection, metrics=measured));tables.append((prediction==gold[test]).astype(float))
            correct[condition]=np.stack(tables)
            results[condition]=dict(seeds=records, bootstrap=grouped_bootstrap(gold[test], groups, correct[condition]))
        comparisons={condition:grouped_bootstrap(gold[test], groups, correct['all'], reference=correct[condition]) for condition in CONDITIONS if condition!='all'}
        summary=dict(status='PASS_EXPLORATORY_SAME_SOURCE_CUE_ABLATION', config_sha256=sha256(CONFIG), identity_sha256=identities(),
            elapsed_seconds=time.perf_counter()-started, fitting_seconds=fitting, head_parameters=p['head_parameters'], trained_heads=len(selected),
            split_counts={s:len(i) for s,i in split.items()}, coverage_sha256=sha256(OUTPUT/'coverage.json'), results=results,
            primary=comparisons['without_face'], secondary_all_minus_condition=comparisons,
            train_only_majority_prior=dict(label=p['classes'][prior], bootstrap=grouped_bootstrap(gold[test], groups, prior_correct)),
            limits=p['limits'], contribution_status='Matched frozen readout intervention diagnostic, no representation adaptation or established novelty.')
        write_json(OUTPUT/'summary.json',summary);state.update(status=summary['status'],elapsed_seconds=summary['elapsed_seconds']);write_json(OUTPUT/'status.json',state)
        print(json.dumps(dict(status=summary['status'], primary=summary['primary'])))
    except Exception as error:
        state.update(status='FAIL', failure_reason=type(error).__name__+': '+str(error), elapsed_seconds=time.perf_counter()-started)
        write_json(OUTPUT/'status.json', state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--lock-only',action='store_true')
    args=parser.parse_args();lock() if args.lock_only else run()
