"""Author-imputed token pooling sensitivity, preserving original strict controls.

The decision is based on train-only missing-stream coverage. Uses all
source-supported native contextual tokens without
claiming their crops are observed. No labels/captions/core enter pooling.
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from _common import ROOT
from run_ncslgr_probes import metrics, normalized
from signrepr.io import read_jsonl, sha256, write_json
from signrepr.ncslgr import inclusive_frame_bounds
from signrepr.statistics import grouped_bootstrap

CONFIG = ROOT / 'configs/protocol_ncslgr_author_pool_v1.json'
OUTPUT = ROOT / 'runs/ncslgr_author_pool_v1'
NATIVE = ROOT / 'features/shubert_native_ncslgr_v1'
SIGNREP = ROOT / 'features/signrep_ncslgr_diagnostic_v1'
MANIFEST = ROOT / 'data/external/ncslgr/diagnostic_v1/manifest.jsonl'
NATIVE_PROTOCOL = ROOT / 'configs/protocol_native_ncslgr_v1.json'


def identities():
    files = [Path(__file__), ROOT / 'scripts/run_ncslgr_probes.py', ROOT / 'src/signrepr/io.py',
             ROOT / 'src/signrepr/ncslgr.py', ROOT / 'src/signrepr/statistics.py', MANIFEST,
             MANIFEST.parent / 'protocol.lock.json', NATIVE_PROTOCOL,
             SIGNREP / 'extraction_report.json', SIGNREP / 'index.jsonl']
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def lock():
    if CONFIG.exists() or OUTPUT.exists():
        raise ValueError('Preserve existing protocol or attempt')
    prior = json.loads((MANIFEST.parent / 'protocol.lock.json').read_text())
    training = {r['sample_id'] for _, r in read_jsonl(MANIFEST) if r['split'] == 'train' and r['view_role'] == 'body'}
    index_path = NATIVE / 'index.jsonl'
    snapshot = [r for _, r in read_jsonl(index_path) if r['sample_id'] in training]
    p = {'status': 'LOCKED_EXPLORATORY_AUTHOR_IMPUTED_FUNCTION_POOLING',
         'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'identity_sha256': identities(),
         'native_cache_identity': json.loads((NATIVE / 'cohort.lock.json').read_text()),
         'classes': prior['classes'], 'seeds': prior['training_seeds'], 'lr_grid': prior['lr_grid'],
         'epochs': prior['epochs'], 'weight_decay': prior['weight_decay'], 'head_parameters': 2307,
         'backbones': ['signrep', 'shubert_last', 'shubert_average'], 'device': 'cpu', 'torch_threads': 1,
         'selection': 'Final-epoch validation macro recall; same three LRchoices,100epochs,3seeds; deterministic first tie; save selected head before test.',
         'native_pooling': 'All native contextual tokens with source-frame centers inside known utterance support, including author carry-forward/black crop inputs. Observation masks audited separately; imputed is not observed.',
         'signrep_pooling': 'Nonpadded valid window centers inside same known utterance support. Mean then per-sample L2, original train-only mean/SD fit per backbone.',
         'population': 'Fixed222utterances/NEG-WH-YN; common eligible IDs with nonempty utterance-supported tokens. Explicit exclusions; stop if any class lacks train/val/test support.',
         'primary': 'shubert_last_minus_signrep_macro_recall', 'secondary': 'shubert_average_minus_signrep_macro_recall; no best-test-layer selection',
         'decision_evidence': {'native_index_snapshot_sha256': sha256(index_path), 'train_rows_inspected': len(snapshot),
                               'train_zero_all_four_stream_rows': sum(r['fully_observed_frames']==0 for r in snapshot),
                               'strict_summary_exists_at_lock': (ROOT / 'runs/fair_ncslgr_v1/summary.json').exists(),
                               'reason': 'Choice based on training missingness. Strict native result files already exist at lock; their metrics have not been inspected in this workflow before lock. Exploratory sensitivity, no certified blind or independent evaluation; preserve strict suites.'},
         'strict_original_protocol': 'configs/protocol_fair_ncslgr_v1.json',
         'limits': ['Exploratory missingness sensitivity, not replacement of strict observed-token protocol.',
                    'Earlier SignRep test metrics known; not independent confirmatory evaluation.',
                    'All source frames are decoded, but some cue crops are carried forward or black; source-frame support does not imply four observed modalities.',
                    'Native full context, cue crops and token rates differ from SignRep; no causal representation comparison.',
                    'One test signer and shared XML archive groups; conditional source-group CIs do not include signer-population/retraining uncertainty.',
                    'Positive recorded function only; no verified absence, full grammatical scope, semantic pair or method novelty claim.']}
    write_json(CONFIG, p)
    print(json.dumps({'status': p['status'], 'config_sha256': sha256(CONFIG), 'decision_evidence': p['decision_evidence']}))


def native_ready():
    state = json.loads((NATIVE / 'status.json').read_text())
    if state['status'] == 'RUNNING':
        return False, state
    if state['status'] != 'PASS':
        raise ValueError('Full native extraction did not PASS: '+state['status'])
    return (NATIVE / 'extraction_report.json').exists(), state


def prepare(p):
    if identities() != p['identity_sha256']:
        raise ValueError('Source/code identity changed after lock')
    rows = [r for _, r in read_jsonl(MANIFEST) if r['view_role'] == 'body']
    rows.sort(key=lambda r: r['utterance_sample_id'])
    expected = {r['sample_id'] for r in rows}
    if len(expected) != len(rows) or len({r['utterance_sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate body source or utterance IDs')
    indexes = {}
    for name, folder in [('native', NATIVE), ('signrep', SIGNREP)]:
        entries = [r for _, r in read_jsonl(folder / 'index.jsonl')]
        indexes[name] = {r['sample_id']: r for r in entries}
        if len(entries) != len(indexes[name]):
            raise ValueError('Duplicate feature IDs')
    if set(indexes['native']) != expected or not expected.issubset(indexes['signrep']):
        raise ValueError('Full native source coverage differs')
    report = json.loads((NATIVE / 'extraction_report.json').read_text())
    native_protocol = json.loads(NATIVE_PROTOCOL.read_text())
    if report['status'] != 'PASS' or report['manifest_sha256'] != native_protocol['manifest_sha256']:
        raise ValueError('Native terminal source identity differs')
    retained, features, excluded, observation = [], {b: [] for b in p['backbones']}, [], []
    for row in rows:
        a, b = inclusive_frame_bounds(row['utterance_start_ms'], row['utterance_end_ms'], 30)
        loaded, reasons = {}, []
        for name, index in indexes.items():
            entry = index[row['sample_id']]
            path = Path(entry['shard']); meta = json.loads(path.with_suffix('.json').read_text())
            if entry['status'] != 'SUCCESS' or sha256(path) != meta['shard_sha256']:
                raise ValueError('Immutable feature checksum/status differs')
            if name == 'native' and any(meta['cache_material'].get(k) != v for k,v in p['native_cache_identity'].items()):
                raise ValueError('Native cache identity differs')
            with np.load(path, allow_pickle=False) as data:
                clock = data['center_sec']
                if clock.ndim != 1 or not np.isfinite(clock).all() or not (np.diff(clock)>0).all():
                    raise ValueError('Source clock is invalid')
                valid, frame_valid = data['valid_mask'], data['valid_frame_mask']
                if valid.dtype != bool or valid.shape != clock.shape or frame_valid.dtype != bool or frame_valid.ndim != 2 or frame_valid.shape[0] != len(clock):
                    raise ValueError('Invalid source validity masks')
                support = (clock >= a/30) & (clock < b/30)
                if name == 'native':
                    stream_mask = data['stream_observed_mask']
                    if stream_mask.dtype != bool or stream_mask.shape != (len(clock),4):
                        raise ValueError('Invalid native stream observation mask')
                    observed = stream_mask.all(1)
                    if not np.array_equal(observed, valid & frame_valid.all(1)):
                        raise ValueError('Native observed mask differs')
                    observation.append({'sample_id': row['utterance_sample_id'], 'split': row['split'], 'label': row['label'],
                        'source_supported_frames': int(support.sum()), 'all_four_observed_supported_frames': int((support & observed).sum()),
                        'per_stream_observed_supported_frames': data['stream_observed_mask'][support].sum(0).tolist()})
                    keep = support
                    keys = {'shubert_last': 'embeddings', 'shubert_average': 'embeddings_layer_average'}
                else:
                    keep = support & data['valid_mask'] & data['valid_frame_mask'].all(1)
                    keys = {'signrep': 'embeddings'}
                if not keep.any():
                    reasons.append(name+'_ZERO_UTTERANCE_SUPPORTED_TOKENS')
                    continue
                for backbone,key in keys.items():
                    x = data[key]
                    if x.shape != (len(clock),768) or not np.isfinite(x).all():
                        raise ValueError('Invalid frozen features')
                    loaded[backbone] = normalized(x[keep].mean(0)).astype(np.float32)
        if reasons:
            excluded.append({'sample_id': row['utterance_sample_id'], 'split': row['split'], 'label': row['label'], 'reasons': reasons})
            continue
        retained.append(row)
        for backbone,value in loaded.items():
            features[backbone].append(value)
    labels = {name:i for i,name in enumerate(p['classes'])}
    gold = np.array([labels[r['label']] for r in retained], dtype=np.int64)
    split = {s: np.array([i for i,r in enumerate(retained) if r['split']==s], dtype=np.int64) for s in ['train','val','test']}
    counts = {s: np.bincount(gold[i], minlength=len(labels)).tolist() for s,i in split.items()}
    write_json(OUTPUT / 'coverage.json', {'eligible_ids': [r['utterance_sample_id'] for r in retained], 'class_counts': counts,
        'exclusions': excluded, 'observation_counts': observation, 'native_report_sha256': sha256(NATIVE / 'extraction_report.json'),
        'native_index_sha256': sha256(NATIVE / 'index.jsonl'), 'semantics': 'Native pooling includes author-imputed tokens; counts report observed crops separately.'})
    if any(min(v)==0 for v in counts.values()):
        raise ValueError('Locked class lacks split support; no automatic population redesign')
    return retained, {k:np.stack(v) for k,v in features.items()}, gold, split


def run(wait):
    if OUTPUT.exists():
        raise ValueError('Preserve prior author-pool attempt')
    p = json.loads(CONFIG.read_text())
    if p['status'] != 'LOCKED_EXPLORATORY_AUTHOR_IMPUTED_FUNCTION_POOLING' or identities() != p['identity_sha256']:
        raise ValueError('Protocol identity or status differs')
    OUTPUT.mkdir(parents=True)
    started = time.perf_counter()
    state = {'status': 'WAITING_FOR_NATIVE', 'pid': os.getpid(), 'device': 'cpu', 'config_sha256': sha256(CONFIG)}
    try:
        while True:
            ready,native = native_ready()
            if ready:
                break
            if not wait:
                raise ValueError('Partial native test inference is forbidden')
            state.update(native_completed=native['success'], native_samples=native['samples'], elapsed_seconds=time.perf_counter()-started,
                         updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            write_json(OUTPUT / 'status.json',state); time.sleep(30)
        state['status']='RUNNING'; write_json(OUTPUT / 'status.json',state)
        torch.set_num_threads(p['torch_threads'])
        rows,features,gold,split=prepare(p)
        train,val,test=[split[s] for s in ['train','val','test']]
        fit_started=time.perf_counter(); results,correct={},{}
        groups=[rows[i]['recording_group'] for i in test]
        for backbone,raw in features.items():
            mean,sd=raw[train].mean(0),raw[train].std(0)
            x=torch.from_numpy(((raw-mean)/np.maximum(sd,1e-5)).astype(np.float32));y=torch.from_numpy(gold).long()
            records,tables=[],[]
            for seed in p['seeds']:
                best,candidates=None,[]
                for lr in p['lr_grid']:
                    torch.manual_seed(seed);model=torch.nn.Linear(768,3)
                    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=p['weight_decay'])
                    if sum(v.numel() for v in model.parameters()) != p['head_parameters']:
                        raise ValueError('Head capacity differs')
                    for _ in range(p['epochs']):
                        optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(x[train]),y[train])
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite loss')
                        loss.backward();optimizer.step()
                    model.eval()
                    with torch.inference_mode():
                        score=metrics(gold[val],model(x[val]).argmax(1).numpy())
                    candidates.append({'lr':lr,'validation':score,'final_train_loss':float(loss.detach())})
                    if best is None or score['macro_recall']>best['validation']['macro_recall']:
                        best={'model':model,'lr':lr,'validation':score}
                directory=OUTPUT/f'{backbone}_seed{seed}';directory.mkdir()
                selection={'seed':seed,'selected_lr':best['lr'],'validation':best['validation'],'candidates':candidates,
                           'test_used_for_selection':False,'config_sha256':sha256(CONFIG)}
                write_json(directory/'selection.json',selection)
                torch.save({'state_dict':best['model'].state_dict(),'train_mean':mean,'train_sd':sd,'selection':selection},directory/'head.pt')
                with torch.inference_mode():
                    probability=best['model'](x[test]).softmax(1).numpy();prediction=probability.argmax(1)
                measured=metrics(gold[test],prediction);write_json(directory/'metrics.json',measured)
                prediction_rows=[{'sample_id':rows[i]['utterance_sample_id'],'split':'test','gold':int(gold[i]),'prediction':int(prediction[j]),
                    'correct':int(prediction[j]==gold[i]),'recording_group':rows[i]['recording_group'],'signer':rows[i]['signer'],
                    'conservative_xml_group':rows[i]['conservative_xml_group'],
                    **{f'probability_{name}':float(probability[j,c]) for c,name in enumerate(p['classes'])}} for j,i in enumerate(test)]
                with (directory/'predictions.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(prediction_rows[0]));writer.writeheader();writer.writerows(prediction_rows)
                records.append({'seed':seed,'selection':selection,'metrics':measured});tables.append((prediction==gold[test]).astype(float))
                print(json.dumps({'backbone':backbone,'seed':seed,'metrics':measured}),flush=True)
            correct[backbone]=np.stack(tables)
            results[backbone]={'seeds':records,'bootstrap':grouped_bootstrap(gold[test],groups,correct[backbone])}
        comparisons={b:grouped_bootstrap(gold[test],groups,correct[b],reference=correct['signrep']) for b in ['shubert_last','shubert_average']}
        summary={'status':'PASS_EXPLORATORY_AUTHOR_IMPUTED_FUNCTION_POOLING','config_sha256':sha256(CONFIG),'identity_sha256':identities(),
                 'coverage_sha256':sha256(OUTPUT/'coverage.json'),'elapsed_seconds':time.perf_counter()-started,'fitting_seconds':time.perf_counter()-fit_started,
                 'head_parameters':p['head_parameters'],'split_counts':{s:len(i) for s,i in split.items()},'results':results,
                 'primary':comparisons['shubert_last'],'secondary':comparisons['shubert_average'],'limits':p['limits'],
                 'contribution_status':'Separate frozen imputation-policy sensitivity, no encoder adaptation or established novelty.'}
        write_json(OUTPUT/'summary.json',summary);state.update(status=summary['status'],elapsed_seconds=summary['elapsed_seconds']);write_json(OUTPUT/'status.json',state)
    except Exception as error:
        state.update(status='FAIL',failure_reason=type(error).__name__+': '+str(error),elapsed_seconds=time.perf_counter()-started)
        write_json(OUTPUT/'status.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock-only',action='store_true');parser.add_argument('--wait-for-native',action='store_true')
    args=parser.parse_args();lock() if args.lock_only else run(args.wait_for_native)
