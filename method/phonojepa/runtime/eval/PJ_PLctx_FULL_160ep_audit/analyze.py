"""Read-only audit of existing PLctx runs; writes only sibling report artifacts."""
from pathlib import Path
from datetime import datetime
import collections
import csv
import hashlib
import json
import math
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def csvout(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


metrics = ['loss', 'ctx', 'teacher_entropy', 'gnorm', 'lr', 'near_hidden',
           'far_hidden', 'cls_mean_cos', 'tok_rankme', 'tok_std']
audit = {'created': datetime.now().astimezone().isoformat(),
         'status': 'ANALYZED; existing logs/checkpoints/ranks inspected, no training rerun',
         'runs': {}, 'sources': {}}
all_phases, all_epochs = [], []
for name in ['PJ_PLctx_FULL_160ep', 'PJ_PLctx_16k']:
    run = ROOT / 'runtime/runs' / name
    man = read(run / 'manifest.json')
    rows = jsonl(run / 'train_log.jsonl')
    total = man['steps_done']
    steps_per_epoch = man['n_clips'] // man['config']['batch']
    expected = [1] + list(range(50, total + 1, 50))
    entry = {'manifest': man, 'log_rows': len(rows),
             'expected_logging_steps_match': [r['step'] for r in rows] == expected,
             'last_logged_step': rows[-1]['step'],
             'first_log': rows[0], 'last_log': rows[-1],
             'last_diagnostics': [r for r in rows if 'tok_rankme' in r][-1],
             'nonfinite_logged_values': sum(not math.isfinite(v) for r in rows
                                           for v in r.values() if isinstance(v, (float, int))),
             'clips_drawn': total * man['config']['batch'],
             'average_draws_per_clip': total * man['config']['batch'] / man['n_clips'],
             'steps_per_second': total / man['seconds'],
             'clips_per_second': total * man['config']['batch'] / man['seconds'],
             'logged_gradient_clip_fraction': float(np.mean([r['gnorm'] > 1 for r in rows])),
             'summary': {}, 'phases': []}
    for k in metrics:
        vals = np.array([r[k] for r in rows if k in r])
        entry['summary'][k] = dict(zip(['min', 'p01', 'median', 'p99', 'max'],
                                     np.quantile(vals, [0, .01, .5, .99, 1]).tolist()))
        entry['summary'][k]['n'] = len(vals)
    for lo, hi in [(0, .1), (.1, .25), (.25, .5), (.5, .75), (.75, .9), (.9, 1)]:
        rr = [r for r in rows if lo * total < r['step'] <= hi * total]
        phase = {'run': name, 'budget_start_pct': lo * 100, 'budget_end_pct': hi * 100,
                 'logged_batches': len(rr)}
        for k in metrics:
            vals = np.array([r[k] for r in rr if k in r])
            phase[k + '_mean'] = float(vals.mean()) if len(vals) else None
            phase[k + '_median'] = float(np.median(vals)) if len(vals) else None
        entry['phases'].append(phase)
        all_phases.append(phase)
    for ep in range(1, math.ceil(total / steps_per_epoch) + 1):
        rr = [r for r in rows if (ep - 1) * steps_per_epoch < r['step'] <= ep * steps_per_epoch]
        e = {'run': name, 'epoch': ep, 'logged_batches': len(rr)}
        for k in metrics:
            vals = [r[k] for r in rr if k in r]
            e[k + '_mean'] = float(np.mean(vals)) if vals else None
        all_epochs.append(e)
    audit['runs'][name] = entry
    audit['sources'][str(run / 'manifest.json')] = sha(run / 'manifest.json')
    audit['sources'][str(run / 'train_log.jsonl')] = sha(run / 'train_log.jsonl')

name = 'PJ_PLctx_FULL_160ep'
state = read(ROOT / 'runtime/launch' / (name + '.status.json'))
stdout = (ROOT / 'runtime/launch' / (name + '.train.log')).read_text()
audit['training_checks'] = {
    'source_file_count': len(state['source_sha256']),
    'source_hash_mismatches': [p for p, h in state['source_sha256'].items() if sha(ROOT / p) != h],
    'skipped_nonfinite_steps': stdout.count('skip non-finite step'),
    'tracebacks': stdout.count('Traceback (most recent call last)'),
    'collapse_alarm_entries': stdout.count('"collapse_alarm"'),
    'completion_line_present': 'DONE 200640' in stdout,
    'stage_exit_code': state['stages'][-1]['exit_code'],
}
splits = {}
for split in ['train', 'val', 'test']:
    path = Path('/home/dongvk/datasets/ASL_Citizen/splits') / (split + '.csv')
    rr = list(csv.DictReader(path.open()))
    splits[split] = rr
    audit['sources'][str(path)] = sha(path)
audit['data'] = {
    'split_counts': {s: len(rr) for s, rr in splits.items()},
    'signer_counts': {s: len({r['Participant ID'] for r in rr}) for s, rr in splits.items()},
    'train_classes': len({r['Gloss'] for r in splits['train']}),
    'intersections': {a + '_' + b: {
        'video': len({r['Video file'] for r in splits[a]} & {r['Video file'] for r in splits[b]}),
        'signer': len({r['Participant ID'] for r in splits[a]} & {r['Participant ID'] for r in splits[b]})}
        for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]},
    'cache_provenance': state['cache_provenance'],
}
meta = read(ROOT / 'runtime/cache/aslc_train_full.meta.json')
dur = np.array([v[3] / v[2] for v in meta.values()])
audit['data']['train_duration_hours'] = float(dur.sum() / 3600)
audit['data']['duration_seconds_quantiles'] = dict(zip(['min', 'p25', 'median', 'p75', 'max'],
                                                     np.quantile(dur, [0, .25, .5, .75, 1]).tolist()))
audit['data']['train_class_support_range'] = [min(collections.Counter(r['Gloss'] for r in splits['train']).values()),
                                            max(collections.Counter(r['Gloss'] for r in splits['train']).values())]

import torch
from pj.model import PhonoJEPA
model = PhonoJEPA()
audit['model'] = {'parameter_counts': {n: sum(p.numel() for p in getattr(model, n).parameters())
                                     for n in ['student', 'teacher', 'predictor', 'heads']},
                  'context_head_parameters': sum(p.numel() for p in model.heads.ctx.parameters()),
                  'readout_grp_dimension': 3 * model.student.d}
run = ROOT / 'runtime/runs' / name
ck = torch.load(run / 'final.pt', map_location='cpu', weights_only=False)
latest = torch.load(run / 'ckpt.pt', map_location='cpu', weights_only=False)
audit['checkpoint'] = {
    'step': ck['step'], 'sha256': sha(run / 'final.pt'),
    'latest_step': latest['step'],
    'latest_and_final_model_tensors_equal': all(torch.equal(latest[n][k], v)
        for n in ['student', 'teacher', 'predictor', 'heads'] for k, v in ck[n].items()),
    'latest_and_final_bytes_equal': sha(run / 'ckpt.pt') == sha(run / 'final.pt'),
    'nonfinite_model_values': {n: sum(int((~torch.isfinite(t)).sum()) for t in ck[n].values()
                                     if torch.is_floating_point(t))
                              for n in ['student', 'teacher', 'predictor', 'heads']},
    'retained_pt_files': [p.name for p in sorted(run.glob('*.pt'))],
}
del ck, latest, model

pilot_path = ROOT / 'runtime/eval/PJ_PLctx_FULL_160ep.json'
pilot = read(pilot_path)
audit['pilot_evaluation'] = pilot
audit['sources'][str(pilot_path)] = sha(pilot_path)
old = np.load(ROOT / 'runtime/eval/models/PJ_PLctx_16k_grp.npz')
new = np.load(ROOT / 'runtime/eval/models/PJ_PLctx_FULL_160ep_grp.npz')
signer_diffs = []
for protocol, rk, sk in [('LOSO', 'loso_ranks', 'loso_sg'), ('ASLC300', 'a300_ranks', 'a300_sg')]:
    assert np.array_equal(old[sk], new[sk])
    for signer in np.unique(old[sk]):
        ix = old[sk] == signer
        a, b = old[rk][ix] <= 1, new[rk][ix] <= 1
        signer_diffs.append({'protocol': protocol, 'signer': str(signer), 'n': int(ix.sum()),
                             'old_Rec1_pct': float(a.mean() * 100), 'full_Rec1_pct': float(b.mean() * 100),
                             'diff_pp': float((b.mean() - a.mean()) * 100),
                             'gained_queries': int((~a & b).sum()), 'lost_queries': int((a & ~b).sum())})
audit['paired_signer_differences'] = signer_diffs

dictionary = ROOT / 'runtime/eval/PJ_PLctx_FULL_160ep_dictionary'
audit['dictionary'] = {}
test_signers = []
for ds in ['aslc', 'wlasl']:
    p = dictionary / ds
    rows = jsonl(p / 'test.jsonl')
    ranks = np.load(p / '_grp_ranks.npy')
    result = read(p / 'results.json')
    protocol = read(p / 'protocol.json')
    qlabel = np.array([r['label'] for r in rows])
    measured = {'DCG': float(np.mean(1 / np.log2(1 + ranks))),
                'Rec1': float(np.mean(ranks <= 1)), 'Rec5': float(np.mean(ranks <= 5))}
    for k, v in measured.items():
        assert np.isclose(v, result['readouts']['_grp'][k], atol=1e-12, rtol=0)
    assert protocol['checkpoint_sha256'] == audit['checkpoint']['sha256']
    item = {'results': result, 'protocol': protocol, 'verification': read(p / 'verification.json'),
            'metrics_recomputed_match': True, 'checkpoint_hash_match': True,
            'rank_quantiles': dict(zip(['min', 'p25', 'median', 'p75', 'p90', 'p95', 'max'],
                                      np.quantile(ranks, [0, .25, .5, .75, .9, .95, 1]).tolist())),
            'correct_top1': int((ranks <= 1).sum()), 'correct_top5': int((ranks <= 5).sum()),
            'class_macro_Rec1': float(np.mean([(ranks[qlabel == c] <= 1).mean() for c in np.unique(qlabel)])),
            'class_macro_Rec5': float(np.mean([(ranks[qlabel == c] <= 5).mean() for c in np.unique(qlabel)]))}
    audit['dictionary'][ds] = item
    if ds == 'aslc':
        sg = np.array([r['signer'] for r in rows])
        for signer in np.unique(sg):
            rr = ranks[sg == signer]
            test_signers.append({'signer': str(signer), 'n': len(rr),
                                 'DCG': float(np.mean(1 / np.log2(1 + rr))),
                                 'Rec1_pct': float(np.mean(rr <= 1) * 100),
                                 'Rec5_pct': float(np.mean(rr <= 5) * 100)})
    for path in [p / 'results.json', p / '_grp_ranks.npy', p / 'protocol.json']:
        audit['sources'][str(path)] = sha(path)
audit['aslc_test_signers'] = test_signers

csvout('phases.csv', all_phases)
csvout('epochs.csv', all_epochs)
csvout('paired_signers.csv', signer_diffs)
csvout('aslc_test_signers.csv', test_signers)
(OUT / 'audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
full_rows = [r for r in all_epochs if r['run'] == name]
for ax, (key, title) in zip(axs.flat, [
    ('loss', 'Training loss (near + far)'), ('lr', 'Learning rate'),
    ('gnorm', 'Gradient norm BEFORE clipping'), ('tok_rankme', 'Token effective rank (fixed batch)'),
    ('cls_mean_cos', 'CLS mean cosine (fixed batch)'), ('teacher_entropy', 'Teacher entropy (far view)')]):
    rr = [r for r in full_rows if r[key + '_mean'] is not None]
    ax.plot([r['epoch'] for r in rr], [r[key + '_mean'] for r in rr], lw=1.6)
    if key == 'gnorm':
        ax.axhline(1, color='red', ls='--', lw=1, label='clip threshold = 1')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
    ax.axvline(16, color='grey', ls=':', lw=1)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.grid(alpha=.2)
fig.suptitle('PLctx: full ASL Citizen train, 160 epochs | means over logged samples per epoch\n'
             'Dotted line: warmup end; diagnostics are proxies, not validation accuracy', fontsize=12)
fig.savefig(OUT / 'training_curves.png', dpi=180)
fig.savefig(OUT / 'training_curves.svg')
plt.close(fig)
print(json.dumps({'artifacts': str(OUT), 'training_checks': audit['training_checks'],
                  'checkpoint': audit['checkpoint'], 'data': audit['data'],
                  'last_diagnostics': audit['runs'][name]['last_diagnostics']}, indent=2))
