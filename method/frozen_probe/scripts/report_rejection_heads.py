"""Bounded descriptive figure and numerical disagreement cases after audit."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from bounded_artifacts import verify_store, inspect_file_plan
sys.path.insert(0, str(Path(__file__).resolve().parent))
from register_axis_stage import dependencies, put, sha


def read(path): return json.loads(Path(path).read_text())


def register():
    root = Path('method/frozen_probe'); records = Path('method/research_lab/records')
    entry = Path(__file__).relative_to(Path.cwd())
    sources = set(dependencies(entry)) | {root / 'scripts/register_axis_stage.py'}
    snapshots = {'source_snapshot/' + str(p.relative_to(root)): str(p) for p in sorted(sources)}
    inputs = {p: sha(p) for p in snapshots.values()}; parents = {}
    for key in ('HEADANALYSIS001', 'REJECTEVAL001', 'HEADPILOT001'):
        directory = Path(read(records / f'experiments/{key}_registration.json')['output_dir'])
        assert read(directory / 'status.json')['status'] == 'completed'; parents[key] = directory
        inputs.update(read(directory / 'config.json')['input_hashes'])
        for p in directory.rglob('*'):
            if p.is_file(): inputs[str(p)] = sha(p)
    assert read(parents['HEADANALYSIS001'] / 'head_analysis.json')['verification_passed']
    protocol = records / 'protocols/HEADREPORT001_descriptive_export.md'; inputs[str(protocol)] = sha(protocol)
    caps = {n: Path(p).stat().st_size for n, p in snapshots.items()}
    caps.update({'config.json': 1 << 20, 'environment.json': 65536, 'status.json': 131072,
        'error.txt': 65536, 'combined_log.txt': 65536, 'joint_head_curves.png': 1 << 20,
        'joint_head_curves.pdf': 1 << 20, 'numerical_disagreements.json': 65536, 'report_summary.json': 65536})
    c = dict(protocol='HEADREPORT001_descriptive_export', H_id='H007', seed=1701,
        timeout_seconds=60, GPU_allocation_seconds=0, HPO_trials=0, source_snapshots=snapshots,
        input_hashes=inputs, artifact_file_caps=caps, artifact_manifest_cap=131072, artifact_ceiling_bytes=4 << 20,
        analysis_runtime=str(parents['HEADANALYSIS001']), measured_runtime=str(parents['REJECTEVAL001']),
        case_head_runtime=str(parents['HEADPILOT001']))
    plan = inspect_file_plan({**caps, 'artifact_manifest.json': c['artifact_manifest_cap']}, c['artifact_ceiling_bytes'], ('status.json', 'error.txt'))
    assert plan['passed']
    name = 'HEADREPORT001'; path = root / f'configs/{name}.json'; put(path, c); digest = sha(path)
    rid = f'{name}_{datetime.now(timezone.utc):%Y%m%d}_{digest[:8]}_s1701'; output = root / 'runtime' / rid
    command = ['/home/haipd/miniconda3/bin/python', str(entry), '--config', str(path), '--run-id', rid, '--output-dir', str(output), '--seed', '1701']
    reg = dict(registered_utc=datetime.now(timezone.utc).isoformat(), run_id=rid, config=str(path), config_sha256=digest,
        output_dir=str(output), CPU_limit_seconds=60, GPU_seconds=0, durable_bytes_ceiling=c['artifact_ceiling_bytes'],
        maximum_planned_logical_bytes=plan['maximum_logical_bytes'], input_hashes=inputs, command=command)
    put(records / f'experiments/{name}_registration.json', reg)
    with (records / f'experiments/{name}.md').open('x') as f: f.write(f'# {name} — descriptive export\n\n{rid}; outcome pending.\n')
    print(json.dumps({k: v for k, v in reg.items() if k != 'input_hashes'}, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--register', action='store_true')
    for key in ('config', 'run-id', 'output-dir'): parser.add_argument('--' + key)
    parser.add_argument('--seed', type=int); a = parser.parse_args()
    if a.register: register(); return
    with BoundedCPUContext(a.output_dir, a.run_id, a.config, a.seed) as run:
        c = run.config; analysis = Path(c['analysis_runtime']); measured = Path(c['measured_runtime']); head = Path(c['case_head_runtime'])
        for root in (analysis, measured, head): verify_store(root)
        proof = read(analysis / 'head_analysis.json'); assert proof['verification_passed']
        meta = read(measured / 'rejection_metadata.json'); old = read(measured / 'rejection_eval.json')
        means = list(csv.DictReader((analysis / 'mean.csv').open()))
        curves = np.load(analysis / 'head_count_curves.npy', allow_pickle=False)
        old_curves = np.load(measured / 'count_curves.npy', allow_pickle=False)
        per_seed = list(csv.DictReader((analysis / 'per_seed.csv').open()))
        import matplotlib
        matplotlib.use('Agg'); import matplotlib.pyplot as plt
        styles = [('maxcosine', '#1f77b4'), ('margin', '#7f7f7f'), ('maxsoftmax', '#2ca02c'),
                  ('NAC16', '#9467bd'), ('residual_linear', '#ff7f0e'), ('FRDisc_frozen_MLP', '#d62728')]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True); count = 0
        for ax, name, title in zip(axes, ['base_dev', 'MS_dev'], ['ASL Citizen source dev', 'MS-ASL validation']):
            for method, color in styles:
                new = method in ('residual_linear', 'FRDisc_frozen_MLP')
                selected = [s for s in (proof['curve_segments'] if new else meta['curve_segments'])
                            if s['part'] == name and s['size'] == 50 and s['method'] == method]
                assert len(selected) == (9 if new else 3)
                for s in selected:
                    values = (curves if new else old_curves)[s['start']:s['stop']]
                    nk = int(next(r['n_known'] for r in per_seed if r['part'] == name and r['size'] == '50' and r['method'] == method and r['support_seed'] == str(s.get('support_seed', s.get('seed')))))
                    nu = int(next(r['n_unknown'] for r in per_seed if r['part'] == name and r['size'] == '50' and r['method'] == method and r['support_seed'] == str(s.get('support_seed', s.get('seed')))))
                    ax.plot(values[:, 3] / nu, values[:, 2] / nk, color=color, alpha=.22, lw=.7); count += 1
                op = next(r for r in means if (r['part'], r['size'], r['method'], r['policy']) == (name, '50', method, 'FROZEN50'))
                ax.scatter(float(op['FAR']), float(op['CCR']), color=color, s=45, edgecolor='white', linewidth=.5, label=method, zorder=5)
            ax.set(title=title, xlabel='Unknown false acceptance rate (FAR)', xlim=(0, 1), ylim=(0, 1)); ax.grid(alpha=.2)
        axes[0].set_ylabel('Correct known-class acceptance rate (CCR)')
        axes[1].legend(frameon=False, fontsize=8, loc='upper left')
        fig.suptitle('Frozen SignRep, 50-class one-shot dictionaries: fixed scores and supervised heads', fontsize=11)
        fig.text(.5, .015, 'Thin curves: individual fixed cells. Dots: equal cell means at source-known 95% acceptance calibration.\nHeads: 3 initializations × 3 galleries; fixed scores: 3 galleries. Extra source labels used by heads; no uncertainty bands.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .09, 1, .94))
        for extension in ('png', 'pdf'):
            buffer = io.BytesIO(); fig.savefig(buffer, format=extension, dpi=180)
            run.store.write_bytes('joint_head_curves.' + extension, buffer.getvalue())
        plt.close(fig)
        assert count == 60
        logits = np.load(head / 'head_scores.npy', allow_pickle=False); scores = np.load(measured / 'scores.npy', allow_pickle=False)
        predictions = np.load(measured / 'predictions.npy', allow_pickle=False); ranks = np.load(measured / 'known_rank.npy', allow_pickle=False)
        head_report = read(head / 'head_evaluation.json'); rng = np.random.default_rng(1701); cases = []
        for name in ('base_dev', 'MS_dev'):
            part = next(p for p in meta['parts'] if p['name'] == name); sl = slice(part['start'], part['stop'])
            known = ranks[0, 2, sl] > 0; correct = predictions[0, 2, sl] == np.asarray(meta['labels'][sl])
            rt = old['calibration']['1701:50:maxcosine']['threshold']; ht = head_report['calibration']['1701:50:FRDisc_frozen_MLP']['threshold']
            ra = scores[0, 2, 0, sl] >= rt; ha = logits[0, 2, 1, sl] >= ht
            buckets = {'known_correct_recovered': known & correct & ~ra & ha,
                'known_correct_lost': known & correct & ra & ~ha,
                'unknown_additionally_accepted': ~known & ~ra & ha,
                'unknown_newly_rejected': ~known & ra & ~ha,
                'known_wrong_accepted_by_head': known & ~correct & ha}
            for bucket, mask in buckets.items():
                available = np.flatnonzero(mask) + part['start']
                selected = sorted(rng.choice(available, min(2, len(available)), replace=False).tolist())
                samples = [dict(sample_id=meta['query_ids'][i], true_class=meta['labels'][i], official_signer=meta['signers'][i],
                    predicted_class=int(predictions[0, 2, i]), known_rank=int(ranks[0, 2, i]),
                    raw_score=float(scores[0, 2, 0, i]), MLP_logit=float(logits[0, 2, 1, i]),
                    raw_accepted=bool(scores[0, 2, 0, i] >= rt), MLP_accepted=bool(logits[0, 2, 1, i] >= ht)) for i in selected]
                cases.append(dict(part=name, bucket=bucket, eligible_count=len(available), samples=samples))
        run.save('numerical_disagreements.json', dict(posthoc=True, initialization=1701, support_seed=1701,
            size=50, policy='FROZEN50', selection='fixed part/bucket order, PCG64 seed1701, uniform up to2 per bucket', cases=cases,
            scope='Numerical metadata only, descriptive subset; no linguistic/visual/human interpretation or target-based method selection.'))
        run.save('report_summary.json', dict(passed=True, scientific_screen_passed=proof['scientific_screen_passed'],
            primary=proof['primary'], figure_curves=60, mean_operating_points=12, numerical_case_buckets=10,
            matplotlib_version=matplotlib.__version__, GPU_seconds=0, new_encoder_calls=0,
            scope='Derived exploratory plot; no averaged curves, CI bands, new hypothesis, tuning or method promotion.'))


if __name__ == '__main__': main()
