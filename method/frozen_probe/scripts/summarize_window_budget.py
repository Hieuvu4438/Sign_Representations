"""Rebuild joined policy eligibility and a table from immutable pilot evidence."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cache import file_sha
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'output-dir', 'quality-run', 'timing-run'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    for directory in (args.quality_run, args.timing_run):
        status = json.loads((directory / 'status.json').read_text())
        if status['status'] != 'completed' or status['protocol'] != 'P002_v1':
            raise ValueError('Missing complete pilot evidence')
        expected = status.get('output_hashes', {'timing.json': status.get('output_sha256')})
        for name, digest in expected.items():
            if file_sha(directory / name) != digest:
                raise ValueError('Pilot output changed')
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import numpy as np
        from window_budget import POLICIES
        q = json.loads((args.quality_run / 'quality.json').read_text())
        t = json.loads((args.timing_run / 'timing.json').read_text())
        timing_config = json.loads((args.timing_run / 'config.json').read_text())
        samples = timing_config['cost_samples']
        rows = t['observations']
        methods = ('full_activity',) + POLICIES
        expected = {(m, k, r) for m in methods for k in samples for r in range(3)}
        observed = {(r['method'], r['sample_id'], r['repeat']) for r in rows}
        if len(rows) != 105 or observed != expected or len(observed) != len(rows):
            raise ValueError('Incomplete/duplicate timing observation coverage')
        for row in rows:
            components = [row[k] for k in ('decode_seconds', 'preprocess_seconds',
                'encoder_wrapper_seconds', 'pool_seconds', 'three_gallery_search_seconds')]
            if (not np.isfinite(components).all() or min(components) < 0
                    or not np.isclose(sum(components), row['end_to_end_seconds'], atol=1e-9, rtol=0)):
                raise ValueError('Timing component accounting differs')
        combined = {}
        for method in methods:
            selected = [r for r in rows if r['method'] == method]
            values = [r['end_to_end_seconds'] for r in selected]
            ratios = [np.median([r['end_to_end_seconds'] for r in selected if r['sample_id'] == k]) /
                      np.median([r['end_to_end_seconds'] for r in rows if r['method'] == 'full_activity' and r['sample_id'] == k])
                      for k in samples]
            counts = sum(r['encoded_windows'] for r in selected)
            full = sum(r['encoded_windows'] for r in rows if r['method'] == 'full_activity')
            median_ratio = float(np.median(ratios))
            summary = t['summaries'][method]
            for key, value in (('p50_seconds', float(np.quantile(values,.5))),
                               ('p95_seconds', float(np.quantile(values,.95))),
                               ('median_paired_ratio', median_ratio),
                               ('sample_window_reduction_factor', full/counts)):
                if not np.isclose(summary[key], value, atol=1e-12, rtol=0):
                    raise ValueError('Saved timing summary differs from raw observations')
            if method == 'full_activity':
                primary = q['primary_dcg'][method]
                loss, windows, factor, quality_ok = 0., q['dense_windows'], 1., False
            else:
                quality = q['policies'][method]
                primary, windows = quality['primary_dcg'], quality['encoded_windows']
                loss, factor = q['primary_dcg']['full_activity'] - primary, q['dense_windows']/windows
                quality_ok = loss <= .02 + 1e-12 and windows*2 <= q['dense_windows']
                if quality_ok != quality['quality_count_screen'] or not np.isclose(loss, quality['dcg_loss'], atol=1e-12):
                    raise ValueError('Saved quality floor disagrees')
            cost_ok = median_ratio <= .8 and counts*2 <= full
            if cost_ok != summary['actual_cost_screen']:
                raise ValueError('Saved actual cost floor disagrees')
            combined[method] = dict(primary_dcg=primary, dcg_loss=loss, cohort_windows=windows,
                cohort_window_reduction=factor, p50_seconds=summary['p50_seconds'],
                p95_seconds=summary['p95_seconds'], median_paired_e2e_ratio=median_ratio,
                quality_count_ok=quality_ok, actual_cost_ok=cost_ok, joint_pilot_eligible=quality_ok and cost_ok)
        viable = [m for m in POLICIES if combined[m]['joint_pilot_eligible']]
        selected = sorted(viable, key=lambda m: (-combined[m]['cohort_window_reduction'], -combined[m]['primary_dcg'], m))[0] if viable else None
        run.save('combined.json', dict(run_id=args.run_id, methods=combined, selected_policy=selected,
            policy_trials=6, quality_source=str(args.quality_run), timing_source=str(args.timing_run),
            quality_status_sha256=file_sha(args.quality_run/'status.json'),
            timing_status_sha256=file_sha(args.timing_run/'status.json'), test_access='none',
            scope='exploratory100-class dev; five query participants; timing five fixed sources×three repeats',
            limitation='model/gallery setup excluded from per-query timing; shared GPU/OS cache; stride8 is already published, not new sampling'))
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=['method'] + list(combined['full_activity']))
        writer.writeheader()
        for method, values in combined.items():
            writer.writerow(dict(method=method, **values))
        (run.output / 'quality_cost.csv').write_text(stream.getvalue())
        text = '# H002 first pilot — adapted dev cost/quality\n\nDCG fractions; windows over1691 sources. Latency covers5 fixed sources×3 repeats;\nmodel/gallery setup separate, shared GPU and uncontrolled OS cache. No test/novel\nmethod/confirmatory noninferiority claim.\n\n| Policy | DCG | DCG loss | Windows | Reduction | p50/p95 s | Paired ratio | Joint floor |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n'
        for method, v in combined.items():
            flag = 'reference' if method == 'full_activity' else ('pass' if v['joint_pilot_eligible'] else 'fail')
            text += f"| {method} | {v['primary_dcg']:.6f} | {v['dcg_loss']:.6f} | {v['cohort_windows']} | {v['cohort_window_reduction']:.3f}× | {v['p50_seconds']:.3f}/{v['p95_seconds']:.3f} | {v['median_paired_e2e_ratio']:.3f} | {flag} |\n"
        text += f'\nSelected by frozen rule: **{selected}**. All support-seed metrics and\nconditional participant intervals remain in QUAL001; all105 raw observations\nremain in TIME001. Successful screen permits further scoped analysis; it is\nnot a paper direction/novelty lock.\n'
        (run.output / 'table.md').write_text(text)
        run.status['output_hashes'] = {name: file_sha(run.output/name) for name in ('combined.json','quality_cost.csv','table.md')}


if __name__ == '__main__':
    main()
