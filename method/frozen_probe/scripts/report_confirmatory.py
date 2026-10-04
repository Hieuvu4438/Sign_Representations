"""Render existing registered results and every seeded case; no new scoring."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import file_sha
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        inputs = {}
        for key, specification in config['inputs'].items():
            directory = REPO / specification['run']
            status = json.loads((directory / 'status.json').read_text())
            if status['status'] != 'completed' or file_sha(directory / 'status.json') != specification['status_sha256']:
                raise ValueError('Changed or incomplete report input')
            name = specification['file']
            if file_sha(directory / name) != status['output_hashes'][name]:
                raise ValueError('Changed report source output')
            inputs[key] = json.loads((directory / name).read_text())
        analysis, schema = inputs['analysis'], inputs['schema']
        methods = list(analysis['mean_metrics'])
        glosses = dict(zip(schema['class_ids'], schema['class_glosses']))
        signers = dict(zip(schema['query_ids'], schema['participant_ids']))
        if analysis['N_expected'] != analysis['N_scored'] or len(signers) != analysis['N_scored']:
            raise ValueError('Incomplete report denominator')
        lines = ['# Registered held-out diagnostics and seeded cases', '',
                 'Presentation of CONFAN002 outputs only. No new metric, case selection or test-based policy change.', '',
                 'N=1217 queries; 100 fixed classes; 11 participant IDs; three fixed galleries, eight examples/class.', '',
                 'Participant and slice differences are descriptive DCG fractions, averaged across the fixed galleries.', '',
                 '| Participant | N | Uniform minus dense DCG |', '| --- | ---: | ---: |']
        for signer, row in analysis['participant_differences'].items():
            lines.append(f"| {signer} | {row['N']} | {row['uniform_minus_dense_dcg']:.9f} |")
        for dimension, slices in analysis['descriptive_slices'].items():
            if sum(row['N'] for row in slices.values()) != analysis['N_scored']:
                raise ValueError('Slice denominators do not cover the cohort')
            lines += ['', f'## {dimension}', '',
                      '| Slice | N | Participants | Uniform minus dense DCG |',
                      '| --- | ---: | ---: | ---: |']
            for key, row in slices.items():
                lines.append(f"| {key} | {row['N']} | {row['participant_count']} | {row['uniform_minus_dense_dcg']:.9f} |")
        lines += ['', '## All predeclared cases', '',
                  'Gallery seed1701; original SHA256 ordering, first five/category. Exact dataset gloss strings below are class labels.',
                  'No video/content review or linguistic explanation has been performed. Top10 rankings may omit the correct label.', '']
        case_rows = []
        for category, group in analysis['qualitative_cases'].items():
            if len(group['samples']) != min(5, group['eligible_N']):
                raise ValueError('Unexpected registered case count')
            lines += [f"## {category} (eligible N={group['eligible_N']})", '']
            for case in group['samples']:
                sample = case['sample_id']
                d = case['diagnostics']
                lines += [f"**{sample}** — label `{glosses[case['label']]}`, participant {signers[sample]}; "
                          f"{d['frames']} frames/{d['source_windows']} dense windows; "
                          f"effective activity fraction {d['effective_activity_fraction']:.6f}.", '',
                          '| Method | Top10 exact gloss labels (rank order) |', '| --- | --- |']
                for method in methods:
                    ranked = case['rankings'][method]
                    labels = [glosses[class_id] for class_id in ranked]
                    lines.append(f"| {method} | {', '.join(labels)} |")
                    for rank, (class_id, label) in enumerate(zip(ranked, labels), 1):
                        case_rows.append(dict(category=category, eligible_N=group['eligible_N'],
                            sample_id=sample, participant=signers[sample], true_class_id=case['label'],
                            true_gloss=glosses[case['label']], method=method, rank=rank,
                            predicted_class_id=class_id, predicted_gloss=label))
                lines.append('')
        if len(case_rows) != 20 * len(methods) * 10:
            raise ValueError('Case export incomplete')
        (run.output / 'diagnostics.md').write_text('\n'.join(lines) + '\n')
        with (run.output / 'cases.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]))
            writer.writeheader()
            writer.writerows(case_rows)

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
        labels = ['Dense activity', 'Uniform stride 8', 'Random 2718', 'Random 31415']
        timed_methods = list(analysis['costs'])
        colors = ['#555555', '#2166ac', '#d95f02', '#7570b3']
        plot_rows = []
        for method, label, color in zip(timed_methods, labels, colors):
            cost = analysis['costs'][method]
            dcg = analysis['mean_metrics'][method]['dcg']
            axes[0].scatter(cost['p50_seconds'], dcg, c=color, label=label, s=55)
            axes[1].plot([cost['p50_seconds'], cost['p95_seconds']], [label, label], color=color, lw=2)
            axes[1].scatter(cost['p50_seconds'], label, color=color, s=45)
            axes[1].scatter(cost['p95_seconds'], label, color=color, marker='|', s=100)
            plot_rows.append(dict(method=method, mean_dcg=dcg, **cost))
        dense = analysis['mean_metrics']['full_activity']['dcg']
        axes[0].axhline(dense-.02, color='#999999', linestyle='--', label='Dense activity DCG minus .02')
        axes[0].set_xlabel('Median query pipeline time (s)')
        axes[0].set_ylabel('Mean DCG across three fixed galleries')
        axes[0].set_title('Registered quality and observed cost')
        axes[0].legend(fontsize=8, loc='lower right')
        axes[1].set_xlabel('Query pipeline time (s): dot p50, tick p95')
        axes[1].set_title('Timing quantiles; segments are not CIs')
        axes[1].invert_yaxis()
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle('Held-out lookup: 1217 queries, 100 fixed ASL classes', fontsize=12)
        for suffix in ('svg', 'pdf', 'png'):
            fig.savefig(run.output / f'quality_cost.{suffix}', dpi=180, metadata={'Creator': 'frozen_probe/P003'})
        plt.close(fig)
        with (run.output / 'figure_data.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(plot_rows[0]))
            writer.writeheader()
            writer.writerows(plot_rows)
        run.save('report_metadata.json', dict(inputs=config['inputs'], case_rows=len(case_rows),
            matplotlib=matplotlib.__version__, scope='presentation only; original selected cases and all controls retained',
            caption='Quality is averaged over three fixed galleries. Timing: one independent full decode/preprocess/encoder/pool/three-gallery-search observation per query and recipe, shared-eligible GPU and uncontrolled OS cache. Model/gallery setup and integrity/persistence excluded. Full_mean quality shares the dense pass and has no separate timing. Plot margin line alone is not the joint quality/CI/cost screen.'))
        names = ('diagnostics.md', 'cases.csv', 'quality_cost.svg', 'quality_cost.pdf',
                 'quality_cost.png', 'figure_data.csv', 'report_metadata.json')
        run.status['output_hashes'] = {name: file_sha(run.output / name) for name in names}


if __name__ == '__main__':
    main()
