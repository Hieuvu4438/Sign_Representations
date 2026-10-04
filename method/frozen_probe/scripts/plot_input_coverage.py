"""Export native coverage ECDFs from completed structural rows; no score access."""
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
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        source_run = REPO / config['source_run']
        status_path = source_run/'status.json'
        status = json.loads(status_path.read_text())
        if status['status'] != 'completed' or file_sha(status_path) != config['source_status_sha256']:
            raise ValueError('Coverage source run changed/incomplete')
        for name, digest in status['output_hashes'].items():
            if file_sha(source_run/name) != digest:
                raise ValueError('Structural source output changed')
        source = json.loads((source_run/'coverage.json').read_text())
        rows = source['sources']
        policies = [('uniform8', 'Uniform stride 8', '#2166ac'),
                    ('random8_s2718', 'Random, seed 2718', '#d95f02'),
                    ('random8_s31415', 'Random, seed 31415', '#7570b3')]
        fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4), sharey=True, constrained_layout=True)
        figure_data = []
        for ax, split in zip(axes, ('train', 'val', 'test')):
            count = None
            for method, label, color in policies:
                selected = [r for r in rows if r['split'] == split and r['method'] == method]
                values = np.sort([r['frame']['fraction'] for r in selected])
                if not len(values) or not np.isfinite(values).all():
                    raise ValueError('Missing/nonfinite coverage curve')
                count = len(values)
                ax.step(np.r_[0, values, 1], np.r_[0, np.arange(1, count+1)/count, 1],
                        where='post', label=label, color=color, linewidth=1.8)
                figure_data.extend(dict(split=split, policy=method, sample_id=r['sample_id'],
                    frame_coverage_fraction=r['frame']['fraction'], time_coverage_fraction=r['time']['fraction'],
                    longest_uncovered_gap_frames=r['frame']['longest_uncovered_gap']) for r in selected)
            ax.set_title(f"{'Dev' if split == 'val' else split.title()} (N={count})")
            ax.set_xlabel('Fraction of dense-covered native frames exposed')
            # Keep the atom at complete coverage visible instead of on the spine.
            ax.set_xlim(-.02, 1.02)
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set_ylim(0, 1)
            ax.grid(alpha=.2)
        axes[0].set_ylabel('Fraction of clips (empirical CDF)')
        axes[-1].legend(loc='upper left', fontsize=8.5, frameon=False)
        fig.suptitle('Input coverage under matched sparse window counts', fontsize=13)
        for suffix in ('svg', 'pdf', 'png'):
            fig.savefig(run.output/f'coverage_ecdf.{suffix}', dpi=180, metadata={'Creator': 'frozen_probe/A001'})
        plt.close(fig)
        with (run.output/'figure_data.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(figure_data[0]))
            writer.writeheader()
            writer.writerows(figure_data)
        run.save('figure_metadata.json', dict(source_run=status['run_id'], source_status_sha256=file_sha(status_path),
            matplotlib=matplotlib.__version__, numpy=np.__version__, rows=len(figure_data),
            unit='per-clip frame exposure fraction over dense complete native-window domain',
            scope='auxiliary post-metadata/pre-quality analysis, source-only; no inference about retrieval quality or causality',
            caption='Uniform8 and both fixed same-count random8 policies, all fixed sources in each split. Dense coverage is1 by definition. Incomplete tail outside dense domain excluded. A coverage curve is not an accuracy or linguistic-information curve.'))
        run.status['output_hashes'] = {name:file_sha(run.output/name) for name in
            ('coverage_ecdf.svg', 'coverage_ecdf.pdf', 'coverage_ecdf.png', 'figure_data.csv', 'figure_metadata.json')}


if __name__ == '__main__':
    main()
