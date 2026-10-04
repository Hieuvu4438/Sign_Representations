"""Auxiliary structural diagnostic registered before any held-out quality score."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import file_sha
from input_coverage import native_coverage
from run_context import RunContext
from temporal import window_plan
from window_budget import selected_indices


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'output-dir'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import numpy as np
        rows = []
        for path, digest in config['video_audits'].items():
            path = REPO / path
            if file_sha(path) != digest:
                raise ValueError('Locked native audit changed')
            audit = json.loads(path.read_text())
            if not audit['passed']:
                raise ValueError('Incomplete native source audit')
            for source in audit['samples']:
                plan = window_plan(source['pts_s'], 16, 2)
                if len(plan) != source['windows']:
                    raise ValueError('Source window count mismatch')
                for method in config['methods']:
                    indices = selected_indices(method, len(plan), source['source_sha256'])
                    rows.append(dict(sample_id=source['sample_id'], split=source['split'], method=method,
                        selected_indices=indices, **native_coverage(plan, indices)))
        summary = {}
        for split in ('train', 'val', 'test'):
            summary[split] = {}
            for method in config['methods']:
                selected = [r for r in rows if r['split'] == split and r['method'] == method]
                if not selected:
                    raise ValueError('Missing fixed split/method coverage')
                summary[split][method] = dict(N=len(selected), encoded_windows=sum(r['encoded_windows'] for r in selected),
                    dense_windows=sum(r['dense_windows'] for r in selected),
                    mean_frame_coverage_fraction=float(np.mean([r['frame']['fraction'] for r in selected])),
                    mean_time_coverage_fraction=float(np.mean([r['time']['fraction'] for r in selected])),
                    complete_dense_frame_coverage_count=sum(r['frame']['fraction'] == 1 for r in selected),
                    max_uncovered_gap_frames=max(r['frame']['longest_uncovered_gap'] for r in selected),
                    median_longest_gap_frames=float(np.median([r['frame']['longest_uncovered_gap'] for r in selected])))
        run.save('coverage.json', dict(summaries=summary, sources=rows,
            registered_scope='auxiliary post-metadata/pre-quality structural diagnostic; not part of original primary confirmatory screen',
            denominators='dense complete native-window domain; incomplete source tail outside denominator; last-frame time end estimated',
            labels_features_or_scores_accessed=False, causal_claim=False,
            limitation='frame/time exposure proxies only; uniform and random positions/activity differ; no linguistic or causal guarantee'))
        lines = ['| Split | Policy | N | Mean frame coverage | Mean time coverage | Median longest gap (frames) |',
                 '| --- | --- | ---: | ---: | ---: | ---: |']
        for split, methods in summary.items():
            for method, value in methods.items():
                lines.append(f"| {split} | {method} | {value['N']} | {value['mean_frame_coverage_fraction']:.6f} | {value['mean_time_coverage_fraction']:.6f} | {value['median_longest_gap_frames']:.1f} |")
        lines += ['', 'Auxiliary after metadata, before outcome feedback; structural exposure only. No quality, mechanism-causality or primary-screen inference.',
                  'Coverage denominator is the dense complete-window domain, excluding dropped incomplete source tail.']
        (run.output/'table.md').write_text('\n'.join(lines)+'\n')
        run.status['output_hashes'] = {name:file_sha(run.output/name) for name in ('coverage.json', 'table.md')}


if __name__ == '__main__':
    main()
