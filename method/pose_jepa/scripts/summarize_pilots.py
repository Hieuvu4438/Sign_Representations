"""Summarize completed pilot artifacts; no metrics are inferred from loss alone."""
import argparse
import json
from pathlib import Path
import statistics
import yaml
import sys
sys.path.insert(0, str(Path(__file__).parents[1]/'src'))
from signrepr.data.schema import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    terminals = json.loads((args.runtime/'pilot_controls/results.json').read_text())
    if len(terminals) != 5 or any(r['exit_code'] != 0 for r in terminals):
        raise ValueError('All five child runs must terminate successfully before this report')
    records = []
    for terminal in terminals:
        name = terminal['run']; root = args.runtime/f'pilot5k_{name}'
        rows = [json.loads(line) for line in (root/'train.jsonl').read_text().splitlines()]
        successful = [row for row in rows if row['successful']]
        if len(successful) != 5000 or successful[-1]['step'] != 5000:
            raise ValueError(f'Incomplete successful optimizer budget: {name}')
        validation = [(path, json.loads(path.read_text())) for path in sorted(root.glob('validation_*.json'))]
        best_path, best = max(validation, key=lambda pair: pair[1]['metrics']['R@1'])
        losses = {key: statistics.mean(row['raw_losses'][key] for row in successful[-100:]) for key in successful[-1]['raw_losses']}
        config = yaml.safe_load((root/'config.yaml').read_text())
        if config['loss']['latent_weight'] == 0: losses['latent'] = None
        records.append({'run': name, 'successful_updates': len(successful), 'skipped_updates': len(rows)-len(successful),
                        'last_100_raw_losses': losses, 'best_validation_step': int(best_path.stem.split('_')[-1]),
                        'active_terms': [term for term in ['latent', 'geometry', 'motion'] if config['loss'][term+'_weight'] > 0],
                        'teacher_policy': 'cosine EMA' if config['loss']['latent_weight'] else 'online encoder copy, momentum 0, no teacher forward',
                        'validation': best['metrics'], 'representation': best['representation_diagnostics'],
                        'wall_clock_sec': terminal['wall_clock_sec'], 'GPU_hours_single_device_elapsed': terminal['wall_clock_sec']/3600,
                        'peak_model_allocated_vram_bytes': max(row['peak_allocated_vram_bytes'] for row in successful),
                        'checkpoint_sha256': sha256_file(root/'best_validation.pt'),
                        'config_sha256': sha256_file(root/'config.yaml'), 'provenance': json.loads((root/'provenance.json').read_text())})
    eligible = [r for r in records if r['run'] != 'jepa_shuffled']
    selected = sorted(eligible, key=lambda r: (-r['validation']['R@1'], r['run']))[0]
    report = {'scope': '4-class official-split ASL-Citizen legacy-pose sanity subset; not full benchmark',
              'training_protocol': 'in_domain_ssl', 'selection_scope': 'target_validation',
              'effective_batch': 2, 'architecture': 'guide full 8-layer width-256 encoder / 4-layer width-128 predictor',
              'comparison': 'fixed successful optimizer updates, not compute-matched or isolated-wall-clock evidence',
              'selection_rule': 'validation R@1; deterministic lexical run-ID tie; shuffled diagnostic excluded',
              'selected_run': selected['run'], 'runs': records,
              'limitations': ['16 train / 8 validation clips and 4 classes cannot establish benchmark superiority.',
                              'Uncalibrated legacy score mapping and detector extrapolation are explicitly audited.',
                              'Shared hardware has other workloads; elapsed GPU-hours are not isolated compute cost.',
                              'Early reconstruction-control logs contain an unused scheduled EMA value and dummy latent CE; these fields are excluded from interpretation and corrected in final code.',
                              'Minimal-pair/non-manual annotations are unavailable; no physical-loss superiority claim.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({'selected_run': selected['run'], 'runs': len(records)}))


if __name__ == '__main__': main()
