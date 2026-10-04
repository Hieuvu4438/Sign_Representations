"""Parity against the pinned upstream function, without importing its GPU pipeline."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metrics import evaluate, ranked_classes, score_ranking

COMMIT = '17f0148b00ef87a5957ec103da8d10bd8eb1fa23'
URL = f'https://raw.githubusercontent.com/microsoft/ASL-citizen-code/{COMMIT}/I3D/aslcitizen_testing.py'
NAMES = ['dcg', 'top1', 'top5', 'top10', 'top20', 'mrr']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {'run_id': args.run_id, 'status': 'running', 'start_utc': datetime.now(timezone.utc).isoformat()}
    def save(name, data):
        temp = args.output_dir/(name+'.tmp')
        temp.write_text(json.dumps(data, indent=2)+'\n')
        temp.replace(args.output_dir/name)
    save('status.json', status)
    try:
        with urllib.request.urlopen(URL, timeout=30) as response:
            source = response.read(1024 * 1024 + 1)
        if len(source) > 1024 * 1024:
            raise ValueError('Reference source exceeds 1 MiB limit')
        (args.output_dir/'aslcitizen_testing.py.reference').write_bytes(source)
        # Execute only the audited function. No top-level dataset loading,
        # checkpoint loading, CUDA calls or upstream output writes.
        functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'eval_metrics']
        if len(functions) != 1:
            raise ValueError('Upstream function missing or ambiguous')
        namespace = {'np': np, 'math': math}
        exec(compile(ast.Module(body=functions, type_ignores=[]), URL, 'exec'), namespace)
        reference = namespace['eval_metrics']
        fixtures = []
        vocab = list(range(25))
        for rank in [1, 3, 5, 6, 10, 11, 20, 21, 25]:
            label = rank - 1
            _, counts = reference(np.array(vocab), label)
            upstream = dict(zip(NAMES, (float(v) for v in counts)))
            wrapped = score_ranking(vocab, label, vocab)
            # Separate rank-based oracle, with rank-3 hand calculation below.
            oracle = dict(zip(NAMES, [1/math.log2(rank+1), float(rank <= 1),
                        float(rank <= 5), float(rank <= 10), float(rank <= 20), 1/rank]))
            if rank == 3:
                oracle.update(dcg=.5, mrr=1/3)
            passed = all(abs(upstream[k]-oracle[k]) <= 1e-12 and
                         abs(wrapped[k]-oracle[k]) <= 1e-12 for k in NAMES)
            fixtures.append({'id': f'rank_{rank}', 'oracle': oracle,
                             'upstream': upstream, 'wrapper': wrapped, 'passed': passed})
        tied = ranked_classes([1, 1, 2], [0, 1, 2])
        _, counts = reference(np.array(tied), 1)
        fixtures.append({'id': 'tie_local_rule', 'ranking': tied,
                         'wrapper': score_ranking(tied, 1, [0, 1, 2]),
                         'upstream': dict(zip(NAMES, map(float, counts))),
                         'passed': tied == [2, 0, 1] and float(counts[5]) == 1/3})
        invalid = []
        refs = [{'sample_id': 'a', 'label': 0}]
        prediction = {'sample_id': 'a', 'ranking': [0, 1]}
        for name, preds in [('empty', []), ('duplicate', [prediction, prediction]),
                            ('missing_and_extra', [{'sample_id': 'b', 'ranking': [0, 1]}])]:
            try:
                evaluate(preds, refs, [0, 1])
            except ValueError as e:
                invalid.append({'id': name, 'passed': True, 'reason': str(e),
                                'upstream': 'no coverage validation at function boundary'})
            else:
                invalid.append({'id': name, 'passed': False})
        report = {'run_id': args.run_id, 'kind': 'synthetic_scorer_parity',
                  'source_url': URL, 'source_commit': COMMIT, 'source_sha256': digest(source),
                  'wrapper_sha256': digest((ROOT/'metrics.py').read_bytes()),
                  'script_sha256': digest(Path(__file__).read_bytes()),
                  'python': platform.python_version(), 'numpy': np.__version__,
                  'tolerance': 1e-12, 'fixtures': fixtures, 'invalid_inputs': invalid,
                  'passed': all(x['passed'] for x in fixtures+invalid),
                  'scope': 'eval_metrics class-ranking only; no model or benchmark parity',
                  'caveats': ['tie ordering is local ascending class ID, not a verified torch tie rule',
                              'complete vocabulary required; all-wrong rank still has nonzero DCG/MRR',
                              'no upstream dev predictions/checkpoint replay',
                              'no gallery/instance-ranking protocol validated']}
        save('parity.json', report)
        if not report['passed']:
            raise AssertionError('Parity failed')
        status.update(status='completed', exit_code=0, GPU_hours=0,
                      wall_seconds=time.monotonic()-started, end_utc=datetime.now(timezone.utc).isoformat(),
                      output_sha256=digest((args.output_dir/'parity.json').read_bytes()),
                      download_bytes=len(source))
        save('status.json', status)
        print(json.dumps(status))
    except Exception as e:
        status.update(status='failed', exit_code=1, error=repr(e), wall_seconds=time.monotonic()-started)
        save('status.json', status)
        raise


if __name__ == '__main__':
    main()
