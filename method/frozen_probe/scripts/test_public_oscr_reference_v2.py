"""Execute only the audited, pinned OSCR function on preregistered tiny fixtures."""
import argparse
import ast
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'run-id', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    with BoundedCPUContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        source = Path(run.config['reference_source']).read_text()
        tree = ast.parse(source)
        candidates = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'compute_oscr']
        assert len(candidates) == 1
        function = candidates[0]
        assert (function.lineno, function.end_lineno) == (339, 366)
        assert not function.decorator_list and not function.returns
        assert [arg.arg for arg in function.args.args] == ['targets', 'logits', 'os_score']
        assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(function))
        namespace = {'np': np, '__builtins__': {'sorted': sorted, 'range': range, 'len': len, '__import__': __import__}}
        executable = ast.Module(body=[function], type_ignores=[])
        exec(compile(executable, run.config['reference_source'], 'exec'), namespace)
        reference = namespace['compute_oscr']
        outcomes = []
        for fixture in run.config['fixtures']:
            targets, scores = fixture['targets'], fixture['scores']
            logits = [[1., 0.] for _ in targets]
            observed = float(reference(targets, logits, scores))
            matches_prediction = abs(observed - fixture['predicted_reference_area']) <= run.config['float_atol']
            matches_manual = abs(observed - fixture['manual_joint_area']) <= run.config['float_atol']
            row = dict(fixture, observed_reference_area=observed,
                       reference_behavior_matches_prediction=matches_prediction,
                       reference_matches_manual_joint_area=matches_manual)
            outcomes.append(row)
            print(json.dumps(row))
            assert matches_prediction, 'Unexpected pinned reference behavior'
        disagreements = [row['id'] for row in outcomes if not row['reference_matches_manual_joint_area']]
        assert disagreements == ['all_known_wrong', 'wrong_tied_scores']
        run.save('reference_tests.json', dict(
            fixture_audit_passed=True, reference_validated_for_H007=False,
            reference_verdict='REJECTED: manual zero-CCR area invariant violated',
            fixtures=outcomes, disagreements=disagreements,
            manual_source_expression_max_FPR=2 / (2 + 2),
            source_expression_diagnostic='FP/(FP+Nunknown); not a returned upstream curve',
            source_module_imported=False, only_audited_AST_function_executed=True,
            published_paper_scores_reproduced=False, scientific_quality_access='none',
            scope='Pinned version synthetic scorer audit only; no inference about published experimental scores.'))


if __name__ == '__main__':
    main()
