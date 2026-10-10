"""Manual score/calibration oracles plus audited author CPU component parity."""
import ast
from math import exp, log
from pathlib import Path
import numpy as np
from frozen_rejection import fixed_scores, fit_source_known_threshold


def error(function, *args, **kwargs):
    try: function(*args, **kwargs)
    except ValueError: return
    raise AssertionError('Invalid input accepted')


def run_tests(source):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    tree = ast.parse(Path(source).read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('NAC', 'cosine')]
    assert [(n.name, n.lineno, n.end_lineno) for n in functions] == [('cosine', 8, 13), ('NAC', 16, 28)]
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for f in functions for n in ast.walk(f))
    namespace = {'torch': torch, 'F': F, '__builtins__': {'__import__': __import__}}
    exec(compile(ast.Module(body=functions, type_ignores=[]), source, 'exec'), namespace)
    groups, records = [], []
    for logits, expected in [([log(2), 0., -1.], [log(2), log(2), 2/(3+exp(-1)), 2/3]),
                             ([0., 0., -1.], [0., 0., 1/(2+exp(-1)), .5])]:
        scores, prediction, ranking = fixed_scores([logits], [7, 11, 18], nac_k=2)
        assert np.max(np.abs(scores[0] - expected)) <= 1e-12
        assert prediction.tolist() == [7] and ranking[0].tolist() == [7, 11, 18]
        records.append(dict(logits=logits, expected=expected, observed=scores[0].tolist()))
    q = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
    g = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]], dtype=torch.float64)
    assert np.array_equal(namespace['cosine'](q, g).numpy(), [[1., 0., -1.], [0., 1., 0.]])
    groups.append('manual_score_and_unit_cosine_oracles')
    rng = np.random.default_rng(1701)
    parity = []
    for i in range(30):
        n = [10, 25, 50][i % 3]
        logits = rng.uniform(-1, 1, size=(3, n))
        assert all(len(np.unique(row)) == n for row in logits)
        owned, prediction, _ = fixed_scores(logits, list(range(n)))
        ref, index = namespace['NAC'](torch.tensor(logits, dtype=torch.float64), k=min(16, n), s=1)
        delta = float(np.max(np.abs(owned[:, 3] - ref.numpy())))
        assert delta <= 1e-12 and np.array_equal(prediction, index.numpy())
        parity.append(dict(case=i, gallery_classes=n, rows=3, maximum_error=delta, predictions_exact=True))
    groups.append('thirty_author_AST_CPU_parities')
    low = torch.tensor([[.9, .8, .7, .6, .5, .4, .3, .2, .1, 0.]], dtype=torch.float64)
    try: namespace['NAC'](low)
    except RuntimeError: pass
    else: raise AssertionError('Expected upstream topk16 boundary error on10classes')
    owned, _, _ = fixed_scores(low.numpy(), list(range(10)))
    assert owned[0, 2] == owned[0, 3]
    ref, _ = namespace['NAC'](torch.tensor([[0., 0., -1.]], dtype=torch.float64), k=2)
    assert ref.item() == .5
    groups.append('explicit_low_gallery_clamp_and_tie_confidence')
    fit = fit_source_known_threshold(list(range(1, 21)), [True] * 20)
    assert fit['threshold'] == 2 and fit['accepted_known_count'] == 19 and fit['empirical_known_acceptance'] == .95
    tied = fit_source_known_threshold([1., 1., 2.], [True] * 3, 2, 3)
    assert tied['threshold'] == 1 and tied['tie_overshoot'] == 1
    assert fit_source_known_threshold([.3], [True])['threshold'] == .3
    mask = [True] * 20 + [False, False]
    a = fit_source_known_threshold(list(range(1, 21)) + [-100, -100], mask)
    b = fit_source_known_threshold(list(range(1, 21)) + [100, 100], mask)
    assert a == b == fit
    assert fit_source_known_threshold(list(range(20, 0, -1)), [True] * 20) == fit
    groups.append('manual_known_quantile_ties_excluded_unknown_and_permutation')
    for args in (([], []), ([1], [False]), ([float('nan')], [True]), ([1], [1]), ([1, 2], [True]),
                 ([1], [True], 0, 20), ([1], [True], 21, 20), ([1], [True], True, 20)):
        error(fit_source_known_threshold, *args)
    for args in (([], [0, 1]), ([[0]], [0]), ([[0, 1]], [1, 0]), ([[0, 1]], [0, 0]),
                 ([[0, 1]], [0, True]), ([[0, 1]], [-1, 0]), ([[float('nan'), 1]], [0, 1])):
        error(fixed_scores, *args)
    for kwargs in ({'nac_k': 1}, {'nac_k': True}, {'scale': 0}, {'scale': float('nan')}):
        error(fixed_scores, [[0., 1.]], [0, 1], **kwargs)
    groups.append('nineteen_score_and_calibration_error_guards')
    for name in groups: print(name, 'PASS')
    return dict(passed=True, groups=groups, manual_scores=records, author_CPU_parity=parity,
                source_known_calibration=fit, tie_calibration=tied, upstream_default_low_gallery_raises=True,
                numpy_version=np.__version__, torch_version=torch.__version__, device='CPU',
                author_module_imported=False, AST_components_only=True, scientific_quality_access='none',
                scope='NAC scoring component only; k clamp/tie policy explicitly adapted, no face pipeline reproduction.')
