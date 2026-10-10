"""Same-query membership/manual residual oracles and rare-signer coverage."""
import numpy as np
import torch
from paired_rejection_episodes import paired_plan, validate_paired_plan, paired_inputs
from rejection_heads import selected_bce_indices


def run_tests():
    rows = [dict(sample_id=f'{c:02d}-{p}', class_id=c, signer=p, source_group=f'{c:02d}-{p}')
            for c in range(5) for p in ('A', 'B', 'C')]
    rows.append(dict(sample_id='04-D', class_id=4, signer='D', source_group='04-D'))
    rows.sort(key=lambda r: r['sample_id'])
    plan = paired_plan(rows, steps=30, pairs=3, size=2, seed=1701)
    assert validate_paired_plan(rows, plan, size=2)
    assert np.array_equal(plan, paired_plan(rows, steps=30, pairs=3, size=2, seed=1701))
    assert any(rows[int(i)]['signer'] == 'D' for i in plan[:, :, 0].reshape(-1))
    # Explicit known/unknown residuals, including an incorrectly ranked known.
    vectors = np.array([[1., 0.], [1., 0.], [-1., 0.], [0., 1.], [1., 0.]])
    labels = np.array([0, 0, 1, 2, 1])
    episode = np.array([[4, 2, 1, 3], [0, 1, 2, 3]], dtype=np.int16)
    residual, scores, targets = paired_inputs(vectors, labels, episode)
    assert np.array_equal(residual, [[1., -1.], [0., 0.], [1., -1.], [0., 0.]])
    assert np.array_equal(scores, [[-1., 0.], [1., 0.], [-1., 0.], [1., 0.]])
    assert targets.tolist() == [0, 0, -1, -1]
    selected, _ = selected_bce_indices(torch.from_numpy(scores), torch.from_numpy(targets))
    assert selected.tolist() == [2, 1]  # same second query, unknown and correct known
    invalid = plan.copy(); invalid[0, 0, 2] = invalid[0, 0, 1]
    try: validate_paired_plan(rows, invalid, 2)
    except ValueError: pass
    else: raise AssertionError('Duplicate paired class accepted')
    invalid = plan.copy(); invalid[0, 0, 1] = invalid[0, 0, 0]
    try: validate_paired_plan(rows, invalid, 2)
    except ValueError: pass
    else: raise AssertionError('Self/signer match accepted')
    single = [dict(sample_id=str(i), class_id=i, signer='A', source_group=str(i)) for i in range(5)]
    try: paired_plan(single, steps=1, pairs=1, size=2)
    except ValueError: pass
    else: raise AssertionError('Source with no cross-signer supports accepted')
    return dict(passed=True, groups=['ninety_toy_pairs_all_signer_source_memberships',
        'single_class_signer_retained_and_deterministic', 'manual_wrong_known_and_exact_unknown_counterpart',
        'duplicate_selfmatch_and_empty_cross_signer_guards'], paired_rows=90,
        scope='Fixed paired-input software fixtures only, no head/task quality.')
