"""Manual weighted-query/seed-ratio and curve-encoding oracles."""
import numpy as np
from rejection_analysis import cluster_draws, cluster_mean_rate, conditional_interval, pack_count_curve


def run_tests():
    groups, draws = ['A', 'A', 'B'], np.array([[2, 0], [1, 1], [0, 2]])
    a = np.array([[True, False, False]])
    m = np.ones((1, 3), dtype=bool)
    rates, valid = cluster_mean_rate(a, m, groups, draws)
    assert valid.all() and np.allclose(rates, [.5, 1/3, 0], atol=1e-12, rtol=0)
    # Query-count denominator matters: mean of per-cluster means is not1/3.
    assert rates[1] != .25
    seeds = np.array([[True, False, False], [True, False, True]])
    masks = np.array([[True, True, True], [True, False, True]])
    mean, valid = cluster_mean_rate(seeds, masks, groups, draws)
    assert valid.all() and np.allclose(mean, [.75, 2/3, .5], atol=1e-12, rtol=0)
    masks = np.array([[True, True, False]])
    _, valid = cluster_mean_rate(a, masks, groups, draws)
    assert valid.tolist() == [True, True, False]
    assert conditional_interval(np.array([.5, .5, 0]), valid)['interval95'] is None
    assert conditional_interval(np.array([.5, .5, 0]), valid)['invalid_draws'] == 1
    names, sampled = cluster_draws(groups, 20, np.random.default_rng(1701))
    assert names == ['A', 'B'] and np.array_equal(sampled.sum(axis=1), np.full(20, 2))
    points = [dict(threshold='+inf', known_accept_count=0, known_correct_accept_count=0, unknown_accept_count=0),
              dict(threshold=.8, known_accept_count=2, known_correct_accept_count=1, unknown_accept_count=0),
              dict(threshold=.2, known_accept_count=2, known_correct_accept_count=1, unknown_accept_count=2)]
    encoded = pack_count_curve(points)
    assert np.array_equal(encoded, [[0, 0, 0, 0], [.8, 2, 1, 0], [.2, 2, 1, 2]])
    for operation in (lambda: pack_count_curve([]), lambda: pack_count_curve([dict(points[0], known_accept_count=1)]),
                      lambda: cluster_mean_rate(a, m, groups, np.array([[0, 0]]))):
        try: operation()
        except ValueError: pass
        else: raise AssertionError('Invalid counts/resampling accepted')
    groups_passed = ['manual_query_weighted_cluster_rates', 'equal_seed_mean_not_pooled_queries',
                     'undefined_draw_retained_no_CI', 'fixed_multinomial_draw_counts', 'manual_finite_curve_encoding_and_guards']
    for name in groups_passed: print(name, 'PASS')
    return dict(passed=True, groups=groups_passed, manual_cluster_rates=rates.tolist(),
                manual_equal_seed_mean=mean.tolist(), invalid_bootstrap_CI='unavailable; no discarded/resampled draws',
                finite_curve_rows=encoded.tolist(), scientific_quality_access='none')
