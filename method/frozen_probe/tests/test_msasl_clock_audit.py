"""Manual temporal and framehash fixtures, independent of actual target quality."""
from msasl_clock_audit import clock_comparison, parse_framehash


def run_tests():
    # Origin 100, half-open [1,3); raw backward frame102 remains its own ordinal.
    a = clock_comparison([100,101,102,101,103], [100,101,102,103,104], '1', 100, '1', '3')
    assert a['raw_selected_indices'] == [1,2,3]
    assert a['best_selected_indices'] == [1,2]
    assert a['removed_by_best_indices'] == [3] and a['added_by_best_indices'] == []
    assert a['raw_nonincreasing_indices'] == [3] and a['best_nonincreasing_indices'] == []
    assert a['maximum_absolute_clock_delta_seconds'] == 2
    b = clock_comparison([9,11,10,13], [9,11,12,13], '1/10', 0, '1.1', '1.3')
    assert b['raw_selected_indices'] == [1] and b['best_selected_indices'] == [1,2]
    assert b['added_by_best_indices'] == [2] and b['removed_by_best_indices'] == []
    assert b['maximum_absolute_clock_delta_seconds'] == .2
    # Count stride2 windows without padding, and do not collapse equal raw ticks.
    c = clock_comparison([10]*18, list(range(10,28)), '1/10', 0, '1', '3')
    assert c['raw_selected_indices'] == list(range(18))
    assert c['raw_nonincreasing_indices'] == list(range(1,18))
    assert c['raw_complete_native16_stride2_windows'] == 2
    for raw,best in (([],[]),([1],[1,2]),([1.0],[1])):
        try: clock_comparison(raw,best,'1',0,'0','2')
        except ValueError: pass
        else: raise AssertionError('Invalid clocks accepted')
    digest = 'ab'*32
    out = parse_framehash('#tb 0: 1/90000\n0, 101, 102, 3, 12, '+digest+'\n')
    assert out == {'time_base':'1/90000','frames':[dict(dts=101,pts=102,duration=3,bytes=12,sha256=digest)]}
    for text in ('0, 1, 1, 1, 12, '+digest, '#tb 0: 1/30\n1, 1, 1, 1, 12, '+digest,
                 '#tb 0: 1/30\n0, 1, 1, 1, 12, bad'):
        try: parse_framehash(text)
        except ValueError: pass
        else: raise AssertionError('Malformed framehash accepted')
    return dict(passed=True, groups=['rational_halfopen_origin_backward_ordinals',
                                    'changed_membership_and_count_no_padding',
                                    'clock_shape_and_type_guards', 'independent_framehash_parser_oracle'],
                GPU_hours=0, scope='Synthetic software only; no target cohort or quality')
