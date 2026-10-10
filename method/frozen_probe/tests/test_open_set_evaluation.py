"""Manual/rational oracles for adapted joint counts; no sign-data quality access."""
from fractions import Fraction

import numpy as np

from open_set_evaluation import JointOpenSet


IDS = ['q0', 'q1', 'q2', 'q3']
GALLERY = [0, 1]
FIXTURES = [
    ('wrongknown', [1, 1, 2, 2], [.8, .8, .2, .2], 0.),
    ('perfect', [0, 0, 2, 2], [.8, .8, .2, .2], 1.),
    ('reversed', [0, 0, 2, 2], [.2, .2, .8, .8], 0.),
    ('alltied', [0, 0, 2, 2], [.9, .9, .9, .9], .5),
    ('halfcorrect', [0, 1, 2, 2], [.9, .7, .8, .6], .5),
]


def make(labels, scores):
    return JointOpenSet(IDS, labels, [0] * 4, scores, GALLERY, IDS)


def counts(point):
    return point['known_accept_count'], point['known_correct_accept_count'], point['unknown_accept_count']


def expect_error(arguments=None, threshold=None):
    try:
        if arguments is None:
            make(FIXTURES[1][1], FIXTURES[1][2]).at_threshold(threshold)
        else:
            JointOpenSet(**arguments)
    except ValueError:
        return
    raise AssertionError('Invalid evaluator input did not raise ValueError')


def run_tests():
    results = []
    for name, labels, scores, expected in FIXTURES:
        curve = make(labels, scores).curve()
        assert abs(curve['grouped_trapezoid_AUC'] - expected) <= 1e-12
        assert curve['points'][-1]['FAR'] == 1
        assert curve['points'][-1]['CCR'] == curve['points'][-1]['closed_known_accuracy']
        results.append(dict(id=name, expected_area=expected, observed=curve))
    groups = ['five_manual_areas']
    half = make(FIXTURES[-1][1], FIXTURES[-1][2])
    thresholds = [float('inf'), .9, .8, .7, .6, -float('inf')]
    expected_counts = [(0, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 1), (2, 1, 2), (2, 1, 2)]
    for threshold, expected in zip(thresholds, expected_counts):
        point = half.at_threshold(threshold)
        assert counts(point) == expected
        assert (point['n_known'], point['n_unknown']) == (2, 2)
        assert point['FAR'] == expected[2] / 2 and point['CCR'] == expected[1] / 2
    assert half.at_threshold(-float('inf'))['CCR'] == .5
    groups.append('manual_counts_closed_accuracy_endpoint')
    tied = make(FIXTURES[3][1], FIXTURES[3][2])
    assert counts(tied.at_threshold(.9)) == (2, 2, 2)
    assert counts(tied.at_threshold(.9001)) == (0, 0, 0)
    assert counts(make(FIXTURES[1][1], FIXTURES[1][2]).at_threshold(.8)) == (2, 2, 0)
    permutation = [3, 1, 0, 2]
    for _, labels, scores, _ in FIXTURES:
        reordered_ids = [IDS[i] for i in permutation]
        reordered = JointOpenSet(reordered_ids, [labels[i] for i in permutation], [0] * 4,
                                 [scores[i] for i in permutation], GALLERY, reordered_ids)
        assert reordered.curve() == make(labels, scores).curve()
    groups.append('inclusive_ties_and_synchronized_permutation')
    base = dict(sample_ids=IDS, labels=[0, 1, 2, 2], predicted_classes=[0] * 4,
                scores=[.9, .7, .8, .6], gallery_classes=GALLERY, expected_ids=IDS)
    variants = [
        dict(sample_ids=[]), dict(sample_ids=['q0', 'q0', 'q2', 'q3']), dict(sample_ids=['', 'q1', 'q2', 'q3']),
        dict(sample_ids=[0, 1, 2, 3]), dict(sample_ids=[0, 'q1', 'q2', 'q3'], expected_ids=[0, 'q1', 'q2', 'q3']),
        dict(expected_ids=IDS[::-1]), dict(expected_ids=IDS[:-1]),
        dict(labels=[0, 1, 2]), dict(labels=[0, True, 2, 2]), dict(labels=[0., 1., 2., 2.]),
        dict(labels=[0, -1, 2, 2]), dict(predicted_classes=[0, 0, 2, 0]),
        dict(predicted_classes=[0, 0, False, 0]), dict(gallery_classes=[]), dict(gallery_classes=[0, 0]),
        dict(gallery_classes=['0', '1']), dict(scores=[.9, .7, float('nan'), .6]),
        dict(scores=[.9, .7, float('inf'), .6]), dict(scores=[[.9], [.7], [.8], [.6]]),
        dict(scores=['.9', '.7', '.8', '.6']), dict(scores=[True] * 4),
        dict(labels=[0, 1, 0, 1]), dict(labels=[2] * 4),
    ]
    for delta in variants:
        expect_error({**base, **delta})
    groups.append('twenty_three_schema_coverage_class_and_group_guards')
    for threshold in (float('nan'), True, '.8', [.8], 1j):
        expect_error(threshold=threshold)
    groups.append('five_threshold_guards')
    rng = np.random.default_rng(1701)
    rational_cases = []
    for case in range(30):
        labels = [0, 1, 2, 3] + rng.integers(0, 4, size=20).tolist()
        predictions = rng.integers(0, 2, size=24).tolist()
        scores = rng.integers(-2, 3, size=24).tolist()
        ids = [f'fixture_{case}_{i}' for i in range(24)]
        curve = JointOpenSet(ids, labels, predictions, scores, GALLERY, ids).curve()
        nk = sum(label in GALLERY for label in labels)
        nu = len(labels) - nk
        threshold_list = [float('inf')] + sorted(set(scores), reverse=True)
        fractions = []
        for threshold, point in zip(threshold_list, curve['points']):
            ka = sum(label in GALLERY and score >= threshold for label, score in zip(labels, scores))
            kca = sum(label in GALLERY and prediction == label and score >= threshold
                      for label, prediction, score in zip(labels, predictions, scores))
            ua = sum(label not in GALLERY and score >= threshold for label, score in zip(labels, scores))
            assert counts(point) == (ka, kca, ua)
            assert (point['n_known'], point['n_unknown']) == (nk, nu)
            fractions.append((Fraction(ua, nu), Fraction(kca, nk)))
        assert len(curve['points']) == len(threshold_list)
        rational_area = sum((right[0] - left[0]) * (right[1] + left[1]) / 2
                            for left, right in zip(fractions, fractions[1:]))
        assert abs(curve['grouped_trapezoid_AUC'] - float(rational_area)) <= 1e-12
        rational_cases.append(dict(case=case, labels=labels, predictions=predictions, scores=scores,
                                   rational_area=str(rational_area), observed=curve))
    groups.append('thirty_independent_python_fraction_cases')
    for name in groups:
        print(name, 'PASS')
    return dict(passed=True, groups=groups, manual_fixtures=results, rational_cases=rational_cases,
                classification_and_acceptance_joint=True, scientific_quality_access='none',
                variant='adapted inclusive grouped trapezoid, not official scalar/tie parity')
