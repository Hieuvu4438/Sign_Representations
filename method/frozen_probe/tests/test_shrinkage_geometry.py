from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shrinkage_geometry import fit_shrinkage, whitening, transformed, lookup_scores


class ShrinkageTests(unittest.TestCase):
    def test_independent_fourth_moment_shrinkage_oracle(self):
        x = np.array([[2,1,0],[-2,-1,1],[1,-2,-1],[-1,2,0]], dtype=float)
        labels = [0,0,1,1]
        saved = x.copy()
        fitted, metadata = fit_shrinkage(x, labels, 1701)
        for family in ('residual', 'total', 'shuffled'):
            y = x.copy()
            if family == 'total':
                y -= y.mean(axis=0)
            else:
                actual = labels if family == 'residual' else metadata['shuffled_labels']
                for label in set(actual):
                    ids = np.where(np.array(actual) == label)[0]
                    y[ids] -= x[ids].mean(axis=0)
            # Row outer products/fourth moments, without sklearn's blocked formula.
            covariance = sum(np.outer(row, row) for row in y) / len(y)
            d = x.shape[1]
            mu = np.trace(covariance) / d
            delta = np.sum((covariance - mu * np.eye(d)) ** 2) / d
            beta = (np.mean([np.dot(row,row) ** 2 for row in y]) - np.sum(covariance ** 2)) / (len(y) * d)
            coefficient = 0.0 if beta == 0 else min(beta, delta) / delta
            expected = (1 - coefficient) * covariance + coefficient * mu * np.eye(d)
            self.assertAlmostEqual(coefficient, metadata['families'][family]['shrinkage'], places=12)
            np.testing.assert_allclose(fitted[family + '_covariance'], expected, atol=1e-12)
            w = fitted[family + '_whitening']
            np.testing.assert_allclose(w @ expected @ w, np.eye(d), atol=1e-12)
        np.testing.assert_array_equal(x, saved)

    def test_mahalanobis_and_cosine_have_distinct_known_ranks(self):
        fitted = {'mean': np.zeros(2), 'residual_whitening': np.eye(2)}
        q = np.array([[1.,0.]])
        g = np.array([[.1,0.],[1.,.5]])
        for metric, expected in [('cos',0), ('maha',1)]:
            mode = 'residual_lw_' + metric
            scores = lookup_scores(transformed(q,fitted,mode),transformed(g,fitted,mode),mode)
            self.assertEqual(int(np.argmax(scores[0])), expected)
        np.testing.assert_allclose(lookup_scores(q,g,'residual_lw_maha'),[[-.81,-.25]],atol=1e-15)
        w, _ = whitening(np.diag([4.,1.]))
        np.testing.assert_allclose(w,np.diag([.5,1.]),atol=1e-15)

    def test_degenerate_covariance_zero_cosine_and_nonfinite_abort(self):
        for invalid in (np.zeros((2,2)),np.diag([-1.,2.]),np.array([[1.,1.],[0.,1.]])):
            with self.assertRaises(ValueError):whitening(invalid)
        fitted={'mean':np.array([1.,0.]),'residual_whitening':np.eye(2)}
        with self.assertRaises(ValueError):transformed([[1.,0.]],fitted,'residual_lw_cos')
        with self.assertRaises(ValueError):lookup_scores([[np.nan,0.]],[[1.,0.]],'residual_lw_maha')


if __name__ == '__main__':
    unittest.main()
