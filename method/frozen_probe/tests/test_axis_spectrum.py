"""Independent analytic controls and eigenspace-identifiability guard."""
import unittest
import numpy as np
from axis_spectrum import factor, transforms, unit_rows


class AxisTests(unittest.TestCase):
    def test_diagonal_and_rotated_two_dimensional_manual_oracle(self):
        # R directions coordinate axes; T rotates45degrees. Spectra[1,4]/[4,9].
        v = np.array([[1., -1.], [1., 1.]]) / np.sqrt(2)
        r = np.diag([1., 4.]); t = v @ np.diag([4., 9.]) @ v.T
        lr, vr, _ = factor(r); lt, vt, _ = factor(t)
        w = transforms({'R': vr, 'T': vt}, {'R': lr, 'T': lt})
        self.assertTrue(np.allclose(w['RR'], np.diag([1., .5]), atol=1e-14, rtol=0))
        self.assertTrue(np.allclose(w['TR'], [[.75, .25], [.25, .75]], atol=1e-14, rtol=0))
        self.assertTrue(np.allclose(w['RT'], np.diag([.5, 1/3]), atol=1e-14, rtol=0))
        self.assertTrue(np.allclose(w['TT'], [[5/12, 1/12], [1/12, 5/12]], atol=1e-14, rtol=0))
        # Original-space inner product/norm is an independent analytic metric.
        x = np.array([[1., 0.], [1., 1.], [0., 2.]])
        z = unit_rows(x @ w['TR'])
        metric = np.array([[.625, .375], [.375, .625]])
        products = x @ metric @ x.T
        expected = products / np.sqrt(np.diag(products)[:, None] * np.diag(products)[None, :])
        self.assertTrue(np.allclose(z @ z.T, expected, atol=1e-14, rtol=0))

    def test_rotation_after_whitening_cannot_explain_cosine_changes(self):
        x = np.array([[1., 2.], [-2., 1.], [3., -1.]])
        rotation = np.array([[0., -1.], [1., 0.]])
        z = unit_rows(x @ np.diag([.5, 2.]))
        self.assertTrue(np.allclose(z @ z.T, (z @ rotation) @ (z @ rotation).T, atol=1e-15))
        self.assertTrue(np.allclose(unit_rows(x) @ unit_rows(x).T,
                                   unit_rows(x * 7.) @ unit_rows(x * 7.).T, atol=1e-15))

    def test_repeated_near_degenerate_and_invalid_covariance_guards(self):
        for c in (np.eye(2), np.diag([1., 1.+1e-13]), np.diag([0., 2.]),
                  np.array([[1., .2], [0., 2.]]), np.full((2, 2), np.nan)):
            with self.assertRaises(ValueError): factor(c)
        with self.assertRaises(ValueError): unit_rows(np.zeros((1, 2)))


if __name__ == '__main__':
    unittest.main()
