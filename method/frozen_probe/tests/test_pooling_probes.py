import unittest
import numpy as np
from pooling import activity_mean,ordered_bins,source_permutation,pool_controls
from probes import normalize_rows,fit_ridge,predict_ridge,fit_prototypes,nearest_class_scores


class PoolingProbeTests(unittest.TestCase):
    def test_activity_regression_oracle(self):
        x=np.array([[2.,4.],[6.,8.]])
        value,meta=activity_mean(x,[[-1,.25],[1.5,-2]])
        np.testing.assert_allclose(value,[5.2,7.2]);self.assertFalse(meta['fallback_mean'])
        value,meta=activity_mean(x,[[-1,-2],[-.1,-.2]])
        np.testing.assert_allclose(value,[4,6]);self.assertTrue(meta['fallback_mean'])

    def test_irregular_bins_empty_policy_projection(self):
        bins,meta=ordered_bins([[1],[3],[7]],[0,.1,1],4)
        self.assertEqual(meta['bin_counts'],[2,0,0,1])
        np.testing.assert_allclose(bins[:,0],[2,11/3,11/3,7])
        x=np.array([[1.,2.],[3.,4.]])
        projection=np.arange(16).reshape(8,2)/16
        values,_=pool_controls(x,[0,1],[[0,1],[0,1]],'a'*64,projection)
        np.testing.assert_allclose(values['mean_projected'],np.tile([2,3],4)@projection)
        np.testing.assert_allclose(values['max'],[3,4])
        self.assertEqual(sorted(source_permutation('a'*64).tolist()),[0,1,2,3])
        np.testing.assert_equal(source_permutation('a'*64),source_permutation('a'*64))

    def test_ridge_independent_oracle_and_dual_primal(self):
        # X=[-1,1], class-one centered Y=[-.5,.5], alpha2 => slope1/4.
        model=fit_ridge([[-1],[1]],[0,1],2,2)
        np.testing.assert_allclose(model['coef'],[[-.25,.25]])
        np.testing.assert_allclose(model['intercept'],[.5,.5])
        np.testing.assert_allclose(predict_ridge(model,[[0],[2]]),[[.5,.5],[0,1]])
        dual=fit_ridge([[-1,0,0],[1,0,0]],[0,1],2,2)
        np.testing.assert_allclose(predict_ridge(dual,[[0,0,0],[2,0,0]]),[[.5,.5],[0,1]])

    def test_prototype_and_invalid_inputs(self):
        np.testing.assert_allclose(fit_prototypes([[1,0],[2,0],[0,1]],[0,0,1],2),np.eye(2))
        with self.assertRaises(ValueError):normalize_rows([[0,0]])
        with self.assertRaises(ValueError):fit_ridge([[1,0]],[0],2,1)
        with self.assertRaises(ValueError):ordered_bins([[1],[2]],[1,0])
        with self.assertRaises(ValueError):activity_mean([[1]],[[float('nan'),0]])

    def test_nearest_class_collapse_oracle(self):
        scores=nearest_class_scores([[1,0],[0,1],[-1,0]],[0,0,1],[[1,0],[0,1]],2)
        np.testing.assert_allclose(scores,[[1,-1],[1,0]])
        permuted=nearest_class_scores([[-1,0],[1,0],[0,1]],[1,0,0],[[1,0],[0,1]],2)
        np.testing.assert_allclose(scores,permuted)


if __name__=='__main__':unittest.main()
