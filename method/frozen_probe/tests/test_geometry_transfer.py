from collections import Counter
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry_transfer import class_residuals,fit_geometry,apply_geometry,disjoint_cohorts,common_feature_recipe


class GeometryTests(unittest.TestCase):
    def test_manual_residual_total_and_pair_covariance(self):
        x=np.array([[10,1,0],[10,-1,0],[-10,1,0],[-10,-1,0]],dtype=float)
        labels=[0,0,1,1]
        residual=class_residuals(x,labels)
        np.testing.assert_array_equal(residual,[[0,1,0],[0,-1,0],[0,1,0],[0,-1,0]])
        # Independent weighted ordered-pair identity: sum(diff diff^T)/(2*n_c).
        pair=np.zeros((3,3))
        for label in (0,1):
            rows=x[np.asarray(labels)==label]
            for a in rows:
                for b in rows:pair+=np.outer(a-b,a-b)/(2*len(rows))
        np.testing.assert_array_equal(pair,residual.T@residual)
        f=fit_geometry(x,labels,1,1701)
        np.testing.assert_allclose(f['residual_pca']@f['residual_pca'].T,np.diag([0.,1.,0.]),atol=1e-15)
        np.testing.assert_allclose(f['total_pca']@f['total_pca'].T,np.diag([1.,0.,0.]),atol=1e-15)
        self.assertEqual(Counter(f['shuffled_labels']),Counter(labels))

    def test_known_projection_and_fit_immutable(self):
        fitted={'mean':np.array([1.,0.,0.]),'residual_pca':np.array([[0.],[1.],[0.]])}
        before=fitted['mean'].copy()
        np.testing.assert_allclose(apply_geometry([[2,8,0]],fitted,'residual_pca'),[[1.,0.,0.]],atol=1e-15)
        np.testing.assert_array_equal(fitted['mean'],before)
        with self.assertRaises(ValueError):apply_geometry([[1,8,0]],fitted,'residual_pca')

    def test_cross_cohort_exclusivity(self):
        base={'classes':['A'],'samples':[{'sample_id':'a','source_sha256':'x'}]}
        target={'classes':['B'],'samples':[{'sample_id':'b','source_sha256':'y'}]}
        disjoint_cohorts(base,target)
        for bad in [{'classes':['A'],'samples':target['samples']},
                    {'classes':['B'],'samples':[{'sample_id':'a','source_sha256':'y'}]},
                    {'classes':['B'],'samples':[{'sample_id':'b','source_sha256':'x'}]}]:
            with self.assertRaises(ValueError):disjoint_cohorts(base,bad)

    def test_only_exact_audit_descriptor_difference_allowed(self):
        import copy
        fields=('checkpoint_sha256','upstream_commit','packages','preprocessing_code_sha256','temporal_code_sha256')
        base={k:'same' for k in fields};base['contract']={'stride':2,'temporal_metadata':'complete_native_PTS_and_FPS_in_hash_bound_VID002_audit'}
        target=copy.deepcopy(base);target['contract']['temporal_metadata']='complete native PTS and FPS in hash-bound VID003 audit'
        common_feature_recipe(base,target)
        bad=copy.deepcopy(target);bad['contract']['stride']=4
        with self.assertRaises(ValueError):common_feature_recipe(base,bad)
        bad=copy.deepcopy(target);bad['contract']['temporal_metadata']='guessed timeline'
        with self.assertRaises(ValueError):common_feature_recipe(base,bad)


if __name__=='__main__':unittest.main()
