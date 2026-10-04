from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from participant_removal import fit_removal, leading_basis, transform
from metrics import evaluate, ranked_classes


class RemovalTests(unittest.TestCase):
    def test_manual_projection_and_zero_handling(self):
        fitted={'mean':np.zeros(3),'participant':np.array([[1.],[0.],[0.]])}
        actual=transform([[4.,3.,0.]],fitted,'participant')
        np.testing.assert_allclose(actual,[[0.,1.,0.]],atol=1e-15,rtol=0)
        with self.assertRaises(ValueError): transform([[4.,0.,0.]],fitted,'participant')

    def test_class_offsets_do_not_fit_participant_direction(self):
        # Class direction x; participant direction y; dominant residual basis y.
        x=np.array([[10,1,0],[10,-1,0],[-10,1,0],[-10,-1,0]],dtype=float)
        f=fit_removal(x,[0,0,1,1],['A','B','A','B'],1,1701)
        np.testing.assert_allclose(f['participant']@f['participant'].T,np.diag([0.,1.,0.]),atol=1e-15)
        np.testing.assert_allclose(f['residual_pca']@f['residual_pca'].T,np.diag([0.,1.,0.]),atol=1e-15)
        # Class-dependent offset leaves within-class directions unchanged.
        shifted=x+np.array([[2,3,4],[2,3,4],[-8,4,9],[-8,4,9]])
        g=fit_removal(shifted,[0,0,1,1],['A','B','A','B'],1,1701)
        np.testing.assert_allclose(g['participant']@g['participant'].T,f['participant']@f['participant'].T,atol=1e-15)

    def test_rank_guards_and_random_orthogonality(self):
        with self.assertRaises(ValueError): leading_basis(np.zeros((3,4)),1)
        with self.assertRaises(ValueError): leading_basis(np.eye(3),3)
        x=np.array([[10,1,0],[10,-1,0],[-10,1,0],[-10,-1,0]],dtype=float)
        f=fit_removal(x,[0,0,1,1],['A','B','A','B'],1,1701)
        np.testing.assert_allclose(f['random'].T@f['random'],np.eye(1),atol=1e-15)

    def test_noncontiguous_labels_full_ranks_and_dcg(self):
        classes=[4,9,15]
        # Known tied scores resolve via original class IDs, not local indices.
        ranking=ranked_classes([0.,0.,1.],classes)
        self.assertEqual(ranking,[15,4,9])
        scored=evaluate([{'sample_id':'x','ranking':ranking}],[{'sample_id':'x','label':9}],classes)
        self.assertEqual(scored['dcg'],.5)
        self.assertEqual(scored['mrr'],1/3)
        with self.assertRaises(ValueError):
            evaluate([{'sample_id':'x','ranking':[15,4]}],[{'sample_id':'x','label':9}],classes)


if __name__=='__main__':unittest.main()
