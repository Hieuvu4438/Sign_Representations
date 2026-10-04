import copy
import unittest

from dictionary_metrics import evaluate_dictionary
from dictionary_sampling import sample_galleries
from metrics import ranked_classes


class DictionaryProtocolTests(unittest.TestCase):
    def test_independent_micro_macro_imbalance_oracle(self):
        refs = [dict(sample_id=f'a{i}',label=0) for i in range(9)]+[dict(sample_id='b',label=1)]
        preds = [dict(sample_id=r['sample_id'],ranking=[0,1]) for r in reversed(refs)]
        out = evaluate_dictionary(preds,refs,[0,1])
        self.assertEqual(out['micro']['top1'],.9)
        self.assertEqual(out['macro']['top1'],.5)
        self.assertAlmostEqual(out['micro']['dcg'],.9630929753571458,places=14)
        self.assertAlmostEqual(out['macro']['dcg'],.8154648767857288,places=14)
        self.assertEqual(out['class_support'],{'0':9,'1':1})

    def test_manual_rank3_and_order_join_oracle(self):
        refs = [dict(sample_id='a',label=0),dict(sample_id='b',label=1),dict(sample_id='c',label=2)]
        preds = [dict(sample_id='c',ranking=[0,1,2]),dict(sample_id='a',ranking=[0,1,2]),dict(sample_id='b',ranking=[1,0,2])]
        out = evaluate_dictionary(preds,refs,[0,1,2])
        self.assertEqual(out['per_class']['2']['dcg'],.5)
        self.assertAlmostEqual(out['micro']['dcg'],5/6,places=14)
        self.assertAlmostEqual(out['macro']['mrr'],7/9,places=14)
        self.assertEqual(out['micro']['top5'],1)

    def test_missing_class_or_prediction_never_averaged_away(self):
        refs=[dict(sample_id='a',label=0),dict(sample_id='b',label=1)]
        valid=[dict(sample_id='a',ranking=[0,1]),dict(sample_id='b',ranking=[1,0])]
        for preds in ([],valid[:1],valid+[valid[0]],valid+[dict(sample_id='extra',ranking=[0,1])]):
            with self.assertRaises(ValueError):evaluate_dictionary(preds,refs,[0,1])
        with self.assertRaises(ValueError):evaluate_dictionary(valid[:1],refs[:1],[0,1])
        with self.assertRaises(ValueError):evaluate_dictionary(valid,refs,[0,1,2])

    def test_ties_nonfinite_and_invalid_rankings(self):
        self.assertEqual(ranked_classes([1.,1.,2.],[9,4,5]),[5,4,9])
        for x in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):ranked_classes([x,0.],[0,1])
        refs=[dict(sample_id='a',label=0),dict(sample_id='b',label=1)]
        for rank in ([0,0],[0],[0,2]):
            with self.assertRaises(ValueError):
                evaluate_dictionary([dict(sample_id='a',ranking=rank),dict(sample_id='b',ranking=[1,0])],refs,[0,1])

    def cohort(self):
        rows=[]
        for label in (0,1):
            for i in range(3):
                rows.append(dict(sample_id=f't{label}{i}',class_id=label,split='train',signer_id=label,
                                 parent_id=f'train_parent_{label}_{i}',status='eligible'))
            rows.append(dict(sample_id=f'q{label}',class_id=label,split='val',signer_id=10+label,
                             parent_id=f'query_parent_{label}',status='eligible'))
        return rows

    def test_sampling_order_independence_membership_and_shared_supports(self):
        rows=self.cohort();a=sample_galleries(rows,[0,1]);b=sample_galleries(list(reversed(rows)),[0,1])
        self.assertEqual(a,b)
        self.assertEqual(a['query_ids'],['q0','q1'])
        self.assertEqual(set(a['supports']),{'1701','1702','1703'})
        for support in a['supports'].values():
            self.assertEqual(len(support),2)
            self.assertIn(support[0],['t00','t01','t02'])
            self.assertIn(support[1],['t10','t11','t12'])
        self.assertEqual(a['native_train_counts'],{'0':3,'1':3})

    def test_sampler_rejects_leakage_before_reselection(self):
        for field in ('signer_id','parent_id'):
            rows=self.cohort();rows[-1][field]=rows[0][field]
            with self.assertRaises(ValueError):sample_galleries(rows,[0,1])

    def test_sampler_requires_complete_classes_ids_and_eligibility(self):
        rows=self.cohort()
        for malformed in ([r for r in rows if not (r['split']=='val' and r['class_id']==1)],rows+[rows[0]],
                          [r for r in rows if not (r['split']=='train' and r['class_id']==1)]):
            with self.assertRaises(ValueError):sample_galleries(malformed,[0,1])
        for status in ('incomplete','native_short'):
            malformed=copy.deepcopy(rows);malformed[0]['status']=status
            with self.assertRaises(ValueError):sample_galleries(malformed,[0,1])
        for seeds in ([],[True],[1701,1701],[-1]):
            with self.assertRaises(ValueError):sample_galleries(rows,[0,1],seeds)
