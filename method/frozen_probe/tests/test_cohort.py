import unittest
from scripts.build_cohort import select_cohort


class CohortTests(unittest.TestCase):
    def rows(self, split):
        return [{'Video file': f'{split}_{c}_{i}', 'Gloss': c, 'Participant ID': split}
                for c in ('A','B','C') for i in range(8 if split=='train' else 2)]

    def test_nested_support_and_no_dev_selection(self):
        train, dev = self.rows('train'), self.rows('val')
        result = select_cohort(train, dev, 2, 1701, [17,23], [1,2,4,8])
        for support in result['supports'].values():
            for label in result['classes']:
                self.assertEqual(support['1'][label], support['8'][label][:1])
                self.assertEqual(support['4'][label], support['8'][label][:4])
                self.assertTrue(all(i.startswith('train_') for i in support['8'][label]))
        self.assertEqual(result['classes'], select_cohort(train, list(reversed(dev)),2,1701,[17,23],[1,2,4,8])['classes'])
        self.assertEqual(sum(s['split']=='val' for s in result['samples']),4)

    def test_leakage_and_duplicate_ids_rejected(self):
        train, dev = self.rows('train'), self.rows('val')
        dev[0]['Participant ID'] = 'train'
        with self.assertRaises(ValueError):
            select_cohort(train, dev, 2, 1701, [17], [8])
        with self.assertRaises(ValueError):
            select_cohort(train+[train[0]], self.rows('val'), 2, 1701, [17], [8])


if __name__ == '__main__':
    unittest.main()
