import unittest
from scripts.screen_duplicates import compare_pair


class DuplicateScreenTests(unittest.TestCase):
    def sample(self,hash_value,information=10):
        return {'frames':100,'fingerprints':[{'dhash64':hash_value,'gray_std':information}]*5}
    def test_bit_distance_oracle(self):
        zero=self.sample('0000000000000000')
        self.assertEqual(compare_pair(zero,zero)['distances'],[0]*5)
        self.assertEqual(compare_pair(zero,self.sample('0000000000000001'))['distances'],[1]*5)
        full=compare_pair(zero,self.sample('ffffffffffffffff'))
        self.assertEqual(full['distances'],[64]*5)
        self.assertFalse(full['candidate'])
    def test_low_information_and_duration(self):
        self.assertIsNone(compare_pair(self.sample('0000000000000000',0),self.sample('0000000000000000')))
        short=self.sample('0000000000000000');short['frames']=50
        self.assertIsNone(compare_pair(short,self.sample('0000000000000000')))


if __name__=='__main__':unittest.main()
