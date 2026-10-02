import unittest

import torch

from signrepr.scoring import token_nll


class TeacherForcedScoring(unittest.TestCase):
    def test_padding_does_not_change_valid_token_nll(self):
        logits = torch.tensor([[[2., 0.], [0., 2.], [100., -100.]]])
        padded = token_nll(logits, torch.tensor([[0, 1, -100]]))
        short = token_nll(logits[:, :2], torch.tensor([[0, 1]]))
        self.assertEqual(padded['token_count'].item(), 2)
        torch.testing.assert_close(padded['sum_nll'], short['sum_nll'])
        torch.testing.assert_close(padded['mean_nll'], short['mean_nll'])

    def test_length_and_invalid_candidate_are_explicit(self):
        logits = torch.zeros(2, 3, 2)
        result = token_nll(logits, torch.tensor([[0, -100, -100], [0, 1, 0]]))
        self.assertEqual(result['token_count'].tolist(), [1, 3])
        torch.testing.assert_close(result['mean_nll'][0], result['mean_nll'][1])
        with self.assertRaisesRegex(ValueError, 'no scored tokens'):
            token_nll(logits[:1], torch.full((1, 3), -100))
