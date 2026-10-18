import pytest
import torch
from signrepr.training.cslr import CTCHead, ctc_loss
from signrepr.evaluation.temporal import ctc_decode, ctc_minimum_length, word_error_rate


def test_ctc_repeats_blank_and_infeasible_sequences():
    assert ctc_decode([1, 1, 0, 1, 2, 2, 0]) == [1, 1, 2]
    assert ctc_minimum_length([1, 1, 2]) == 4
    head = CTCHead(3, input_dim=16, hidden_dim=8)
    features = torch.randn(2, 5, 16)
    logits = head(features, torch.tensor([5, 3]))
    loss = ctc_loss(logits, [[1, 1, 2], [2, 3]], torch.tensor([5, 3]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    with pytest.raises(ValueError, match='Infeasible'):
        ctc_loss(logits, [[1, 1, 2], [2, 3]], torch.tensor([3, 3]))
    changed = features.clone(); changed[1, 3:] = 1000
    torch.testing.assert_close(head(features, torch.tensor([5, 3]))[1, :3], head(changed, torch.tensor([5, 3]))[1, :3])
    assert word_error_rate([[1, 2, 3]], [[1, 4]]) == {'WER': 2/3, 'substitutions': 1, 'deletions': 1, 'insertions': 0, 'reference_tokens': 3, 'num_utterances': 1}
