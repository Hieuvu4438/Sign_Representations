import numpy as np
import pytest
import torch
from signpath.evaluation.ctc import CTCHead
from signpath.evaluation.metrics import classification_metrics, corpus_wer
from signpath.evaluation.slt import FixedDecoderSLT
from signpath.models.heads import AttentiveHead


@pytest.fixture(autouse=True)
def thread_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(12)
    yield
    torch.set_num_threads(previous)


def test_toy_ctc_gradient_repeated_labels_and_greedy():
    head = CTCHead(8, 5)
    dense = torch.randn(2, 7, 8, requires_grad=True)
    targets = torch.tensor([[1, 1, 2], [3, 2, 0]])
    loss = head.loss(dense, torch.tensor([7, 5]), targets, torch.tensor([3, 2]))
    loss.backward()
    assert torch.isfinite(loss) and dense.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="repeated"):
        head.loss(dense, torch.tensor([3, 5]), targets, torch.tensor([3, 2]))
    # Force token pattern [1,1,blank,1,2,2,blank].
    head.classifier = torch.nn.Identity()
    logits = torch.full((1, 7, 5), -10.)
    pattern = [1, 1, 0, 1, 2, 2, 0]
    logits[0, torch.arange(7), pattern] = 10
    assert head.greedy_decode(logits, torch.tensor([7])) == [[1, 1, 2]]


def test_toy_slt_gradients_causal_and_memory_padding():
    model = FixedDecoderSLT(8, 12, dimension=16, heads=4, layers=1).eval()
    dense = torch.randn(2, 7, 8, requires_grad=True)
    valid = torch.tensor([[True] * 7, [True] * 4 + [False] * 3])
    target = torch.tensor([[1, 4, 5, 2, 0], [1, 6, 7, 8, 2]])
    loss = model.loss(dense, valid, target)
    loss.backward()
    assert torch.isfinite(loss) and dense.grad.abs().sum() > 0
    first = model(dense.detach(), valid, target[:, :-1])
    changed_ids = target[:, :-1].clone()
    changed_ids[:, 2:] = 9
    changed_dense = dense.detach().clone()
    changed_dense[~valid] = 999
    other = model(changed_dense, valid, changed_ids)
    torch.testing.assert_close(first[:, :2], other[:, :2], atol=1e-6, rtol=1e-6)
    assert len(model.greedy_decode(dense.detach(), valid, max_tokens=3)) == 2


def test_attentive_probe_masks_missing_frames():
    head = AttentiveHead(8, 3, heads=2).eval()
    dense = torch.randn(2, 5, 8)
    valid = torch.tensor([[True] * 5, [True] * 3 + [False] * 2])
    changed = dense.clone()
    changed[~valid] = 999
    torch.testing.assert_close(head(dense, valid), head(changed, valid), atol=0, rtol=0)


def test_hand_calculated_metrics():
    scores = np.array([[9, 0, 0], [9, 0, 0], [9, 0, 0], [0, 9, 0]])
    result = classification_metrics(scores, [0, 0, 0, 0])
    assert result["per_instance_top1"] == result["per_class_top1"] == .75
    result = classification_metrics(scores, [0, 0, 1, 1])
    assert result["per_instance_top1"] == result["per_class_top1"] == .75
    assert result["absent_classes"] == [2]
    imbalanced = classification_metrics([[9, 0]] * 4, [0, 0, 0, 1])
    assert imbalanced["per_instance_top1"] == .75 and imbalanced["per_class_top1"] == .5
    wer = corpus_wer([["A", "B", "C"], ["X", "Y"]], [["A", "D"], ["X", "Z", "Y"]])
    assert wer == {"substitutions": 1, "deletions": 1, "insertions": 1, "reference_tokens": 5, "wer": .6}


def test_ctc_compaction_keeps_real_timestamp_supports():
    from signpath.evaluation.ctc import compact_ctc_inputs
    dense = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2).requires_grad_(True)
    valid = torch.tensor([[True, False, True, False, True]])
    times = torch.tensor([[0., .04, .08, .12, .16]], dtype=torch.float64)
    output, lengths, supports = compact_ctc_inputs(dense, valid, times)
    torch.testing.assert_close(output, dense[:, [0, 2, 4]])
    torch.testing.assert_close(supports[0], times[0, [0, 2, 4]])
    assert lengths.tolist() == [3]
    output.sum().backward()
    assert not dense.grad[:, [1, 3]].any()


def test_slt_rejects_internal_sequence_delimiters():
    head = FixedDecoderSLT(8, 12, dimension=16, heads=4, layers=1)
    with pytest.raises(ValueError, match="delimit"):
        head.loss(torch.ones(1, 5, 8), torch.ones(1, 5, dtype=torch.bool), torch.tensor([[1, 4, 2, 7, 2]]))
