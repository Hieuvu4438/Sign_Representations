from types import SimpleNamespace
import torch
from torch import nn
import pytest
from signrepr.training.slt import SLTBridge, translation_labels


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(d_model=12)
        self.output = nn.Linear(12, 8)

    def forward(self, inputs_embeds, attention_mask, labels):
        assert labels[1, -1] == -100
        assert attention_mask.tolist() == [[1]*5, [1, 1, 1, 0, 0]]
        pooled = (inputs_embeds*attention_mask[..., None]).sum(1)/attention_mask.sum(1)[:, None]
        logits = self.output(pooled)[:, None].expand(-1, labels.shape[1], -1)
        return SimpleNamespace(loss=nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100))


def tokenizer(text, **kwargs):
    assert kwargs['truncation'] is False
    return {'input_ids': torch.tensor([[1, 2, 3], [1, 4, 0]]), 'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 0]])}


def test_slt_bridge_dimension_calendar_mask_label_padding():
    bridge = SLTBridge(TinyDecoder(), input_dim=64)
    labels = translation_labels(tokenizer, ['a', 'b'])
    assert labels[1, -1] == -100
    features = torch.randn(2, 5, 64)
    loss = bridge(features, torch.tensor([5, 3]), labels).loss
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in bridge.parameters())
    with pytest.raises(ValueError, match='cap'):
        translation_labels(tokenizer, ['a', 'b'], max_tokens=2)
