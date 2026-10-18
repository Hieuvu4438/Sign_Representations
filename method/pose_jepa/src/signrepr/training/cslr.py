import torch
from torch import nn
from ..evaluation.temporal import ctc_minimum_length


class CTCHead(nn.Module):
    def __init__(self, glosses, input_dim=1024, hidden_dim=256):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(2*hidden_dim, glosses+1)

    def forward(self, features, lengths):
        if torch.any(lengths <= 0) or torch.any(lengths > features.shape[1]):
            raise ValueError('CTC calendar lengths must be positive and within tensor bounds')
        x = self.project(features)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=features.shape[1])
        return self.classifier(x)


def ctc_loss(logits, targets, lengths):
    if len(targets) != len(lengths):
        raise ValueError('CTC target batch mismatch')
    for target, length in zip(targets, lengths):
        if any(int(token) <= 0 or int(token) >= logits.shape[-1] for token in target):
            raise ValueError('CTC training vocabulary uses positive indices; blank=0')
        if int(length) < ctc_minimum_length(target):
            raise ValueError('Infeasible CTC alignment including adjacent repeated glosses')
    flat = torch.tensor([int(token) for target in targets for token in target], dtype=torch.long, device=logits.device)
    target_lengths = torch.tensor(list(map(len, targets)), dtype=torch.long)
    return nn.CTCLoss(blank=0, zero_infinity=False)(logits.float().log_softmax(-1).transpose(0, 1), flat, lengths.cpu(), target_lengths)
