"""Stride-1 temporal CTC head with explicit impossible-alignment rejection."""
import torch
from torch import nn


def compact_ctc_inputs(dense, valid, timestamps):
    """Remove missing internal observations and retain their original time supports."""
    if dense.ndim != 3 or valid.shape != dense.shape[:2] or timestamps.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError("CTC compaction needs aligned dense/valid/timestamp arrays")
    lengths = valid.sum(dim=1)
    if (lengths == 0).any():
        raise ValueError("CTC compaction cannot create observations for an empty sequence")
    sequences = [dense[i, valid[i]] for i in range(len(dense))]
    times = [timestamps[i, valid[i]] for i in range(len(dense))]
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True), lengths, times


class CTCHead(nn.Module):
    def __init__(self, dimension, vocabulary_size, blank=0):
        super().__init__()
        if not 0 <= blank < vocabulary_size:
            raise ValueError("CTC blank must belong to the declared vocabulary")
        self.classifier = nn.Linear(dimension, vocabulary_size)
        self.blank, self.temporal_stride = blank, 1
        self.criterion = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=False)

    def forward(self, dense):
        return self.classifier(dense)

    def loss(self, dense, input_lengths, targets, target_lengths):
        if dense.ndim != 3 or targets.ndim != 2 or input_lengths.shape != (len(dense),) or target_lengths.shape != (len(dense),):
            raise ValueError("CTC requires dense[B,T,D], targets[B,U], and explicit lengths[B]")
        if ((input_lengths <= 0) | (input_lengths > dense.shape[1])).any() or ((target_lengths <= 0) | (target_lengths > targets.shape[1])).any():
            raise ValueError("Empty/out-of-range CTC sequences must be reported explicitly")
        for row in range(len(dense)):
            length = int(target_lengths[row])
            sequence = targets[row, :length]
            if ((sequence == self.blank) | (sequence < 0) | (sequence >= self.classifier.out_features)).any():
                raise ValueError("CTC targets contain blank or out-of-vocabulary labels")
            required = length + int((sequence[1:] == sequence[:-1]).sum())
            if int(input_lengths[row]) < required:
                raise ValueError("Impossible CTC alignment: repeated labels require blank positions")
        with torch.autocast(dense.device.type, enabled=False):
            logits = self(dense.float()).log_softmax(dim=-1).transpose(0, 1)
            return self.criterion(logits, targets.long(), input_lengths.long(), target_lengths.long())

    @torch.no_grad()
    def greedy_decode(self, dense, input_lengths):
        predictions = self(dense).argmax(dim=-1)
        result = []
        for row in range(len(predictions)):
            previous, decoded = None, []
            for index in predictions[row, :int(input_lengths[row])].tolist():
                if index != self.blank and index != previous:
                    decoded.append(index)
                previous = index
            result.append(decoded)
        return result
