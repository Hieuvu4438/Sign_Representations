"""A small functioning fixed-decoder comparison interface; no paired SSL implied."""
import torch
from torch import nn
from torch.nn import functional as F
from ..models.encoder import time_encoding


class FixedDecoderSLT(nn.Module):
    def __init__(self, encoder_dimension, vocabulary_size, *, dimension=48, heads=4, layers=2,
                 pad_id=0, bos_id=1, eos_id=2, dropout=0.0):
        super().__init__()
        if len({pad_id, bos_id, eos_id}) != 3 or min(pad_id, bos_id, eos_id) < 0 or max(pad_id, bos_id, eos_id) >= vocabulary_size:
            raise ValueError("SLT pad/BOS/EOS must be distinct declared vocabulary IDs")
        self.dimension, self.pad, self.bos, self.eos = dimension, pad_id, bos_id, eos_id
        self.memory_projector = nn.Linear(encoder_dimension, dimension)
        self.embedding = nn.Embedding(vocabulary_size, dimension, padding_idx=pad_id)
        self.blocks = nn.ModuleList(nn.TransformerDecoderLayer(dimension, heads, dimension * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True) for _ in range(layers))
        self.classifier = nn.Linear(dimension, vocabulary_size)

    def forward(self, dense, valid, input_ids):
        if not valid.any(dim=1).all() or not (input_ids != self.pad).any(dim=1).all():
            raise ValueError("SLT needs nonempty visual memory and decoder context")
        n = input_ids.shape[1]
        positions = torch.arange(n, device=input_ids.device, dtype=torch.float32)
        x = self.embedding(input_ids) + time_encoding(positions, self.dimension)
        memory = self.memory_projector(torch.where(valid[..., None], dense, 0))
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=input_ids.device), diagonal=1)
        for block in self.blocks:
            x = block(x, memory, tgt_mask=causal, tgt_key_padding_mask=input_ids == self.pad,
                      memory_key_padding_mask=~valid)
        return self.classifier(x)

    def loss(self, dense, valid, target_ids):
        if target_ids.ndim != 2 or target_ids.shape[1] < 2 or not (target_ids[:, 0] == self.bos).all():
            raise ValueError("Teacher-forced SLT targets require BOS and at least one output")
        for row in target_ids:
            nonpad = row[row != self.pad]
            if len(nonpad) < 2 or nonpad[-1] != self.eos:
                raise ValueError("SLT target must end with EOS before padding")
            if (nonpad[1:] == self.bos).any() or (nonpad[:-1] == self.eos).any():
                raise ValueError("BOS/EOS may only delimit a single target sequence")
            length = len(nonpad)
            if (row[:length] == self.pad).any() or (row[length:] != self.pad).any():
                raise ValueError("SLT padding must be a suffix")
        logits = self(dense, valid, target_ids[:, :-1])
        return F.cross_entropy(logits.float().transpose(1, 2), target_ids[:, 1:], ignore_index=self.pad)

    @torch.no_grad()
    def greedy_decode(self, dense, valid, max_tokens=64):
        if max_tokens < 1:
            raise ValueError("SLT decode budget must be positive")
        ids = torch.full((len(dense), 1), self.bos, dtype=torch.long, device=dense.device)
        finished = torch.zeros(len(dense), dtype=torch.bool, device=dense.device)
        for _ in range(max_tokens):
            logits = self(dense, valid, ids)[:, -1]
            logits[:, [self.pad, self.bos]] = -torch.inf
            next_id = logits.argmax(dim=-1)
            next_id = torch.where(finished, self.eos, next_id)
            ids = torch.cat([ids, next_id[:, None]], dim=1)
            finished |= next_id == self.eos
            if finished.all():
                break
        decoded = []
        for row in ids[:, 1:].tolist():
            if self.eos in row:
                row = row[:row.index(self.eos)]
            decoded.append(row)
        return decoded
