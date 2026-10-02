"""Frozen lexical controls and shallow temporal readouts; no encoder adaptation."""
import numpy as np
import torch
from torch import nn


def normalized_mean(embeddings, valid):
    values = embeddings[valid].mean(axis=0)
    return values / max(float(np.linalg.norm(values)), 1e-12)


def classification_metrics(gold, predicted, classes):
    gold, predicted = np.asarray(gold), np.asarray(predicted)
    if not len(gold):
        raise ValueError('No evaluation samples')
    recalls, supports = [], []
    for label in range(classes):
        selected = gold == label
        supports.append(int(selected.sum()))
        if selected.any():
            recalls.append(float((predicted[selected] == label).mean()))
    return {'top1': float((gold == predicted).mean()), 'macro_recall': float(np.mean(recalls)),
            'denominator': len(gold), 'classes_with_support': len(recalls), 'class_support': supports}


class LinearProbe(nn.Module):
    def __init__(self, dimension, classes):
        super().__init__()
        self.classifier = nn.Linear(dimension, classes)

    def forward(self, tokens, mask):
        pooled = (tokens * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.classifier(nn.functional.normalize(pooled, dim=-1))


class TemporalProbe(nn.Module):
    def __init__(self, dimension, classes, hidden):
        super().__init__()
        self.project = nn.Conv1d(dimension, hidden, 1)
        self.temporal = nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden)
        linear_budget = dimension * classes + classes
        base_parameters = hidden * (dimension + classes + 5) + classes
        intermediate = max(1, (linear_budget - base_parameters - hidden) // (2 * hidden + 1))
        self.channel = nn.Sequential(nn.Conv1d(hidden, intermediate, 1), nn.GELU(), nn.Conv1d(intermediate, hidden, 1))
        self.parameter_matching_rule = 'Channel residual width chosen from parameter budget of dimension-to-class linear head, before labels or metrics'
        self.classifier = nn.Linear(hidden, classes)

    def forward(self, tokens, mask):
        valid = mask.unsqueeze(1)
        hidden = nn.functional.gelu(self.project(tokens.transpose(1, 2))) * valid
        hidden = nn.functional.gelu(self.temporal(hidden)) * valid
        hidden = nn.functional.gelu(hidden + self.channel(hidden)) * valid
        pooled = hidden.sum(dim=2) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.classifier(pooled)


def collate(samples, indices, device):
    length = max(len(samples[index]['tokens']) for index in indices)
    dimension = samples[indices[0]]['tokens'].shape[1]
    data = np.zeros((len(indices), length, dimension), dtype=np.float32)
    mask = np.zeros((len(indices), length), dtype=np.float32)
    labels = []
    for offset, index in enumerate(indices):
        tokens = samples[index]['tokens']
        data[offset, :len(tokens)] = tokens
        mask[offset, :len(tokens)] = 1
        labels.append(samples[index]['label'])
    return torch.from_numpy(data).to(device), torch.from_numpy(mask).to(device), torch.tensor(labels, device=device)


def evaluate(model, samples, device, batch_size=64):
    scores = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            tokens, mask, _ = collate(samples, list(range(start, min(start + batch_size, len(samples)))), device)
            scores.append(model(tokens, mask).softmax(dim=-1).cpu().numpy())
    return np.concatenate(scores)
