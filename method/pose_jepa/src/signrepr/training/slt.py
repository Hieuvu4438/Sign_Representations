"""Frozen pose feature bridge to a task-specific mT5 encoder/decoder."""
import torch
from torch import nn


class SLTBridge(nn.Module):
    def __init__(self, decoder, input_dim=1024):
        super().__init__()
        self.decoder = decoder
        self.bridge = nn.Sequential(nn.Linear(input_dim, decoder.config.d_model), nn.LayerNorm(decoder.config.d_model))

    def inputs(self, features, lengths):
        if torch.any(lengths <= 0) or torch.any(lengths > features.shape[1]):
            raise ValueError('Invalid SLT calendar lengths')
        mask = torch.arange(features.shape[1], device=features.device)[None] < lengths[:, None]
        return {'inputs_embeds': self.bridge(features), 'attention_mask': mask.long()}

    def forward(self, features, lengths, labels):
        if labels.dtype != torch.long:
            raise ValueError('SLT labels must be int64 with padding=-100')
        return self.decoder(**self.inputs(features, lengths), labels=labels)

    @torch.inference_mode()
    def generate(self, features, lengths, **kwargs):
        return self.decoder.generate(**self.inputs(features, lengths), **kwargs)


def translation_labels(tokenizer, translations, max_tokens=None, device='cpu'):
    encoded = tokenizer(translations, padding=True, truncation=False, return_tensors='pt')
    if max_tokens is not None and encoded['attention_mask'].sum(-1).max() > max_tokens:
        raise ValueError('Translation exceeds audited token cap; increase it instead of silently truncating')
    labels = encoded['input_ids'].masked_fill(~encoded['attention_mask'].bool(), -100)
    return labels.to(device)


def load_mt5(model_name='google/mt5-base', revision=None, local_files_only=False):
    from transformers import AutoTokenizer, MT5ForConditionalGeneration
    decoder = MT5ForConditionalGeneration.from_pretrained(model_name, revision=revision, local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, local_files_only=local_files_only)
    return decoder, tokenizer
