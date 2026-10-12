"""Tensor-only unmasked inference, equivalent to packed eval and exportable at dynamic T."""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F

from ..models.encoder import time_encoding
from ..models.aliases import source_aliases


def explicit_attention(attention, query, memory, padding):
    """Eval-only MHA algebra using the trained weights, without export kernel guards."""
    d, heads = attention.embed_dim, attention.num_heads
    bias = attention.in_proj_bias
    if query is memory:
        q, k, v = F.linear(query, attention.in_proj_weight, bias).chunk(3, dim=-1)
    else:
        weight = attention.in_proj_weight
        q = F.linear(query, weight[:d], bias[:d] if bias is not None else None)
        k = F.linear(memory, weight[d:2 * d], bias[d:2 * d] if bias is not None else None)
        v = F.linear(memory, weight[2 * d:], bias[2 * d:] if bias is not None else None)
    b, nq, nk = query.shape[0], query.shape[1], memory.shape[1]
    q = q.reshape(b, nq, heads, d // heads).transpose(1, 2)
    k = k.reshape(b, nk, heads, d // heads).transpose(1, 2)
    v = v.reshape(b, nk, heads, d // heads).transpose(1, 2)
    scores = ((q * (d // heads) ** -.5) @ k.transpose(-1, -2)).masked_fill(padding[:, None, None, :], -torch.inf)
    attended = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, nq, d)
    return F.linear(attended, attention.out_proj.weight, attention.out_proj.bias)


def explicit_encoder_block(block, x, padding):
    if block.norm_first:
        normalized = block.norm1(x)
        x = x + explicit_attention(block.self_attn, normalized, normalized, padding)
        return x + block.linear2(block.activation(block.linear1(block.norm2(x))))
    x = block.norm1(x + explicit_attention(block.self_attn, x, x, padding))
    return block.norm2(x + block.linear2(block.activation(block.linear1(x))))


class OfflineEncoderGraph(nn.Module):
    """Teacher/predictor are deliberately absent from this deployment state."""
    def __init__(self, online=None, head=None, *, head_kind="linear", l2_normalize=True, frozen_visual=False):
        super().__init__()
        self.frozen_visual = frozen_visual
        self.head_kind, self.l2_normalize = head_kind, l2_normalize
        self.head = deepcopy(head) if head is not None else None
        if not frozen_visual:
            if online is None:
                raise ValueError("Temporal deployment needs an explicit trained online encoder")
            self.projectors = deepcopy(online.projectors)
            self.type_embeddings = deepcopy(online.type_embeddings)
            self.blocks, self.norm = deepcopy(online.blocks), deepcopy(online.norm)
            self.dimension = online.dimension
        self.eval().requires_grad_(False)

    def forward(self, streams, valid, timestamps, source_frame_index):
        b, t = valid.shape[:2]
        alias_first, unique = source_aliases(source_frame_index, valid.any(dim=2))
        if self.frozen_visual:
            values = torch.stack(streams[:4], dim=2)
            weights = valid[:, :, :4]
            values = torch.where(weights[..., None], values, 0)
        else:
            origin_index = (source_frame_index >= 0).to(torch.int64).argmax(dim=1, keepdim=True)
            origin = timestamps.gather(1, origin_index)
            relative_seconds = (timestamps - origin).float()
            time = time_encoding(relative_seconds, self.dimension)
            projected = []
            for stream, projector in enumerate(self.projectors):
                values = torch.where(valid[:, :, stream, None], streams[stream], 0)
                value = projector(values) + time
                if self.type_embeddings is not None:
                    value = value + self.type_embeddings.weight[stream]
                projected.append(value)
            x = torch.stack(projected, dim=2).flatten(1, 2)
            real = (valid & unique[..., None]).flatten(1)
            # An all-invalid sample gets one finite sentinel key; no sentinel is pooled.
            first = torch.arange(t * 7, device=valid.device)[None] == 0
            allowed = real | ((~real.any(dim=1))[:, None] & first)
            x = torch.where(real[..., None], x, 0)
            for block in self.blocks:
                x = explicit_encoder_block(block, x, ~allowed)
            x = torch.where(real[..., None], self.norm(x), 0)
            values = x.reshape(b, t, 7, self.dimension)
            values = values.gather(1, alias_first[..., None, None].expand_as(values))
            values, weights = torch.where(valid[..., None], values, 0), valid
        dense = (values * weights[..., None]).sum(dim=2) / weights.sum(dim=2, keepdim=True).clamp_min(1)
        frame_valid = weights.any(dim=2) & (source_frame_index >= 0)
        earlier = torch.tril(torch.ones(t, t, dtype=torch.bool, device=valid.device), diagonal=-1)
        same = source_frame_index[:, :, None] == source_frame_index[:, None, :]
        duplicates = (same & earlier[None] & frame_valid[:, None, :]).any(dim=2)
        unique = frame_valid & ~duplicates
        clip = (dense * unique[..., None]).sum(dim=1) / unique.sum(dim=1, keepdim=True).clamp_min(1)
        if self.head is None:
            logits = clip[:, :0]
        elif self.head_kind == "attentive":
            memory = F.normalize(dense, dim=-1) if self.l2_normalize else dense
            memory = torch.where(frame_valid[..., None], memory, 0)
            first_frame = torch.arange(t, device=valid.device)[None] == 0
            allowed_frames = frame_valid | ((~frame_valid.any(dim=1))[:, None] & first_frame)
            pooled = explicit_attention(self.head.attention, self.head.query.expand(b, -1, -1), memory, ~allowed_frames)
            logits = self.head.classifier(pooled[:, 0])
        else:
            logits = self.head(F.normalize(clip, dim=-1) if self.l2_normalize else clip)
        logits = torch.where(frame_valid.any(dim=1, keepdim=True), logits, 0)
        return dense, timestamps, frame_valid, clip, logits


def graph_inputs(batch):
    return batch.streams, batch.valid, batch.timestamps, batch.source_frame_index


def validate_inputs(inputs, raw_dimensions):
    streams, valid, timestamps, sources = inputs
    if valid.ndim != 3 or valid.shape[2] != 7 or valid.dtype != torch.bool or valid.shape[1] < 1:
        raise ValueError("Deployment validity requires bool[B,T>=1,7]")
    b, t = valid.shape[:2]
    if len(streams) != 7 or any(x.shape != (b, t, d) or x.dtype != torch.float32 or not torch.isfinite(x).all() for x, d in zip(streams, raw_dimensions)):
        raise ValueError("Deployment raw stream dimensions/dtype/finite values mismatch")
    if timestamps.shape != (b, t) or timestamps.dtype != torch.float64 or not torch.isfinite(timestamps).all():
        raise ValueError("Deployment timestamps require finite float64[B,T] seconds")
    if sources.shape != (b, t) or sources.dtype != torch.int64 or ((sources < 0) & valid.any(dim=2)).any():
        raise ValueError("Negative source IDs are only allowed in invalid/padded observations")
    for row in range(b):
        times = timestamps[row, sources[row] >= 0]
        if (times.diff() < 0).any():
            raise ValueError("Deployment observations must be timestamp ordered")
