from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class EmbeddingOutput:
    dense: torch.Tensor
    timestamps: torch.Tensor
    valid: torch.Tensor
    clip: torch.Tensor
    sample_ids: tuple[str, ...]


def pool_raw_tokens(encoded, batch):
    b, t, streams = batch.valid.shape
    values = encoded.tokens.reshape(b, t, streams, -1)
    weights = batch.valid[..., None].to(values.dtype)
    dense = (values * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1)
    valid = batch.valid.any(dim=2)
    # Upsampled aliases do not multiply a source observation's pooling contribution.
    unique = torch.zeros_like(valid)
    for row in range(b):
        seen = set()
        for frame in range(int(batch.lengths[row])):
            source = int(batch.source_frame_index[row, frame])
            unique[row, frame] = bool(valid[row, frame]) and source not in seen
            if valid[row, frame]:
                seen.add(source)
    clip = (dense * unique[..., None]).sum(dim=1) / unique.sum(dim=1, keepdim=True).clamp_min(1)
    return EmbeddingOutput(dense, batch.timestamps, valid, clip, batch.sample_ids)
