from __future__ import annotations

import torch
from torch import nn


def pooled_valid(tokens, valid):
    weights = valid.to(tokens.dtype)
    if (weights.sum(1) == 0).any():
        raise ValueError("no valid visible RGB tokens")
    return (tokens*weights[..., None]).sum(1)/weights.sum(1)[:, None]


class TinyEncoderAdapter(nn.Module):
    """CPU fixture encoder. This is never a scientific Hiera baseline."""
    def __init__(self, input_frames=32, spatial_size=(32, 32), dim=32):
        super().__init__()
        self.input_frames, self.spatial_size, self.dim = input_frames, tuple(spatial_size), dim
        self.patch = nn.Conv3d(3, dim, kernel_size=(2, 8, 8), stride=(2, 8, 8))
        self.norm = nn.LayerNorm(dim)
        self.grid = (input_frames//2, spatial_size[0]//8, spatial_size[1]//8)
        self.num_mask_units = self.grid[0]*self.grid[1]*self.grid[2]

    def random_mask(self, batch, ratio, device, generator=None):
        n = self.num_mask_units
        k = int(n*(1-ratio))
        if not 0 < k <= n:
            raise ValueError("zero visible mask units")
        indices = torch.rand(batch, n, device=device,generator=generator).argsort(1)[:, :k]
        return torch.zeros(batch, n, dtype=torch.bool, device=device).scatter_(1, indices, True)

    def _tokens(self, rgb, frame_valid, times):
        if rgb.shape[2:] != (self.input_frames, *self.spatial_size):
            raise ValueError("RGB input geometry differs from adapter")
        grid = self.patch(rgb).permute(0, 2, 3, 4, 1)
        grid = self.norm(grid)
        t, h, w = self.grid
        tt, yy, xx = torch.meshgrid(torch.arange(t, device=rgb.device), torch.arange(h, device=rgb.device),
                                    torch.arange(w, device=rgb.device), indexing="ij")
        coord = torch.stack((tt/(max(t-1, 1)), yy/max(h-1, 1), xx/max(w-1, 1)), -1).reshape(-1, 3)
        # Receptive cells use the later center index for the even patch kernel.
        centers = torch.arange(t, device=rgb.device)*2+1
        valid = frame_valid[:, centers, None, None].expand(-1, -1, h, w).reshape(len(rgb), -1)
        return grid, coord.expand(len(rgb), -1, -1), valid, times[:, centers]

    def forward_visible(self, rgb, keep_mask, frame_valid, times):
        if keep_mask.shape != (len(rgb), self.num_mask_units) or keep_mask.dtype != torch.bool:
            raise ValueError("keep mask shape/semantics mismatch")
        if not (keep_mask.sum(1) == keep_mask.sum(1)[0]).all() or not keep_mask.any(1).all():
            raise ValueError("equal nonzero visible count required")
        # Tiny patches do not overlap, so pixel masking cannot leak across units.
        grid, coords, valid, _ = self._tokens(rgb, frame_valid, times)
        k = int(keep_mask.sum(1)[0])
        tok = grid.reshape(len(rgb), -1, self.dim)[keep_mask].reshape(len(rgb), k, self.dim)
        coords = coords[keep_mask].reshape(len(rgb), k, 3)
        valid = valid[keep_mask].reshape(len(rgb), k)
        return tok, coords, valid, pooled_valid(tok, valid)

    def encode_dense(self, rgb, frame_valid, times):
        grid, _, _, h_times = self._tokens(rgb, frame_valid, times)
        h = grid.mean((2, 3))
        centers = torch.arange(self.grid[0], device=rgb.device)*2+1
        valid = frame_valid[:, centers]
        return h, h_times, valid, pooled_valid(h, valid)
