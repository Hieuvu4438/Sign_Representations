from torch import nn


class RegularizationProjector(nn.Sequential):
    def __init__(self, encoder_dim, hidden=768, out_dim=256):
        super().__init__(nn.Linear(encoder_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
