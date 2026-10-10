import torch
from torch import nn


class QueryPredictor(nn.Module):
    """Continuous time/span/space queries plus learned edge/articulator identity."""
    def __init__(self, dim, output_dim, kinds, width=128, heads=4, depth=2):
        super().__init__()
        self.context = nn.Linear(dim,width)
        self.identity = nn.Embedding(kinds,width)
        self.coordinates = nn.Sequential(nn.Linear(4,width),nn.GELU(),nn.Linear(width,width))
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(width,heads,4*width,
                         dropout=0,batch_first=True,norm_first=True),depth)
        self.output = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,output_dim))

    def forward(self, context, padding, ids, coordinates):
        q = self.identity(ids)+self.coordinates(coordinates.to(context.dtype))
        return self.output(self.decoder(q,self.context(context),memory_key_padding_mask=padding))
