from torch import nn
from ..models.pose_encoder import PoseEncoder
from ..models.predictor import Predictor
from ..models.heads import PhysicalHeads
from ..models.part_tokenizer import token_metadata


class SSLSystem(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = PoseEncoder(config)
        self.predictor = Predictor(config)
        self.geometry = PhysicalHeads(config["predictor_dim"])
        self.motion = PhysicalHeads(config["predictor_dim"])

    def forward(self, batch, masks):
        context = self.encoder.encode_visible(batch, masks["visible"])
        prediction = self.predictor(context, token_metadata(batch), masks["target"])
        prediction["geometry"] = self.geometry(prediction["hidden"])
        prediction["motion"] = self.motion(prediction["hidden"])
        return prediction
