import copy
from pathlib import Path
import torch
import torch.nn.functional as F
from signrel.models.backbone_adapter import make_backbone


def load_encoder(checkpoint=None,backbone_config=None,device="cpu"):
    if checkpoint:
        state = torch.load(checkpoint,map_location="cpu",weights_only=False)
        backbone_config = dict(state["config"]["model"]["backbone"])
        # A trained checkpoint already contains encoder weights; don't reload upstream initialization.
        backbone_config.pop("checkpoint",None)
        encoder = make_backbone(backbone_config)
        encoder.load_state_dict(state["student_encoder"],strict=True)
        if "upstream_init" not in state:
            raise ValueError("Trained checkpoint lacks upstream initialization provenance")
        # Constructor initialization is not the origin of restored weights.
        encoder.upstream_init = copy.deepcopy(state["upstream_init"])
        from signrel.data.manifest import sha256
        encoder.restored_checkpoint = {"path":str(Path(checkpoint).resolve()),"sha256":sha256(checkpoint),
                                       "global_step":state.get("global_step"),"source_hash":state.get("source_hash"),
                                       "component":"student_encoder","strict_load":True}
    else:
        encoder = make_backbone(backbone_config)
    return encoder.to(device).eval()


@torch.no_grad()
def rgb_embedding(encoder,video):
    tokens = encoder.forward_tokens(video)["tokens"]
    feature = F.layer_norm(tokens.mean(1),(tokens.shape[-1],))
    return F.normalize(feature,dim=-1)
