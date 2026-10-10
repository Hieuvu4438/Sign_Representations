"""Read-only external SignRep baseline; no upstream functions copied or patched."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from signrel.data.manifest import sha256


class SignRepBaseline(nn.Module):
    def __init__(self,repo,checkpoint):
        super().__init__()
        repo = Path(repo).resolve()
        sys.path.insert(0,str(repo))
        from models.final_models.FINAL_hiera_latent_model_head_v25_active import Model
        import models.final_models.FINAL_hiera_latent_model_head_v25_active as source
        if repo not in Path(source.__file__).resolve().parents:
            raise AssertionError("SignRep namespace collision")
        # Constructor arguments verified against official example_usage.py.
        sizes = {"full_body_angles":44,"full_right_angles":82,"full_left_angles":82,
                 "body_pose":183,"right_pose":63,"left_pose":63,
                 "body_dist":792,"right_dist":165,"left_dist":165}
        self.model = Model(backbone_params={"drop_path_rate":0.1,"num_cls_tokens":0},ckpt_dir=None,
                           mask_ratio=0.,out_dim=768,init_weights=False,
                           cls_head_name="models.head_models.dict_head_v6_FINAL_thoughtful",
                           cls_head_params={"latents_params":{"hidden_dim":512,"out_dim":384},"total_frames":16,
                                            "projection_params":{k:{"hidden_dim":None,"output_dim":v} for k,v in sizes.items()}})
        if isinstance(self.model.cls_head,nn.Identity):
            raise AssertionError("Upstream constructor skipped prior head")
        state = torch.load(checkpoint,map_location="cpu",weights_only=True)
        result = self.model.load_state_dict(state["model"],strict=True)
        self.dim = 768
        self.upstream_init = {"name":"SignRep","path":str(Path(checkpoint).resolve()),"sha256":sha256(checkpoint),
                              "missing_keys":result.missing_keys,"unexpected_keys":result.unexpected_keys,
                              "license":"CC BY-NC-SA 4.0","inference":"official features output; mean across time if present"}

    def forward_tokens(self,video):
        if video.shape[2:]!=(16,224,224):
            raise AssertionError("Released SignRep requires 16x224x224 RGB")
        z = self.model(video)["features"]
        if z.ndim==2:
            z = z[:,None]
        if z.ndim!=3 or z.shape[-1]!=self.dim:
            raise AssertionError("Unexpected SignRep features shape")
        return {"tokens":z}
