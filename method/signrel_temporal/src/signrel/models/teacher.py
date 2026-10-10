"""Original DINOv2 ViT-S/14, pinned source and strict public weight loading."""
from pathlib import Path
import subprocess
import torch
import torch.nn.functional as F
from ..utils import sha256


class FrozenTeacher:
    def __init__(self,lock,device='cpu'):
        record = lock['teacher']
        if record['name'] != 'dinov2_vits14' or record['feature_dim'] != 384:
            raise ValueError('Teacher identity mismatch')
        repo = Path(record['repo_dir']).resolve()
        commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
        if commit != record['commit_sha'] or sha256(record['checkpoint_path']) != record['checkpoint_sha256']:
            raise ValueError('Teacher source/weight fingerprint mismatch')
        if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip():
            raise ValueError('Pinned DINO source has tracked modifications')
        self.model = torch.hub.load(str(repo),'dinov2_vits14',source='local',pretrained=False)
        self.model.load_state_dict(torch.load(record['checkpoint_path'],map_location='cpu',weights_only=True),strict=True)
        self.model.eval().requires_grad_(False).to(device)
        self.device = device

    @torch.inference_mode()
    def __call__(self,crops):
        features = self.model.forward_features(crops.to(self.device))['x_norm_clstoken'].float()
        if features.shape != (len(crops),384) or not torch.isfinite(features).all():
            raise ValueError('Unexpected DINO output')
        return F.normalize(features,dim=-1).cpu()
