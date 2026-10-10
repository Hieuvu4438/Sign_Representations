"""Compare the external reference against the actual pinned official wrapper."""
import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from signcoord.models.signrep_reference import SignRepReference
from signcoord.provenance import atomic_json,execution_stamp,sha256_file

torch.set_num_threads(2)
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
repo=PROJECT.parents[1]; checkpoint=repo/'checkpoints/signrep/ckpt.pt'; vendor=repo/'third_party/SignRep'
adapter=SignRepReference(vendor,checkpoint).eval()
Official=importlib.import_module('models.final_models.FINAL_hiera_latent_model_head_v25_active').Model
# Full published prior head, including the official dimensions. No training
# targets are reconstructed here; it is needed for strict full-wrapper loading.
outputs={'full_body_angles':44,'full_right_angles':82,'full_left_angles':82,
         'body_pose':183,'right_pose':63,'left_pose':63,'body_dist':792,
         'right_dist':165,'left_dist':165}
official=Official(backbone_params={'drop_path_rate':.1,'num_cls_tokens':0},ckpt_dir=None,
                  mask_ratio=0.,out_dim=768,init_weights=False,
                  cls_head_name='models.head_models.dict_head_v6_FINAL_thoughtful',
                  cls_head_params={'latents_params':{'hidden_dim':512,'out_dim':384},'total_frames':16,
                                   'projection_params':{k:{'hidden_dim':None,'output_dim':v} for k,v in outputs.items()}}).eval()
official.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'],strict=True)
Transformation=importlib.import_module('augmentation.video.base_video_aug').Transformation
rgb=np.random.default_rng(7).integers(0,256,size=(16,72,120,3),dtype=np.uint8)
frames=Transformation().aug_video(list(rgb)).permute(1,0,2,3)[None]
with torch.inference_mode():
    actual=official(frames); post,latent=adapter(frames)
post_error=(actual['features']-post).abs().max().item()
latent_error=(actual['latent']-latent).abs().max().item()
assert post_error<2e-5 and latent_error<2e-5,(post_error,latent_error)
atomic_json(args.output,{**execution_stamp(),'status':'passed','device':'cpu',
                        'checkpoint_sha256':sha256_file(checkpoint),'official_wrapper_strict_load':True,
                        'official_transform_used':True,'post_fc_max_error':post_error,
                        'latent_max_error':latent_error,'feature_shape':list(post.shape),
                        'scope':'official T16 per-window readout parity; downstream video pooling is a disclosed extension'})
print(args.output)
