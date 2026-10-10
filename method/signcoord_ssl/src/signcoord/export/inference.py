"""Native encoder-only deployment: no pose/cache/decoder/projector imports."""
import torch
from pathlib import Path


def encoder_from_config(c, initialize=False):
    m=c["model"]
    if m["backbone"] == "tiny_fixture":
        from signcoord.models.backbone_adapter import TinyEncoderAdapter
        return TinyEncoderAdapter(m["input_frames"],c["data"]["sampler"]["spatial_size"],m["tiny_dim"])
    if m["backbone"] == "signrep_hiera_adapter":
        from signcoord.models.hiera_adapter import HieraAdapter
        return HieraAdapter(m["vendor_root"],m["input_frames"],
                            m.get("initialization_checkpoint") if initialize else None,
                            m["initialization_mode"] if initialize else "random_init",m.get("attention_kernel","vendor_5d"))
    if m["backbone"] == "hiera_base_video_adapter":
        from signcoord.models.generic_hiera_adapter import GenericHieraAdapter
        return GenericHieraAdapter(m["generic_vendor_root"],m["input_frames"],
                            m.get("initialization_checkpoint") if initialize else None,
                            m["initialization_mode"] if initialize else "random_init",m.get("attention_kernel","vendor_5d"))
    raise ValueError("unsupported exported encoder architecture")


def encode_rgb(encoder,rgb,frame_valid,times):
    h,ht,hv,z=encoder.encode_dense(rgb,frame_valid,times)
    if (z.norm(dim=-1)<=1e-12).any(): raise ValueError("zero-norm representation")
    return {"sequence":h,"sequence_times":ht,"sequence_valid":hv,"clip_raw":z,
            "clip":torch.nn.functional.normalize(z,dim=-1)}


def load_bundle(path,device="cpu"):
    path=Path(path).resolve()
    bundle=torch.load(path,map_location="cpu",weights_only=True)
    if bundle.get("schema_version") != "signcoord_encoder_bundle_v1": raise ValueError("invalid bundle schema")
    config=bundle['encoder_config']
    if bundle['metadata'].get('external_vendor_path_required') is False:
        from signcoord.provenance import sha256_file
        for relative,expected in bundle['metadata']['deployment_source_hashes'].items():
            file=(path.parent/relative).resolve()
            if path.parent not in file.parents or sha256_file(file)!=expected:
                raise ValueError(f'deployment source hash mismatch: {relative}')
        for key in ('vendor_root','generic_vendor_root'):
            if key in config['model']:
                relative=Path(config['model'][key])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('portable vendor path must remain inside the bundle')
                config['model'][key]=str(path.parent/relative)
    encoder=encoder_from_config(config)
    encoder.load_state_dict(bundle["encoder_state"],strict=True)
    return encoder.to(device).eval(),bundle["metadata"]
