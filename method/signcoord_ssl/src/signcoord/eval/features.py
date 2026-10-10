"""Frozen full-interval RGB extraction with duplicate timestamp coverage pooling."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from signcoord.data.manifest import read_manifest
from signcoord.data.sampling import nearest_frame_plan
from signcoord.data.transforms import letterbox
from signcoord.data.video_io import decode_frames,video_metadata
from signcoord.data.video_index import record_metadata
from signcoord.export.inference import encode_rgb,encoder_from_config
from signcoord.engine.checkpoint import load_checkpoint
from signcoord.provenance import atomic_json,execution_stamp,object_hash,sha256_file,tensor_state_hash


def extract_features(checkpoint,manifest,output,device="cpu",include_sequence=True):
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('feature output exists; immutable cache requires new path')
    saved=load_checkpoint(checkpoint); c=saved["resolved_config"]
    encoder=encoder_from_config(c)
    encoder.load_state_dict({k.removeprefix("encoder."):v for k,v in saved["model_state"].items() if k.startswith("encoder.")},strict=True)
    encoder.to(device).eval().requires_grad_(False)
    records=read_manifest(manifest)
    sampler=c["data"]["sampler"]; fps=sampler["grid_fps"]; frames=sampler["frames"]
    embeddings=[];raw_embeddings=[]; sequences=[]; sequence_times=[]; offsets=[0]
    with torch.inference_mode():
        for r in records:
            pts,_,_=record_metadata(r)
            begin,end=r["start_ms"]/1000,r["end_ms"]/1000
            n=max(1,int(np.ceil((end-begin)*fps-1e-9)))
            # Deterministic half-context stride, with a padded tail; timestamps
            # shared by overlapping windows are averaged once before pooling.
            starts=list(range(0,max(1,n-frames+1),frames//2))
            last=2*int(max(0,n-frames+1)//2)
            if last>starts[-1]: starts.append(last)
            by_time=defaultdict(list)
            for start in starts:
                plan=nearest_frame_plan(pts,begin+start/fps,end,frames,fps,allowed_start_s=begin)
                rgb=torch.from_numpy(decode_frames(r["media_path"],plan["frame_index"])).permute(0,3,1,2).float()/255
                rgb,_,_,_=letterbox(rgb,output_size=sampler["spatial_size"])
                mean=rgb.new_tensor([.485,.456,.406])[None,:,None,None]; std=rgb.new_tensor([.229,.224,.225])[None,:,None,None]
                rgb=((rgb-mean)/std).permute(1,0,2,3)[None].to(device)
                fv=torch.from_numpy(plan["frame_valid"])[None].to(device)
                times=torch.from_numpy(plan["requested_timestamp_s"])[None].to(device)
                out=encode_rgb(encoder,rgb,fv,times)
                for value,t,v in zip(out["sequence"][0].cpu(),out["sequence_times"][0].cpu(),out["sequence_valid"][0].cpu()):
                    if v: by_time[round(float(t),8)].append(value)
            ordered=sorted(by_time)
            h=torch.stack([torch.stack(by_time[t]).mean(0) for t in ordered])
            z=h.mean(0)
            if z.norm()<=1e-12: raise ValueError("zero-norm full-clip embedding")
            embeddings.append(torch.nn.functional.normalize(z,dim=0).numpy())
            raw_embeddings.append(z.numpy())
            if include_sequence:sequences.append(h.numpy())
            sequence_times.extend(ordered); offsets.append(offsets[-1]+len(h))
    arrays=dict(z=np.stack(embeddings),z_raw=np.stack(raw_embeddings),sequence_times=np.asarray(sequence_times),
                        offsets=np.asarray(offsets),sample_ids=np.asarray([r["sample_id"] for r in records]),
                        source_groups=np.asarray([r["source_group_id"] for r in records]),
                        groups=np.asarray([r["signer_id"] or r["source_group_id"] for r in records]))
    if include_sequence:arrays['sequence']=np.concatenate(sequences)
    np.savez_compressed(path,**arrays)
    pooling={"policy":"mean_overlaps_at_each_unique_native_timestamp_then_uniform_pool_v1","window_stride":frames//2,
             "tail_start_policy":"round_down_even_last_start_with_padding","rgb_only":True}
    metadata={**execution_stamp(),"status":"passed","checkpoint_sha256":sha256_file(checkpoint),
              'encoder_state_hash':tensor_state_hash(encoder.state_dict()),
              "manifest_sha256":sha256_file(manifest),"sampler_hash":object_hash(sampler),"pooling_hash":object_hash(pooling),
              "pooling_protocol":pooling,"sample_count":len(records),"output_sha256":sha256_file(path),
              "labels_read_by_encoder":False,"feature_layer":"final_pre_classifier",
              'sequence_features_stored':include_sequence,'representation_dimension':embeddings[0].shape[0]}
    atomic_json(str(path)+".json",metadata)
    return metadata
