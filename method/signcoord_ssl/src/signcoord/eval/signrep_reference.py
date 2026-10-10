"""Official native-frame T16/stride2 extraction, with disclosed video pooling."""
from pathlib import Path

import numpy as np
import torch

from signcoord.data.manifest import read_manifest
from signcoord.data.video_io import video_metadata,decode_frames
from signcoord.data.video_index import record_metadata
from signcoord.models.signrep_reference import SignRepReference
from signcoord.provenance import atomic_json, execution_stamp, object_hash, sha256_file


def extract_official_reference(c,manifest,output,device='cpu'):
    if c['arm'] != 'A1_ref': raise ValueError('official reference extraction requires A1_ref')
    path=Path(output)
    if path.exists(): raise ValueError('feature output exists; use a new immutable path')
    model=SignRepReference(c['model']['vendor_root'],c['model']['initialization_checkpoint']).to(device).eval()
    # Use the actual pinned official augmentation class, including OpenCV resize
    # and Albumentations normalization, instead of a guessed approximation.
    import importlib
    transform=importlib.import_module('augmentation.video.base_video_aug').Transformation()
    module=importlib.import_module('augmentation.video.base_video_aug')
    if Path(c['model']['vendor_root']).resolve() not in Path(module.__file__).resolve().parents:
        raise RuntimeError('conflicting augmentation package')
    records=read_manifest(manifest); embeddings=[]; features=[]; latents=[]; times=[]; offsets=[0]
    with torch.inference_mode():
        for record in records:
            pts,_,_=record_metadata(record)
            valid=(pts >= record['start_ms']/1000) & (pts < record['end_ms']/1000)
            indices=np.flatnonzero(valid)
            if len(indices)<16: raise ValueError('official protocol drops incomplete T16 windows; video has no complete window')
            rgb,pts=decode_frames(record['media_path'],indices),pts[valid]
            local=[]
            for start in range(0,len(rgb)-15,2):
                frames=transform.aug_video(list(rgb[start:start+16])).permute(1,0,2,3)[None].to(device)
                post,latent=model(frames)
                local.append(post[0].cpu().numpy()); latents.append(latent[0].cpu().numpy())
                times.append(float(pts[start+8]))
            local=np.stack(local); pooled=local.mean(0); norm=np.linalg.norm(pooled)
            if norm<=1e-12: raise ValueError('zero-norm official reference')
            embeddings.append(pooled/norm); features.append(local); offsets.append(offsets[-1]+len(local))
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,z=np.stack(embeddings),window_features=np.concatenate(features),
                        window_latents=np.stack(latents),window_times=np.asarray(times),offsets=np.asarray(offsets),
                        sample_ids=np.asarray([r['sample_id'] for r in records]),
                        source_groups=np.asarray([r['source_group_id'] for r in records]),
                        groups=np.asarray([r['signer_id'] or r['source_group_id'] for r in records]))
    protocol={'context':16,'native_frame_stride':2,'spatial':'official_direct_resize224',
              'normalization':'official_ImageNet','feature':'official_fc_on_unmasked_latent',
              'tail':'drop_incomplete','video_pooling_extension':'mean_post_fc_window_features_then_L2',
              'pretraining_history':'external_reference_unknown_original_sign_overlap'}
    report={**execution_stamp(),'status':'passed','arm':'A1_ref','sample_count':len(records),
            'protocol':protocol,'pooling_hash':object_hash(protocol),'manifest_sha256':sha256_file(manifest),
            'checkpoint_sha256':sha256_file(c['model']['initialization_checkpoint']),
            'output_sha256':sha256_file(path),'labels_read_by_encoder':False,
            'matched_continued_pretraining':False,'scientific_evidence':False}
    atomic_json(str(path)+'.json',report)
    return report
