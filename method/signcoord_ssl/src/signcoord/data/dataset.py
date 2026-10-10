from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from signcoord.provenance import sha256_file,object_hash
from signcoord.targets.geometry import normalized_geometry
from signcoord.targets.reliability import fill_short_gaps
from .manifest import read_manifest
from .pose_cache import read_pose_cache
from .sampling import nearest_frame_plan
from .transforms import letterbox,in_frame
from .video_io import video_metadata, decode_frames
from .video_index import record_metadata,verify_media


class SSLVideoDataset(Dataset):
    """Labels/signers stay in the manifest and never enter student inputs/targets."""
    def __init__(self, manifest, pose_root, frames=32, grid_fps=25., spatial_size=(224,224), verify_hashes=True,
                 max_repeat_fraction=.10, normalization=None, color_jitter=0., sampling_seed=3407):
        self.records = read_manifest(manifest, {"ssl_train", "ssl_dev"})
        self.pose_root, self.frames, self.grid_fps = Path(pose_root), frames, grid_fps
        self.spatial_size, self.verify_hashes = tuple(spatial_size), verify_hashes
        self.max_repeat_fraction, self.color_jitter = max_repeat_fraction, color_jitter
        self.normalization = normalization or {}
        self.sampling_seed, self.sample_epoch = sampling_seed, 0
        self.source_gap_policy=object_hash({'policy':'verified_track_only_bounded_source_gaps_v1',
            'max_gap':self.normalization.get('gap_interpolation_max_frames',2),
            'reliability_factor':self.normalization.get('interpolated_reliability_factor',.5),
            'allowed_statuses':[1,2],'endpoint_confidence':.30})
        self._pose_artifact_hashes=None
        self.target_recipe_hash=object_hash({'normalization':self.normalization,
            'spatial_size':self.spatial_size,'max_repeat_fraction':self.max_repeat_fraction,
            'grid_fps':self.grid_fps,'frames':self.frames})

    def pose_path(self,key):
        root=self.pose_root.resolve();path=(root/key).resolve()
        if Path(key).is_absolute() or root not in path.parents:
            raise ValueError('canonical pose cache path must remain inside pose_cache_root')
        return path

    @property
    def pose_artifact_hashes(self):
        if self._pose_artifact_hashes is None:
            hashes={}
            for record in self.records:
                key=record.get('pose_cache_key')
                if not key:raise ValueError('SSL manifest requires a canonical pose cache; run extract-pose first')
                if key not in hashes:hashes[key]=sha256_file(self.pose_path(key))
                if record.get('pose_cache_sha256') not in (None,hashes[key]):
                    raise ValueError('canonical pose cache differs from manifest content hash')
            self._pose_artifact_hashes=hashes
        return self._pose_artifact_hashes

    @property
    def pose_artifact_hash(self):
        return object_hash(self.pose_artifact_hashes)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        if not self.verify_hashes:
            raise ValueError('unverified media reads are outside the canonical dataset contract')
        pts, source_width, source_height = record_metadata(r)
        if not r.get('pose_cache_key'):raise ValueError('SSL manifest requires a canonical pose cache; run extract-pose first')
        pose_path=self.pose_path(r['pose_cache_key'])
        verify_media(pose_path,self.pose_artifact_hashes[r['pose_cache_key']])
        pose, metadata = read_pose_cache(pose_path, r["media_sha256"])
        if len(pose["pts_s"]) != len(pts) or not np.allclose(pose["pts_s"], pts, atol=1e-6, rtol=0):
            raise ValueError("pose/RGB source frame or PTS alignment mismatch")
        interpolated=torch.zeros_like(torch.from_numpy(pose['observed']))
        gap_enabled=bool(metadata.get('tracking_identity_verified')) and self.normalization.get('gap_interpolation_max_frames',2)>0
        if gap_enabled:
            source_xy=torch.from_numpy(pose['xy_px']).float()
            source_observed=torch.from_numpy(pose['observed']) & in_frame(source_xy,source_width,source_height)
            allowed=torch.from_numpy(np.isin(pose['status'],[1,2]))
            xy,confidence,usable,interpolated=fill_short_gaps(source_xy,torch.from_numpy(pose['confidence']),
                source_observed,torch.from_numpy(pose['pts_s']),torch.from_numpy(pose['track_ids']),
                self.normalization.get('gap_interpolation_max_frames',2),
                reliability_factor=self.normalization.get('interpolated_reliability_factor',.5),gap_allowed=allowed)
            pose['xy_px'],pose['confidence'],pose['observed']=xy.numpy(),confidence.numpy(),usable.numpy()
        begin,end = r["start_ms"]/1000,r["end_ms"]/1000
        max_start = end-self.frames/self.grid_fps
        if max_start>begin:
            rng = np.random.default_rng(self.sampling_seed+index+self.sample_epoch*len(self))
            begin = rng.uniform(begin,max_start)
        plan = nearest_frame_plan(pts, begin, end, self.frames, self.grid_fps,allowed_start_s=r["start_ms"]/1000)
        idx = plan["frame_index"]
        video = decode_frames(r["media_path"], idx)
        rgb = torch.from_numpy(video).permute(0,3,1,2).float()/255
        xy = torch.from_numpy(pose["xy_px"][idx].copy()).float()
        source_inside = in_frame(xy,source_width,source_height)
        rgb, xy, affine, _ = letterbox(rgb, xy, self.spatial_size)
        if self.color_jitter:
            # Shared scalar factors across the entire temporal clip: no flicker.
            generator=torch.Generator().manual_seed(self.sampling_seed+index+self.sample_epoch*len(self)+104729)
            brightness,contrast = 1+(torch.rand(2,generator=generator)-.5)*2*self.color_jitter
            rgb = ((rgb-rgb.mean((0,2,3),keepdim=True))*contrast+rgb.mean((0,2,3),keepdim=True))*brightness
            rgb = rgb.clamp(0,1)
        conf = torch.from_numpy(pose["confidence"][idx]).float()
        observed = torch.from_numpy(pose["observed"][idx]) & source_inside
        fv = torch.from_numpy(plan["frame_valid"])
        target = normalized_geometry(xy[None], conf[None], observed[None], fv[None], self.spatial_size,
                 min_shoulder_frames=self.normalization.get("min_valid_shoulder_frames",4),
                 min_scale_fraction=self.normalization.get("min_scale_fraction",.02))
        target = {k: v[0] for k,v in target.items()}
        target["spectral_frame_valid"] = torch.from_numpy(plan["timing_valid"])
        if plan["repeat_fraction"] > self.max_repeat_fraction:
            target["spectral_frame_valid"][:] = False
        mean = rgb.new_tensor([.485,.456,.406])[None,:,None,None]
        std = rgb.new_tensor([.229,.224,.225])[None,:,None,None]
        student_inputs = {"rgb": ((rgb-mean)/std).permute(1,0,2,3), "frame_valid": fv,
                          "times": torch.from_numpy(plan["requested_timestamp_s"])}
        return {"student_inputs": student_inputs, "targets": target,
                "audit": {"sample_id": r["sample_id"], "repeat_fraction": plan["repeat_fraction"], "affine": affine,
                          'source_gap_interpolation_enabled':gap_enabled,
                          'interpolated_source_points':int(interpolated.sum()),
                          'sampled_interpolated_points':int(interpolated[idx].sum()),
                          'source_gap_disabled_reason':None if gap_enabled else 'tracking_identity_unverified_or_gap_policy_off'}}
