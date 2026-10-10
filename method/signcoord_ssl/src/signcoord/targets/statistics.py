import json

import torch

from signcoord.provenance import atomic_json, execution_stamp, sha256_file
from signcoord.targets.schema import ANCHORS, PART_SLICES, SCHEMA_HASH
from .motion import interval_velocity
from .spectral import fit_energy_thresholds
from .magnitude import fit_magnitude_calibration


def fit_target_statistics(dataset, manifest_path, output, phase_output,
                          window=16, hop=8, min_conf=.30, percentile=10, floor=1e-6):
    if any(r["split"] != "ssl_train" for r in dataset.records):
        raise ValueError("target calibration may only use SSL train")
    # Retain target tensors only; never keep the decoded RGB corpus in memory.
    collected = {"global": [], "reliability": [], "times": [], "spectral": []}
    for i in range(len(dataset)):
        sample = dataset[i]
        collected["global"].append(sample["targets"]["global"])
        collected["reliability"].append(sample["targets"]["reliability"])
        collected["times"].append(sample["student_inputs"]["times"])
        collected["spectral"].append(sample["targets"]["spectral_frame_valid"])
    glob, rel, times, spectral = (torch.stack(collected[k]) for k in ("global", "reliability", "times", "spectral"))
    vel, vel_rel = interval_velocity(glob,times,rel)
    groups = {}
    for part, sl in PART_SLICES.items():
        x, r = glob[...,sl,:],rel[...,sl]
        weights = r[...,None]
        if r.sum() == 0:
            raise ValueError(f"no train observations for {part}")
        mean=(x*weights).sum((0,1,2))/r.sum()
        scale=((((x-mean).square()*weights).sum()/(2*r.sum())).sqrt().clamp_min(.05))
        vr=vel_rel[...,sl]
        if vr.sum() == 0:
            raise ValueError(f"no train velocities for {part}")
        vscale=((vel[...,sl,:].square()*vr[...,None]).sum()/(2*vr.sum())).sqrt().clamp_min(.05)
        groups[part]={"mean":mean.tolist(),"scale":scale.item(),"velocity_scale":vscale.item(),
                      "coordinate_weight":r.sum().item(),"velocity_weight":vr.sum().item()}
    common={**execution_stamp(),"schema_hash":SCHEMA_HASH,"ssl_manifest_hash":sha256_file(manifest_path),
            "fit_split":"ssl_train","sample_ids":[r["sample_id"] for r in dataset.records],
            "grid_fps":dataset.grid_fps,"frames":dataset.frames,"sample_plan_seed":dataset.sampling_seed,
            "sample_epoch":dataset.sample_epoch,"timing_policy":"source_half_interval_and_manifest_bounded_nearest_v3",
            "source_gap_policy_hash":dataset.source_gap_policy,
            "pose_artifact_hash":dataset.pose_artifact_hash,
            "target_recipe_hash":dataset.target_recipe_hash,
            "geometry_input_policy":"source_and_transformed_frame_bounds_v1"}
    stats={**common,"schema_version":"target_stats_v1","groups":groups}
    anchor_rel=vel_rel[...,ANCHORS]*(spectral[:,1:] & spectral[:,:-1])[...,None]
    calibration=fit_energy_thresholds(vel[...,ANCHORS,:],anchor_rel,window,hop,
                                     percentile=percentile,floor=floor,min_conf=min_conf)
    magnitude=fit_magnitude_calibration(vel[...,ANCHORS,:],anchor_rel,window,hop,
                                      percentile=percentile,floor=floor,min_conf=min_conf)
    phase={**common,"schema_version":"phase_calibration_v1","window":window,"hop":hop,
           "percentile":percentile,"floor":floor,"min_confidence":min_conf,
           "thresholds":{k:v.tolist() for k,v in calibration.items()},"magnitude":magnitude}
    atomic_json(output,stats)
    atomic_json(phase_output,phase)
    return stats,phase


def read_statistics(path, manifest_path=None, dataset=None):
    stats=json.load(open(path))
    if stats["schema_hash"] != SCHEMA_HASH or stats["fit_split"] != "ssl_train":
        raise ValueError("statistics schema/split mismatch")
    if manifest_path is not None and stats["ssl_manifest_hash"] != sha256_file(manifest_path):
        raise ValueError("statistics fitted to a different SSL manifest")
    if dataset is not None and (stats["frames"] != dataset.frames or stats["grid_fps"] != dataset.grid_fps):
        raise ValueError("statistics temporal sampler mismatch")
    if dataset is not None and stats.get("timing_policy") != "source_half_interval_and_manifest_bounded_nearest_v3":
        raise ValueError("statistics sampling timing policy changed; recalibrate train-only statistics")
    if dataset is not None and stats.get("geometry_input_policy") != "source_and_transformed_frame_bounds_v1":
        raise ValueError("statistics source visibility policy changed; recalibrate train-only statistics")
    if dataset is not None and stats.get('source_gap_policy_hash')!=dataset.source_gap_policy:
        raise ValueError('statistics source gap/tracking policy changed; recalibrate train-only statistics')
    if dataset is not None and stats.get('pose_artifact_hash')!=dataset.pose_artifact_hash:
        raise ValueError('statistics canonical pose contents changed; recalibrate train-only statistics')
    if dataset is not None and stats.get('target_recipe_hash')!=dataset.target_recipe_hash:
        raise ValueError('statistics geometry/spectral recipe changed; recalibrate train-only statistics')
    return stats


def invert_standardization(pred, groups):
    return torch.cat([pred[...,sl,:]*groups[part]["scale"]+pred.new_tensor(groups[part]["mean"])
                      for part,sl in PART_SLICES.items()],dim=-2)
