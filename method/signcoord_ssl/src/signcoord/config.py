from __future__ import annotations

import copy
from pathlib import Path

import yaml

from signcoord.targets.spectral import window_starts

PROJECT = Path(__file__).resolve().parents[2]
BASE_CONFIG = Path(__file__).with_name("base_config.yaml")
ARMS = {"A0": (0., "off"), "A1_ref": (0., "off"), "A1_inspired": (0., "off"), "A2": (0., "off"), "A3": (.10, "off"),
        "A5": (.10, "magnitude"), "A6": (.10, "lag"),
        "A7": (.10, "phase"), "A8": (.10, "phase_basic"), "A9": (.10, "phase_shuffle")}


def merge_known(base, updates, prefix=""):
    for key, value in updates.items():
        if key not in base:
            raise ValueError(f"unknown config key: {prefix+key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"expected mapping: {prefix+key}")
            merge_known(base[key], value, prefix+key+".")
        else:
            base[key] = value


def load_config(path, overrides=(), arm=None, seed=None):
    base = yaml.safe_load(BASE_CONFIG.read_text())
    # Explicit implementation-specific backend paths/dimensions.
    base["model"].update(vendor_root=str(PROJECT.parents[1]/"third_party/SignRep"),
                         generic_vendor_root=str(PROJECT/"vendor/hiera"), tiny_dim=32,
                         attention_kernel="sdpa_4d_zero_cls")
    base["data"].update(downstream_manifests=[])
    updates = yaml.safe_load(Path(path).read_text())
    merge_known(base, updates)
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError("--set expects dotted.key=value")
        target = base
        for part in key.split(".")[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"unknown config path: {key}")
            target = target[part]
        leaf = key.split(".")[-1]
        if leaf not in target:
            raise ValueError(f"unknown config key: {key}")
        target[leaf] = yaml.safe_load(value)
    if arm is not None:
        base["arm"] = arm
    if seed is not None:
        base["seed"] = seed
    if base["arm"] not in ARMS:
        raise ValueError(f"arm {base['arm']} not wired yet; supported {list(ARMS)}")
    velocity, kind = ARMS[base["arm"]]
    if velocity == 0:
        base["loss"]["velocity_weight"] = 0.
    base["loss"]["relation"]["kind"] = kind
    validate(base)
    return base


def validate(c):
    evaluation=c['evaluation']
    if type(evaluation['selection_enabled']) is not bool or type(evaluation['selection_keep_features']) is not bool:
        raise ValueError('selection switches must be boolean')
    if evaluation['selection_enabled']:
        if evaluation['selection_split']!='dev' or evaluation['gallery_split']!='train' or evaluation['query_split']!='dev':
            raise ValueError('checkpoint selection requires train gallery and dev queries')
        if evaluation['selection_metric']!='retrieval_rec1' or evaluation['prototype_gallery']:
            raise ValueError('checkpoint selection uses frozen instance retrieval Rec@1')
        if any(not evaluation[key] for key in ('selection_gallery_manifest','selection_query_manifest','selection_labels_path')):
            raise ValueError('checkpoint selection requires gallery/query manifests and explicit labels')
    if type(c['train']['eval_every_updates']) is not int or c['train']['eval_every_updates']<=0:
        raise ValueError('positive integer required: train.eval_every_updates')
    if c['train']['optimizer']!='adamw' or c['train']['scheduler']!='cosine':
        raise ValueError('supported trainer recipe is AdamW with cosine scheduling')
    if c['train']['precision'] not in ('fp32','fp16','bf16','bf16_if_supported_else_fp16'):
        raise ValueError('unsupported training precision')
    if c['data']['sampler']['resize']!='letterbox' or c['model']['mask_semantics']!='true_means_keep':
        raise ValueError('supported preprocessing/mask recipe is letterbox/true_means_keep')
    if c['model']['encoder_taps']!=['final']:
        raise ValueError('only final encoder tap is implemented')
    if c['model']['decoder']['part_order']!=['L','R','F','B'] or c['model']['decoder']['target_supervision']!='all_valid':
        raise ValueError('decoder requires canonical part order and all-valid supervision')
    if c['data']['normalization']['origin']!='clip_median_shoulder_center' or c['data']['normalization']['scale']!='clip_median_shoulder_width':
        raise ValueError('only fixed clip-median shoulder normalization is implemented')
    if not c['data']['augmentation']['same_photometric_params_over_time']:
        raise ValueError('photometric parameters must be shared over the clip')
    if c['model']['attention_kernel'] not in ('vendor_5d','sdpa_4d_zero_cls'):
        raise ValueError('unknown attention kernel')
    if c['arm']=='A1_ref' and c['model']['attention_kernel']!='vendor_5d':
        raise ValueError('official reference requires the original vendor_5d attention kernel')
    normal=c['data']['normalization']
    if type(normal['gap_interpolation_max_frames']) is not int or not 0<=normal['gap_interpolation_max_frames']<=2:
        raise ValueError('primary source gap policy supports at most two frames')
    if not 0<=normal['interpolated_reliability_factor']<=1:
        raise ValueError('interpolated reliability factor must lie in [0,1]')
    geometry=c["loss"]["geometry"]
    if geometry["huber_delta"]<=0 or geometry["local_total_weight"]<0:
        raise ValueError("invalid geometry Huber/weight")
    for name,parts in (("global_part_weights",{"L","R","F","B"}),("local_hand_weights",{"L","R"})):
        weights=geometry[name]
        if set(weights)!=parts or any(v<0 for v in weights.values()) or abs(sum(weights.values())-1)>1e-6:
            raise ValueError(f"geometry.{name} must be nonnegative and sum to one")
    frames = c["data"]["sampler"]["frames"]
    if type(frames) is not int or frames != c["model"]["input_frames"]:
        raise ValueError("sampler frames must match encoder input geometry")
    if c["data"]["sampler"]["grid_fps"] <= 0 or not 0 <= c["model"]["mask_ratio"] < 1:
        raise ValueError("invalid fps/mask ratio")
    rel = c["loss"]["relation"]
    if rel["kind"] != "off" and len(window_starts(frames-1,rel["window"],rel["hop"])) < 2:
        raise ValueError("coordination requires >=2 complete velocity windows")
    if rel["warmup_zero_updates"] >= rel["ramp_end_update"]:
        raise ValueError("relation ramp must end after warmup")
    if c["model"]["decoder"]["width"] % c["model"]["decoder"]["heads"]:
        raise ValueError("decoder width must be divisible by heads")
    for key in ("microbatch_per_gpu", "accumulation_steps", "max_updates", "save_every_updates"):
        if type(c["train"][key]) is not int or c["train"][key] <= 0:
            raise ValueError(f"positive integer required: train.{key}")
    for key in ("horizontal_flip", "time_reverse", "independent_part_time_shift", "pose_guided_masking"):
        if c["data"]["augmentation"][key]:
            raise ValueError(f"augmentation {key} needs a separate evaluated policy")
    if c["data"]["sampler"]["temporal_speed_aug"]:
        raise ValueError("temporal speed augmentation is outside MVP")
    if c["loss"]["signed_relation_weight"] != 0 or c["loss"]["style_policy"] != "off":
        raise ValueError("signed relational geometry/style are not wired in primary MVP")
    if rel["window_tail_policy"] != "drop_incomplete" or rel["coherence_gate"]:
        raise ValueError("primary kernel requires dropped tail and no coherence gate")
    if c["model"]["backbone"] == "tiny_fixture" and c["model"]["initialization_mode"] != "random_init":
        raise ValueError("tiny encoder is a random-init fixture, never pretrained Hiera")
    if c["arm"] == "A0" and (c["model"]["backbone"] != "hiera_base_video_adapter" or
                               c["model"]["initialization_mode"] != "generic_hiera_mae"):
        raise ValueError("A0 is the frozen generic video MAE baseline")
    if c["arm"] == "A1_ref" and (c["model"]["input_frames"] != 16 or c["model"]["initialization_mode"] != "signrep_pretrained"):
        raise ValueError("official SignRep reference requires pretrained T16")


def relation_weight(c, update):
    r = c["loss"]["relation"]
    if r["kind"] == "off":
        return 0.
    fraction = (update-r["warmup_zero_updates"])/(r["ramp_end_update"]-r["warmup_zero_updates"])
    return r["weight"]*max(0., min(1., fraction))
