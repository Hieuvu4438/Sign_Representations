"""Strict JSON config: unknown keys, wrong types, bad enums and cross-field errors raise."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

AUXILIARIES = ("none", "spectral_logmag_l1", "velocity_l1", "acceleration_l1", "window_time_l1")

DEFAULT: Dict[str, Any] = {
    "schema_version": "signrep_spectral_config_v1",
    "mode": "canonical_continuation_v1",
    "arm_id": "",
    "paths": {
        "vendor_root": "third_party/SignRep",
        "checkpoint": "checkpoints/signrep/ckpt.pt",
        "dataset_root": None,
        "manifest": None,
        "scaler": None,
        "calibration": None,
    },
    "checkpoint_sha256": "f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b",
    "data": {
        "dataset": "ASL_Citizen",
        "train_split": "train",
        "clip_length": 16,
        "sampling": "uniform_video_then_start",
        "short_clip_policy": "exclude",
        "tail_policy": "drop_incomplete",
        "clock_policy": "decoded_pts",
        "max_relative_dt_deviation": 0.05,
        "augmentation": "upstream_resize224_imagenet_rgb",
    },
    "targets": {
        "schema": "hand_image_canonical_v1_xy_r2",
        "hand_order": ["left", "right"],
        "scale_joint_pair": [0, 9],
        "scale_min_observations": 4,
        "scale_floor": 0.02,
        "max_relative_extent": 4.0,
        "confidence_threshold": 1.0,
        "missing_policy": "strict_window",
        "interpolate": False,
        "scaler_std_floor": 0.05,
        "scaler_seed": 1700,
        "scaler_windows_per_video": 1,
    },
    "model": {
        "out_dim": 768,
        "total_frames": 16,
        "mask_ratio": 0.6,
        "decoder_hidden_dim": 512,
        "decoder_latent_dim": 384,
        "decoder_initialization": "xavier_uniform_zero_bias_v1",
        "decoder_seed": 1702,
        "heads": {"left_pose": 42, "right_pose": 42, "wrist_image_xy": 4},
        "attention_kernel": "sdpa_4d_reshape",
    },
    "loss": {
        "smooth_l1_beta": 1.0,
        "geometric_head_weights": [1.0, 1.0, 1.0],
        "var_weight": 0.0,
        "cov_weight": 0.0,
        "auxiliary": "spectral_logmag_l1",
        "lambda_spec": 0.1,
        "lambda_aux": 0.0,
        "lambda_mode": "fixed",
        "target_grad_ratio": 0.1,
        "spec_warmup_updates": 100,
        "groups": ["left_hand_motion", "right_hand_motion", "wrist_translation"],
        "windows": [{"n": 8, "hop": 4}, {"n": 16, "hop": 8}],
        "window": "hann_periodic",
        "fft_norm": "backward",
        "amplitude_normalization": "window_sum",
        "centering": "clip_observed_mean",
        "reduction": "fixed_group_scale_mean_v1",
        "include_dc": True,
        "zero_padding": False,
    },
    "train": {
        "seed": 1701,
        "device": "cpu",
        "batch_size": 2,
        "num_workers": 0,
        "amp": False,
        "gradient_accumulation": 1,
        "lr_encoder": 0.00001,
        "lr_decoder": 0.0001,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.05,
        "grad_clip_norm": 1.0,
        "max_updates": 2,
        "scheduler": "constant",
        "lr_warmup_updates": 0,
        "log_every_updates": 1,
        "save_every_updates": 0,
        "resume": None,
        "ddp": False,
        "calibration_batches": 8,
        "calibration_seed": 1703,
    },
    "evaluation": {
        "feature_key": "features",
        "window_length": 16,
        "stride": 2,
        "pooling": "uniform_window_mean_then_l2",
        "batch_size": 64,
        "num_workers": 8,
        "amp": True,
    },
    "budget": {"max_wall_seconds": 900, "max_output_bytes": None, "max_peak_memory_bytes": None},
}

ENUMS = {
    ("mode",): ("canonical_continuation_v1",),
    ("loss", "auxiliary"): AUXILIARIES,
    ("loss", "lambda_mode"): ("fixed", "grad_ratio_calibrated"),
    ("train", "scheduler"): ("constant", "cosine"),
    ("model", "attention_kernel"): ("vendor_default", "sdpa_4d_reshape"),
    ("evaluation", "pooling"): ("uniform_window_mean_then_l2",),
    ("targets", "missing_policy"): ("strict_window",),
}
NULLABLE_STR = {("paths", k) for k in ("dataset_root", "manifest", "scaler", "calibration")} | {
    ("train", "resume"),
}
NULLABLE_NUM = {("budget", "max_output_bytes"), ("budget", "max_peak_memory_bytes")}


class ConfigError(ValueError):
    pass


def _check(node: Any, ref: Any, path: tuple) -> None:
    if isinstance(ref, dict):
        if not isinstance(node, dict):
            raise ConfigError(f"{'.'.join(path)} must be an object")
        unknown = set(node) - set(ref)
        if unknown and path != ("model", "heads"):
            raise ConfigError(f"unknown keys at {'.'.join(path) or '<root>'}: {sorted(unknown)}")
        missing = set(ref) - set(node)
        if missing:
            raise ConfigError(f"missing keys at {'.'.join(path) or '<root>'}: {sorted(missing)}")
        for k in ref:
            _check(node[k], ref[k], path + (k,))
        return
    if path in NULLABLE_STR:
        if node is not None and not isinstance(node, str):
            raise ConfigError(f"{'.'.join(path)} must be string or null")
        return
    if path in NULLABLE_NUM:
        if node is not None and not isinstance(node, (int, float)):
            raise ConfigError(f"{'.'.join(path)} must be number or null")
        return
    if isinstance(ref, bool):
        ok = isinstance(node, bool)
    elif isinstance(ref, (int, float)):
        ok = isinstance(node, (int, float)) and not isinstance(node, bool)
    elif isinstance(ref, str):
        ok = isinstance(node, str)
    elif isinstance(ref, list):
        ok = isinstance(node, list)
    else:
        ok = True
    if not ok:
        raise ConfigError(f"{'.'.join(path)} has wrong type {type(node).__name__}")


def validate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    _check(cfg, DEFAULT, ())
    for path, allowed in ENUMS.items():
        v = cfg
        for p in path:
            v = v[p]
        if v not in allowed:
            raise ConfigError(f"{'.'.join(path)}={v!r} not in {allowed}")
    loss, train = cfg["loss"], cfg["train"]
    aux = loss["auxiliary"]
    if aux != "spectral_logmag_l1" and loss["lambda_spec"] != 0:
        raise ConfigError("lambda_spec must be 0 unless auxiliary=spectral_logmag_l1")
    if aux not in ("none", "spectral_logmag_l1") and loss["lambda_mode"] == "fixed" and loss["lambda_aux"] <= 0:
        raise ConfigError("control auxiliaries need lambda_aux>0 or grad_ratio_calibrated")
    if aux == "spectral_logmag_l1" and loss["lambda_aux"] != 0:
        raise ConfigError("spectral arm uses lambda_spec, not lambda_aux")
    if aux == "none" and (loss["lambda_spec"] != 0 or loss["lambda_aux"] != 0 or loss["lambda_mode"] != "fixed"):
        raise ConfigError("baseline must have lambda_spec=lambda_aux=0 and lambda_mode=fixed")
    if train["ddp"]:
        raise ConfigError("DDP is not implemented in the MVP")
    if train["resume"] is not None:
        raise ConfigError("resume is not implemented in this MVP (start a new run instead)")
    if train["gradient_accumulation"] != 1:
        raise ConfigError("gradient_accumulation != 1 not supported in the MVP")
    if cfg["data"]["clip_length"] != 16 or cfg["model"]["total_frames"] != 16:
        raise ConfigError("public SignRep constructor hardcodes T=16")
    for w in loss["windows"]:
        if set(w) != {"n", "hop"} or w["n"] > 16 or w["n"] < 2 or w["hop"] < 1:
            raise ConfigError(f"bad window {w}")
    if (loss["var_weight"] or loss["cov_weight"]) and train["batch_size"] < 2:
        raise ConfigError("var/cov regularizers need batch_size>=2")
    if len(loss["geometric_head_weights"]) != len(cfg["model"]["heads"]):
        raise ConfigError("geometric_head_weights length must match heads")
    return cfg


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "heads":
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, paths_config: str | Path | None = None) -> Dict[str, Any]:
    with open(path) as f:
        cfg = json.load(f)
    if "extends" in cfg:
        parent = Path(path).parent / cfg.pop("extends")
        cfg = deep_update(load_config(parent), cfg)
    if paths_config:
        with open(paths_config) as f:
            local = json.load(f)
        if set(local) - {"paths"}:
            raise ConfigError("paths-config may only contain 'paths'")
        cfg = deep_update(cfg, local)
    return cfg


def resolved(path: str | Path, paths_config: str | Path | None = None) -> Dict[str, Any]:
    return validate(load_config(path, paths_config))


def config_hash(cfg: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
