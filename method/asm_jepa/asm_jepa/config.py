"""YAML config loading: relative `inherits`, recursive merge, `--set key=value`, strict schemas.

Every stage has an explicit schema (nested dict of defaults). Unknown keys are rejected, both in files
and in `--set` overrides, so a config key can never be silently ignored. Leaves whose default is a dict
listed in OPEN_DICT_KEYS (e.g. per-family weights) are validated key by key too.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from asm_jepa.utils import sha256_json

PRETRAIN_SCHEMA = {
    "experiment": "aslc_jepa_base",
    "seed": 0,
    "run_dir": "runs/aslc_jepa_base_s0",
    "data": {
        "processed_root": "data/processed/aslc",
        "ssl_manifest": "ssl_train.jsonl",
        "gallery_split": "train",
        "val_split": "val",
        "schema": "coco133_to_aslc114_v1",
        "input_channels": 7,
        "max_frames": 64,
        "patch_frames": 4,
        "sampling": "full_clip_bins",
        "temporal_jitter_fraction": 0.15,
        "confidence_threshold": 0.20,
        # Raw extractor scores (RTMW SimCC) are not in [0,1]; the confidence channel is clip(score / scale, 0, 1).
        "confidence_scale": 10.0,
        "part_coverage_threshold": 0.20,
        "stats_path": "stats/train_v1.npz",
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "coordinate_saturation": 4.0,
        # audit-driven observation filters (asm_jepa/data/observation.py); off = original behaviour
        "in_image_only": False,
        "in_image_margin": 0.0,
        "hand_wrist_max_dist": 0.0,
    },
    "features": {
        "relations": False,
        "geometry_paths": False,
        "timing_paths": False,
        "path_families": ["left_wrist", "right_wrist", "left_minus_right_wrist"],
        "interval_widths_sec": [0.32, 0.64, 1.28],
        "centers_per_clip": 4,
        "logsignature_level": 2,
        "path_min_points": 3,
        "path_gap_policy": "strict_all_observed",
    },
    "model": {
        "dim": 256,
        "depth": 6,
        "heads": 8,
        "ffn_dim": 1024,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "drop_path": 0.10,
        "share_hand_tokenizer": True,
        "predictor_dim": 128,
        "predictor_depth": 2,
        "predictor_heads": 4,
        "predictor_ffn_dim": 512,
        "feature_key": "state_part_concat1024_l2",
    },
    "mask": {
        "ratio": 0.50,
        "mixture_random": 0.50,
        "mixture_part_blocks": 0.30,
        "mixture_temporal_blocks": 0.20,
        "block_min_patches": 2,
        "block_max_patches": 4,
        "temporal_block_min_patches": 2,
        "temporal_block_max_patches": 3,
        "hand_visible_fraction_min": 0.25,
        "min_visible_state_tokens": 8,
        "max_resample_attempts": 20,
        "ratio_fallback_factor": 0.8,
        "max_ratio_fallbacks": 6,
        "dependency_closure": True,
        "shoulders_public": True,
    },
    "augment": {
        "rotation_degrees": 5.0,
        "scale_range": [0.9, 1.1],
        "student_jitter_sigma": 0.003,
        "mirror_probability": 0.0,
        "time_reverse_probability": 0.0,
    },
    "loss": {
        "latent": "centered_ce",
        "student_temperature": 0.10,
        "teacher_temperature_start": 0.08,
        "teacher_temperature_final": 0.06,
        "teacher_temperature_warmup_steps": 3000,
        "center_momentum": 0.90,
        "family_weights": {"state": 1.0, "relation": 0.5, "geometry": 0.25, "timing": 0.25},
        # per-group weights inside a family, order [body, lh, rh, face, lh-rh, lh-nose, rh-nose, geometry, timing];
        # null = all 1 (original behaviour)
        "group_weights": None,
        "lambda_anchor": 0.0,
        "anchor_huber_beta": 0.10,
        "anchor_target": "shared_clean_view",
        "lambda_global": 0.0,
        "global_std_target": 1.0,
        "global_cov_weight": 0.04,
        "global_min_statistical_batch": 16,
    },
    "train": {
        "objective": "jepa",  # jepa | mae (R1 coordinate-MAE control)
        "max_steps": 60000,
        "batch_size_per_gpu": 32,
        "accum_steps": 4,
        "optimizer": "adamw",
        "lr": 3.0e-4,
        "min_lr": 1.0e-6,
        "betas": [0.9, 0.95],
        "eps": 1.0e-8,
        "weight_decay": 0.05,
        "warmup_steps": 3000,
        "grad_clip_norm": 1.0,
        "precision": "bf16",
        "ema_start": 0.996,
        "ema_end": 0.9999,
        "ema_schedule": "cosine",
        "compile": False,
        "deterministic": False,
        "checkpoint_every_steps": 1000,
        "keep_step_checkpoints": 3,
        "milestone_every_steps": 0,  # >0: also keep milestone_<step>.pt forever (stage-wise evaluation)
        "log_every_steps": 100,
        "profile_warmup_steps": 100,
    },
    "eval": {
        "frozen_every_steps": 5000,
        "selection_metric": "val/frozen_gloss_max/r1",
        "selection_uses_val_labels": True,
        "encoder": "teacher",
        "query_chunk": 256,
        "gallery_chunk": 8192,
        "tie_break": "class_id_ascending",
        "test_during_training": False,
        "extract_batch_size": 128,
        "diagnostic_batch_size": 256,
    },
}

LINEAR_SCHEMA = {
    "mode": "linear_probe",
    "seed": 0,
    "feature_key": "state_part_concat1024_l2",
    "feature_dim": 1024,
    "num_classes_from_vocab": True,
    "head": "linear",
    "data": {"processed_root": "data/processed/aslc"},
    "train": {
        "optimizer": "adamw",
        "lr": 1.0e-3,
        "weight_decay": 0.0,
        "batch_size": 256,
        "epochs": 100,
        "warmup_epochs": 5,
        "min_lr": 1.0e-5,
        "label_smoothing": 0.0,
    },
    "selection_metric": "val/cls/top1_instance",
    "test_during_training": False,
}

FINETUNE_SCHEMA = {
    "mode": "finetune_islr",
    "seed": 0,
    "model_config": "../pretrain/state.yaml",
    "feature_key": "state_part_concat1024_l2",
    "num_classes_from_vocab": True,
    "head": "layernorm_dropout_linear",
    "head_dropout": 0.20,
    "augment": True,
    "data": {
        "processed_root": "data/processed/aslc",
        "train_split": "train",
        "val_split": "val",
        "num_workers": 8,
        "pin_memory": True,
    },
    "train": {
        "optimizer": "adamw",
        "encoder_lr": 5.0e-5,
        "head_lr": 3.0e-4,
        "weight_decay": 1.0e-3,
        "batch_size_per_gpu": 32,
        "accum_steps": 4,
        "epochs": 100,
        "warmup_epochs": 10,
        "final_lr_ratio": 0.01,
        "label_smoothing": 0.10,
        "grad_clip_norm": 1.0,
        "precision": "bf16",
        "eval_batch_size": 128,
        "log_every_steps": 50,
    },
    "selection_metric": "val/cls/top1_instance",
    "test_during_training": False,
}

RETRIEVAL_SCHEMA = {
    "protocol": "gloss_max",
    "encoder": "teacher",
    "feature_key": "state_part_concat1024_l2",
    "query_chunk": 256,
    "gallery_chunk": 8192,
    "tie_break": "class_id_ascending",
    "gallery_split": "train",
    "denominator": "full_official_split",
    "ks": [1, 5, 10],
    "bootstrap_samples": 1000,
    "bootstrap_seed": 0,
}

POSE_SCHEMA = {
    "data": {"raw_root": "data/raw/ASL_Citizen", "processed_root": "data/processed/aslc"},
    "pose": {
        "library": "rtmlib",
        "revision": "03a1693e59e4f7cd84582c0fb30459b3bf18ad42",
        "solution": "Wholebody",
        "mode": "performance",
        "to_openpose": False,
        "backend": "onnxruntime",
        "device": "cuda",
        "require_cuda_provider": True,
        "model_cache": "data/models/rtmw133",
        "detector_url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        "detector_input_size": [640, 640],
        "pose_url": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
        "pose_input_size": [288, 384],
        "native_resolution": True,
        "native_timestamps": True,
        "cfr_fallback_timestamps": True,
        "detection_interval_frames": 1,
        "select_one_signer": True,
        "confidence_threshold": 0.20,
        "interpolate": False,
        "smooth": False,
        "cache_format": "npz",
        "overwrite_valid_cache": False,
    },
}

DATA_SCHEMA = {
    "dataset": "asl_citizen",
    "expected_version": "v1.0",
    "raw_root": "data/raw/ASL_Citizen",
    "processed_root": "data/processed/aslc",
    "splits_dir": "splits",
    "videos_dir": "videos",
    "split_files": {"train": "train.csv", "val": "val.csv", "test": "test.csv"},
    # Positional adapter (official loader order: user, filename, gloss) used when headers differ.
    "csv_positions": {"signer": 0, "filename": 1, "gloss": 2},
    "csv_headers": {"signer": "Participant ID", "filename": "Video file", "gloss": "Gloss"},
    "expected_counts": {"train": 40154, "val": 10304, "test": 32941, "glosses": 2731,
                        "train_signers": 35, "val_signers": 6, "test_signers": 11},
    "probe_workers": 8,
}

SCHEMAS = {"pretrain": PRETRAIN_SCHEMA, "linear": LINEAR_SCHEMA, "finetune": FINETUNE_SCHEMA,
           "retrieval": RETRIEVAL_SCHEMA, "pose": POSE_SCHEMA, "data": DATA_SCHEMA}

# dict-valued leaves whose keys are fixed by the schema (validated, not free-form)
OPEN_DICT_KEYS: set[str] = set()


class ConfigError(ValueError):
    pass


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_with_inherits(path: Path, seen=()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ConfigError(f"inheritance cycle at {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    parents = raw.pop("inherits", []) or []
    if isinstance(parents, str):
        parents = [parents]
    merged: dict = {}
    for parent in parents:
        merged = _merge(merged, _load_with_inherits((path.parent / parent), seen + (path,)))
    return _merge(merged, raw)


def _validate(cfg: dict, schema: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if k not in schema:
            raise ConfigError(f"unknown config key: {key}")
        if isinstance(schema[k], dict):
            if not isinstance(v, dict):
                raise ConfigError(f"{key} must be a mapping")
            out[k] = _validate(v, schema[k], key + ".")
        else:
            out[k] = v
    for k, default in schema.items():
        if k not in out:
            out[k] = copy.deepcopy(default) if not isinstance(default, dict) else _validate({}, default, f"{prefix}{k}.")
    return out


def apply_set(cfg: dict, assignment: str, schema: dict) -> dict:
    if "=" not in assignment:
        raise ConfigError(f"--set expects key=value, got {assignment!r}")
    key, value = assignment.split("=", 1)
    parts = key.strip().split(".")
    node_schema = schema
    for p in parts:
        if not isinstance(node_schema, dict) or p not in node_schema:
            raise ConfigError(f"--set: unknown config key {key!r}")
        node_schema = node_schema[p]
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(value)
    return cfg


def load_config(path, schema: str, sets=()) -> dict:
    schema_dict = SCHEMAS[schema]
    cfg = _load_with_inherits(Path(path)) if path is not None else {}
    for s in sets or ():
        cfg = apply_set(cfg, s, schema_dict)
    cfg = _validate(cfg, schema_dict)
    cfg["_meta"] = {"source": str(Path(path).resolve()) if path else None, "schema": schema}
    return cfg


def strip_meta(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def dump_resolved(cfg: dict, path) -> None:
    from asm_jepa.utils import atomic_write_bytes

    atomic_write_bytes(path, yaml.safe_dump(strip_meta(cfg), sort_keys=False).encode())


def config_hash(cfg: dict) -> str:
    return sha256_json(strip_meta(cfg))


ARCH_KEYS = (
    ("data", "schema"), ("data", "input_channels"), ("data", "max_frames"), ("data", "patch_frames"),
    ("data", "confidence_threshold"), ("data", "confidence_scale"), ("data", "part_coverage_threshold"),
    ("data", "in_image_only"), ("data", "in_image_margin"), ("data", "hand_wrist_max_dist"),
    ("features",), ("model",),
)


def arch_signature(cfg: dict) -> dict:
    """Keys that must agree between a checkpoint and a config that re-uses its encoder."""
    out = {}
    for path in ARCH_KEYS:
        node, default = cfg, PRETRAIN_SCHEMA
        for p in path:
            default = default[p]
            # keys added after a checkpoint was written fall back to the schema default (= old behaviour)
            node = node[p] if p in node else default
        out[".".join(path)] = copy.deepcopy(node)
    return out


def arch_diff(a: dict, b: dict) -> list[str]:
    sa, sb = arch_signature(a), arch_signature(b)
    diffs = []
    for k in sa:
        if isinstance(sa[k], dict):
            for kk in sorted(set(sa[k]) | set(sb[k])):
                if sa[k].get(kk) != sb[k].get(kk):
                    diffs.append(f"{k}.{kk}: {sa[k].get(kk)!r} != {sb[k].get(kk)!r}")
        elif sa[k] != sb[k]:
            diffs.append(f"{k}: {sa[k]!r} != {sb[k]!r}")
    return diffs
