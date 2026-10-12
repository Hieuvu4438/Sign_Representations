"""Strict two-stage config loading, with recorded inheritance and stable paths."""
from copy import deepcopy
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
import math
import yaml

from .cache.hashing import canonical_hash


class ConfigError(ValueError):
    pass


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ConfigError(f"Duplicate or non-string YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def deep_merge(base, override):
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def extension_defaults(value):
    """Original proposal14.1 configs remain valid as optional features are added."""
    result = deepcopy(value)
    if not isinstance(result, dict):
        return result
    schema = yaml.safe_load(files("signpath").joinpath("config_schema.yaml").read_text())
    if isinstance(result.get("cache"), dict):
        result["cache"].setdefault("reader_lru_samples", schema["cache"]["reader_lru_samples"])
    result.setdefault("downstream", deepcopy(schema["downstream"]))
    return result


def _load(path, active, chain):
    path = path.resolve()
    if path in active:
        raise ConfigError(f"Config inheritance cycle at {path}")
    with path.open() as stream:
        config = yaml.load(stream, Loader=UniqueLoader)
    if not isinstance(config, dict):
        raise ConfigError(f"Expected a YAML mapping: {path}")
    parent = config.pop("extends", None)
    if parent is not None:
        if not isinstance(parent, str):
            raise ConfigError("extends must be a path string")
        config = deep_merge(_load(path.parent / parent, active | {path}, chain), config)
    chain.append(str(path))
    return config


def _validate_types(value, schema, prefix=""):
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            raise ConfigError(f"{prefix}: expected mapping")
        unknown, missing = set(value) - set(schema), set(schema) - set(value)
        if unknown or missing:
            raise ConfigError(f"{prefix}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
        for key in schema:
            _validate_types(value[key], schema[key], f"{prefix}.{key}".strip("."))
    elif schema is None:
        if value is not None and not isinstance(value, str):
            raise ConfigError(f"{prefix}: expected string or null")
    elif isinstance(schema, list):
        if not isinstance(value, list) or not value:
            raise ConfigError(f"{prefix}: expected nonempty list")
        for item in value:
            _validate_types(item, schema[0], prefix + "[]")
    elif isinstance(schema, bool):
        if type(value) is not bool:
            raise ConfigError(f"{prefix}: expected bool")
    elif isinstance(schema, (int, float)):
        valid = type(value) is int if type(schema) is int else type(value) in (int, float)
        if not valid or not math.isfinite(value):
            raise ConfigError(f"{prefix}: expected finite {type(schema).__name__}")
    elif not isinstance(value, str):
        raise ConfigError(f"{prefix}: expected string")


def validate(config):
    config = extension_defaults(config)
    schema = yaml.safe_load(files("signpath").joinpath("config_schema.yaml").read_text())
    _validate_types(config, schema)
    def require(condition, message):
        if not condition:
            raise ConfigError(message)
    def choice(section, key, values):
        require(config[section][key] in values, f"Unsupported {section}.{key}: {config[section][key]}")
    require(config["schema_version"] == 1, "schema_version must be 1")
    data, cache, pose, geo, mask, model, loss, train, runtime = (
        config[k] for k in ("data", "cache", "pose", "geometry", "masking", "model", "loss", "training", "runtime"))
    choice("cache", "kind", {"per_frame_dino", "synthetic_fixture"})
    choice("data", "adapter", {"asl_citizen", "local_jsonl"})
    require(not data["expose_labels_to_pretrain"] and data["pretrain_split"] == "train",
            "Pretraining must use train only and cannot expose labels")
    require(not runtime["allow_random_pretrained_fallback"], "Random pretrained fallback is forbidden")
    require(model["raw_dimensions"] == [cache["visual_dim"]] * 4 + [28, 84, 84], "Raw dimensions do not match cache")
    require(cache["visual_streams"] == ["global", "left_hand", "right_hand", "face"], "Visual stream order mismatch")
    require(model["raw_streams"] == ["global", "left_hand", "right_hand", "face", "body", "left_pose", "right_pose"], "Raw stream order mismatch")
    for section in (model["encoder"], model["predictor"]):
        require(section["heads"] > 0 and model["embed_dim"] > 0 and model["embed_dim"] % section["heads"] == 0, "embed_dim must be divisible by heads")
        require(section["layers"] > 0, "Transformer layers must be positive")
    require(model["predictor"]["embed_dim"] == model["embed_dim"], "Predictor dimension mismatch")
    require(all(1 <= layer <= model["encoder"]["layers"] for layer in model["encoder"]["return_layers"]), "Invalid encoder return layer")
    require(model["predictor"]["query_values"] == "metadata_only", "Queries may contain public metadata only")
    require(geo["degree"] == 2 and geo["basis"] == "level1_then_upper_lie2" and not geo["include_level0"], "Only degree-2 Lie basis without level0 is supported")
    choice("geometry", "descriptor", {"logsig2_anchor_timing", "ordinary_motion_anchor_timing"})
    choice("geometry", "role", {"target_only", "input_enabled"})
    choice("experiment", "supervision_regime", {"sign_only_ssl", "paired", "supervised_downstream"})
    choice("masking", "regime", {"strict_temporal", "cross_modal"})
    choice("masking", "strategy", {"synchronized_temporal", "anatomical_pose", "stream_temporal", "modality_dropout", "shubert_code_random"})
    choice("masking", "geometry_target_eligibility", {"fully_hidden", "any_overlap"})
    choice("training", "precision", {"fp32", "fp16", "bf16", "bf16_if_supported_else_fp32"})
    choice("loss", "semantic", {"normalized_l1", "cosine"})
    choice("geometry", "compute_dtype", {"float32", "float64"})
    choice("evaluation", "split_during_development", {"val"})
    fixed_contracts = {
        "data": {"sampler": "uniform_source_then_segment", "short_clip_policy": "pad_invalid"},
        "cache": {"tensor_dtype_on_disk": "float16", "timestamps_dtype": "float64", "roi_policy": "per_frame_letterbox"},
        "geometry": {"path_groups": "default_11_groups_v1", "incomplete_boundary_window": "invalid"},
        "masking": {"closure": "full_declared_support", "empty_semantic_policy": "resample_then_global_skip", "empty_geometry_policy": "zero_term_with_count", "short_clip_min_context": "quarter_of_valid_tokens"},
        "model": {"time_encoding": "physical_seconds_sinusoidal"},
        "loss": {"semantic_normalization": "global_active_type_mean", "geometry": "smooth_l1", "geometry_normalization": "global_active_group_scale_mean", "accumulation_normalization": "microbatch_global_means"},
        "training": {"optimizer": "adamw", "finite_policy": "globally_skip_and_record"},
    }
    for section, fields in fixed_contracts.items():
        for key, value in fields.items():
            require(config[section][key] == value, f"Unsupported {section}.{key}: {config[section][key]}")
    require(geo["projection"]["kind"] == "fixed_gaussian" and geo["projection"]["output_dim"] > 0, "Unsupported geometry projection")
    require(geo["lift"] == "none" and not geo["append_time_channel"], "Lifts/time channels are unavailable until M10")
    require(geo["projection"]["fit_statistics_split"] == "train", "Target statistics must be fitted on train")
    require(geo["projection"]["output_dim"] == model["predictor"]["geometry_output_dim"], "Geometry output dimension mismatch")
    require(0 < mask["requested_frame_ratio"] < 1, "Mask ratio must lie in (0,1)")
    require(0 < mask["span_seconds"][0] <= mask["span_seconds"][-1], "Invalid mask span seconds")
    require(all(w > 0 for w in geo["window_seconds"]), "Geometry windows must be positive")
    require(geo["center_stride_samples"] > 0 and geo["min_distinct_observations"] >= 2, "Invalid geometry sampling")
    if mask["regime"] == "strict_temporal":
        require(mask["strategy"] == "synchronized_temporal" and mask["map_resampled_aliases"], "Strict temporal requires synchronized alias-aware masking")
        require(cache["detector_mode"] == "IMAGE" and cache["temporal_dependency_policy"] == "same_frame"
                and pose["smoothing"] == "none" and pose["interpolation"] == "none", "Strict cache requires same-frame extraction without tracking/smoothing")
        require(mask["geometry_target_eligibility"] == "fully_hidden", "Strict geometry targets must be fully hidden")
    else:
        require(mask["strategy"] != "synchronized_temporal", "Synchronized temporal masking belongs to strict_temporal")
    require(pose["topology"] == "upper7_hands42_v1" and pose["coordinate_system"] == "per_frame_body_isotropic", "Unsupported pose schema")
    require(not pose["rotation_canonicalization"] and not pose["include_hand_local"], "Optional pose transforms unavailable until M10")
    require(not config["evaluation"]["test_access_during_sweep"], "Test-set model selection is forbidden")
    require(train["max_successful_steps"] > 0 and train["microbatch_per_gpu"] > 0 and train["gradient_accumulation"] > 0, "Training budget/batch must be positive")
    require(data["segment_samples"] >= 2 and data["target_fps"] > 0 and cache["visual_dim"] > 0, "Invalid sample or feature dimensions")
    require(all(v >= 0 for k, v in loss.items() if k.endswith("_weight")), "Loss weights must be nonnegative")
    require(runtime["max_runtime_hours"] > 0, "Runtime budget must be positive")
    require(0 < train["learning_rate"] and 0 <= train["final_learning_rate"] <= train["learning_rate"] and train["weight_decay"] >= 0, "Invalid optimizer rates")
    require(len(train["betas"]) == 2 and all(0 <= v < 1 for v in train["betas"]) and 0 <= train["warmup_fraction"] < 1, "Invalid betas/warmup")
    require(0 <= model["teacher"]["tau_start"] <= model["teacher"]["tau_end"] <= 1, "Invalid teacher momentum")
    require(mask["retries"] > 0 and mask["min_visible_raw_tokens"] > 0, "Invalid mask retry/context limits")
    require(all(train[key] > 0 for key in ("log_interval", "diagnostic_interval", "checkpoint_interval", "validation_interval", "keep_last")), "Runtime intervals/retention must be positive")
    require(train["grad_clip_norm"] > 0 and loss["geometry_beta"] > 0, "Gradient clipping and SmoothL1 beta must be positive")
    require(cache["reader_lru_samples"] >= 0, "Cache reader capacity cannot be negative")
    choice("downstream", "task", {"islr", "cslr", "slt"})
    choice("downstream", "head", {"linear", "attentive", "finetune"})
    choice("downstream", "representation", {"temporal", "frozen_visual"})
    ds = config["downstream"]
    require(ds["layer_policy"] == "last", "Layer-mixture probing requires a separate implemented protocol")
    require(ds["max_epochs"] > 0 and ds["patience"] > 0 and ds["batch_size"] > 0, "Invalid downstream training budget")
    require(0 <= ds["overlap_samples"] < ds["window_samples"] and ds["window_samples"] >= 2, "Invalid full-clip tiling schedule")
    require(all(lr > 0 for lr in ds["learning_rates"]) and ds["encoder_learning_rate"] > 0, "Invalid downstream learning rates")


@dataclass(frozen=True)
class ResolvedConfig:
    values: dict
    root: Path
    chain: tuple[str, ...]
    hash: str

    def path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def provenance(self):
        return {"config": self.values, "config_hash": self.hash, "config_chain": list(self.chain), "project_root": str(self.root)}


def load_config(path: str | Path, project_root: str | Path | None = None) -> ResolvedConfig:
    chain = []
    values = extension_defaults(_load(Path(path), set(), chain))
    validate(values)
    # Installed packages also work with configs outside the source checkout.
    root = Path(project_root).resolve() if project_root else Path(chain[0]).parent.parent
    return ResolvedConfig(values, root, tuple(chain), canonical_hash(values))
