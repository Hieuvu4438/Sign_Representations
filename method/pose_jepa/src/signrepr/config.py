"""Validated recipe configuration. Paths are relative to the YAML directory."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import yaml


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def validate_config(cfg):
    cfg = deepcopy(cfg)
    m, d, loss, mask, opt = (cfg[k] for k in ("model", "data", "loss", "mask", "optim"))
    fixed = [(d, 'normalization', 'clip_median_shoulders'), (d, 'canonicalization_scope', 'sample_before_windowing'),
             (d, 'time_mode', 'timestamps_if_known_else_frame_index'), (d, 'motion_requires_known_timing', True),
             (d, 'target_fps', None), (m, 'input_channels', 3), (m, 'encoder_pre_norm', True),
             (m, 'graph_partitions', ['self', 'undirected_neighbor']),
             (m, 'spatial_normalization', 'per_joint_channel_layernorm'),
             (loss, 'geometry_exclude_local_roots', True), (loss, 'center_quality_weighted', True),
             (opt, 'optimizer', 'adamw')]
    for section, key, expected in fixed:
        if section.get(key) != expected:
            raise ValueError(f'Unsupported baseline configuration {key}={section.get(key)!r}')
    if d["joint_layout"] != "coco_wholebody_133" or d["part_order"] != ["body", "left_hand", "right_hand", "face"]:
        raise ValueError("Unsupported joint convention/part order")
    if d["segment_length"] != 2 or d["max_frames"] < 2 or d["max_frames"] % 2:
        raise ValueError("Segment length must be 2, window must be positive and aligned")
    if d["coordinate_profile"] not in ("viewport_normalized", "aspect_corrected"):
        raise ValueError("Unknown coordinate profile")
    if d.get('sampling', 'continuous') not in ('continuous', 'isolated'):
        raise ValueError('SSL sampling must be continuous windows or whole-clip isolated subsampling')
    for kind in ("encoder", "predictor"):
        if m[f'{kind}_heads'] < 1 or m[f'{kind}_dim'] < 1 or m[f"{kind}_dim"] % m[f"{kind}_heads"] or m[f"{kind}_layers"] < 1:
            raise ValueError("Invalid Transformer dimensions/layers")
    if m["latent_dim"] != m["encoder_dim"] or m["sequence_dim"] != 4 * m["encoder_dim"]:
        raise ValueError("Latent/sequence dimension mismatch")
    if m["dropout"] != 0 or m["hand_canonical_flip"] or not m["share_hand_frontend"]:
        raise ValueError("Main recipe requires dropout=0, shared hands, no flip")
    if m["init_checkpoint"] is not None:
        raise ValueError("Stage 1 scratch recipe cannot load unrelated external weights")
    if not 0 <= mask["ratio"] < 1 or len(mask["part_weights"]) != 4 or any(w <= 0 for w in mask["part_weights"]):
        raise ValueError("Invalid mask budget/part weights")
    if not 0 <= mask["span_budget_fraction"] <= 1 or not 1 <= mask["min_span_segments"] <= mask["max_span_segments"]:
        raise ValueError("Invalid span configuration")
    if mask["motion_aware"] or mask["learned_saliency"]:
        raise ValueError("Saliency extensions are outside the baseline")
    if any(loss[k] < 0 for k in ("latent_weight", "geometry_weight", "motion_weight")):
        raise ValueError("Loss weights must be nonnegative")
    if loss["student_temperature"] <= 0 or loss["teacher_temperature"] <= 0:
        raise ValueError("Temperatures must be positive")
    if not 0 <= loss["center_momentum"] <= 1:
        raise ValueError("Invalid center momentum")
    if loss.get('latent_objective', 'ce') not in ('ce', 'regression'):
        raise ValueError('Unknown latent objective')
    if any(m.get(k) is not None and m[k] < 1 for k in ('encoder_dim', 'predictor_dim', 'latent_dim')):
        raise ValueError('Invalid model width')
    if not 0 <= d['confidence_threshold'] <= 1 or not 0 < d['min_part_valid_fraction'] <= 1:
        raise ValueError('Invalid confidence/validity thresholds')
    if any(loss[k] <= 0 for k in ('geometry_huber_beta', 'motion_huber_beta', 'motion_reference_fps')):
        raise ValueError('Physical beta/reference FPS must be positive')
    if m['joint_aggregation'] not in ('flatten_project', 'mean'):
        raise ValueError('Unknown joint aggregation')
    if cfg['run']['protocol'] not in ('cross_corpus_transfer', 'in_domain_ssl'):
        raise ValueError('Unknown training protocol')
    if any(loss.get(k, 0) != 0 for k in ("dense_context_weight", "variance_weight", "covariance_weight")):
        raise ValueError("Unimplemented extension objective")
    if opt["gradient_accumulation_steps"] < 1 or opt["per_rank_batch"] < 1 or opt["max_steps"] < 1:
        raise ValueError("Invalid update/batch budget")
    if d['max_motion_gap_sec'] <= 0 or d['num_workers'] < 0 or opt['warmup_steps'] < 0 or opt['learning_rate'] <= 0:
        raise ValueError('Invalid timing/worker/learning-rate configuration')
    if cfg.get('extensions') and any(cfg['extensions'].values()):
        raise ValueError('Optional modality/regularization extensions are not enabled in this baseline')
    if 'metric' in cfg:
        metric = cfg['metric']
        if metric['head_warmup_steps'] < 0 or metric['joint_steps'] < 1 or metric['classes_per_batch'] < 1 or metric['clips_per_class'] < 2 or metric['temperature'] <= 0:
            raise ValueError('Invalid metric stage budget/sampler/temperature')
    if 'downstream' in cfg and any(cfg['downstream'][k] < 1 for k in ('epochs', 'batch_size', 'patience')):
        raise ValueError('Invalid downstream training budget')
    return cfg


def load_config(path):
    path = Path(path).resolve()
    cfg = validate_config(yaml.safe_load(path.read_text()))
    for key in ("train_manifest", "validation_manifest", "downstream_validation_protocol"):
        if cfg["data"].get(key):
            cfg["data"][key] = str((path.parent / cfg["data"][key]).resolve())
    cfg["run"]["output_dir"] = str((path.parent / cfg["run"]["output_dir"]).resolve())
    return cfg


def load_protocol_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if cfg.get('protocol') not in ('cross_corpus_transfer', 'in_domain_ssl') or cfg.get('model_selection_scope') not in ('target_validation', 'source_validation'):
        raise ValueError('Invalid protocol/model selection scope')
    if cfg['protocol'] == 'cross_corpus_transfer' and not cfg.get('target_corpora'):
        raise ValueError('Cross-corpus protocol requires explicit target corpora')
    cfg['manifest'] = str((path.parent/cfg['manifest']).resolve())
    return cfg
