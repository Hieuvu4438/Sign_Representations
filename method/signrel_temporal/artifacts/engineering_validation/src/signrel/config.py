"""Strict resolved configuration; all paths are relative to the project root."""
import copy
from dataclasses import dataclass
from pathlib import Path
import math
import yaml
from .utils import fingerprint

ROOT = Path(__file__).resolve().parents[2]


def merge(parent, child):
    result = copy.deepcopy(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def reject_unknown(value, reference, path='config'):
    if not isinstance(value, dict):
        raise ValueError(f'{path} must be a mapping')
    unknown = set(value) - set(reference)
    if unknown:
        raise ValueError(f'Unknown keys at {path}: {sorted(unknown)}')
    for key, child in value.items():
        if isinstance(reference[key], dict):
            reject_unknown(child, reference[key], path+'.'+key)


@dataclass(frozen=True)
class ModelConfig:
    streams: list[str]
    hidden_dim: int
    layers: int
    heads: int
    feedforward_dim: int
    dropout: float
    norm_first: bool
    time_embedding: str
    nested_tensor: bool


@dataclass(frozen=True)
class ResolvedConfig:
    values: dict
    model: ModelConfig
    sha256: str


def validate(cfg):
    base = yaml.safe_load((ROOT/'configs/base.yaml').read_text())
    reject_unknown(cfg, base)
    model, data, teacher, train = (cfg[k] for k in ('model','data','teacher','train'))
    relation, corruption, budget = cfg['loss']['relation'], cfg['corruption'], cfg['budget']
    if cfg['schema_version'] != 1 or model['streams'] != ['LH','RH','BODY']:
        raise ValueError('Unsupported schema or stream order')
    if teacher['name'] != 'dinov2_vits14' or teacher['feature_dim'] != 384 or not teacher['frozen']:
        raise ValueError('MVP requires frozen original DINOv2 ViT-S/14')
    if teacher['feature_key'] != 'x_norm_clstoken' or teacher['normalization'] != 'l2':
        raise ValueError('Incompatible teacher feature convention')
    if model['hidden_dim'] % model['heads'] or model['hidden_dim'] % 2:
        raise ValueError('hidden_dim must be even and divisible by heads')
    if model['layers'] < 1 or model['nested_tensor'] or model['time_embedding'] != 'sinusoidal_seconds':
        raise ValueError('Unsupported encoder convention')
    if train['ddp']:
        raise ValueError('DDP disabled until exact global reductions are independently verified')
    if train['optimizer'] != 'adamw' or train['accumulation_reduction'] != 'mean_of_microbatch_weighted_means':
        raise ValueError('Unsupported optimizer/reduction')
    if train['device'] not in ('auto','cpu','cuda') or train['precision'] not in ('fp32','bf16','fp16','bf16_if_supported_else_fp32'):
        raise ValueError('Unsupported device/precision')
    for key in ('microbatch_size','gradient_accumulation','max_updates','checkpoint_every','log_every'):
        if type(train[key]) is not int or train[key] < 1:
            raise ValueError(f'train.{key} must be a positive integer')
    if data['min_frames'] < 1 or data['min_frames'] > data['window_frames']:
        raise ValueError('Invalid window length')
    if data['sampling'] != 'recording_uniform' or data['allow_caption_boundaries']:
        raise ValueError('MVP needs recording-uniform random windows without caption prior')
    if relation['mode'] not in ('cross_signed','cross_zero','intra','shuffled'):
        raise ValueError('Unsupported MVP relation mode')
    if relation['stream_pairs'] != [['LH','RH']]:
        raise ValueError('Only hand appearance relations are supported, not BODY cosine')
    if any(type(lag) is not int or abs(lag) >= data['window_frames'] for lag in relation['lags']):
        raise ValueError('Lag must be an integer inside the window')
    if len(set(relation['lags'])) != len(relation['lags']):
        raise ValueError('Duplicate lags')
    if relation['mode'] == 'cross_zero' and relation['lags'] != [0]:
        raise ValueError('A3 requires only zero lag')
    if relation['mode'] == 'intra' and any(lag <= 0 for lag in relation['lags']):
        raise ValueError('Intra-stream lags must be positive and nonzero')
    if not relation['require_masked_endpoint'] or not relation['reject_duplicate_source_frame_for_nonzero_lag']:
        raise ValueError('MVP relation eligibility cannot be disabled')
    if relation['sampling'] != 'lag_stratified' or relation['max_pairs_per_sample'] < 1:
        raise ValueError('Unsupported pair budget')
    if cfg['loss']['hand_reduction'] != 'squared_l2_sum_then_stream_mean':
        raise ValueError('Hand loss scale convention must remain explicit')
    for value in (cfg['loss']['body_smooth_l1_beta'], relation['smooth_l1_beta']):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Smooth L1 beta must be positive')
    for value in (cfg['loss']['body_weight'], relation['max_weight']):
        if not math.isfinite(value) or value < 0:
            raise ValueError('Loss weights must be finite and nonnegative')
    for value in (data['quality_threshold'], corruption['hand_mask_ratio'], corruption['body_mask_ratio'],
                  corruption['shared_span_probability'], model['dropout']):
        if not 0 <= value <= 1:
            raise ValueError('Quality, ratios and dropout must be in [0,1]')
    if any(corruption[k] for k in ('time_resample','horizontal_flip','temporal_reverse','whole_hand_dropout')):
        raise ValueError('Optional corruptions require a separately implemented experiment')
    if not 1 <= corruption['span_min'] <= corruption['span_max'] or corruption['minimum_visible_tokens'] < 4:
        raise ValueError('Invalid span/visibility policy')
    if budget['allow_paid_cloud'] or (budget['phase'] == 'full_scale' and not budget['allow_full_scale']):
        raise ValueError('Paid cloud prohibited; full-scale budget not authorized')
    if budget['max_wall_minutes'] < 0 or budget['max_gpu_hours'] < 0:
        raise ValueError('Negative resource cap')
    if cfg['eval']['pooling'] != 'physical_time_pyramid_mean' or cfg['eval']['temporal_bins'] != [1,2]:
        raise ValueError('MVP requires fixed whole/first-half/second-half pyramid')
    return ResolvedConfig(cfg, ModelConfig(**model), fingerprint(cfg))


def load_config(path, _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError('Cyclic config inheritance')
    seen.add(path)
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError('Config must be a mapping')
    parent = cfg.pop('extends',None)
    if parent:
        cfg = merge(load_config(path.parent/parent,seen).values,cfg)
    return validate(cfg)
