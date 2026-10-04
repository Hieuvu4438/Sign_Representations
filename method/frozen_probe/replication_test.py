"""P004 held-out freeze, preserving original scorer/pooling/timing methods."""
import json
from pathlib import Path

from cache import file_sha
from confirmatory import METHODS, SEEDS

REPO = Path(__file__).resolve().parents[2]


def validate_test_config(config):
    if (config['protocol'] != 'P004_v1' or config['H_id'] != 'H002'
            or config['gallery_budget'] != 1 or config['methods'] != list(METHODS)
            or config['support_seeds'] != list(SEEDS) or config['dcg_loss_margin'] != .02
            or config['min_window_reduction'] != 2. or config['cost_ratio_floor'] != .8
            or config['new_policy_trials'] != 0
            or config['project_gpu_ceiling_seconds'] != 14400
            or config['hypothesis_gpu_ceiling_seconds'] != 7200):
        raise ValueError('Fixed P004 held-out scientific specification changed')
    return config


def check_test_freeze(gate_path, config_path, repo=REPO):
    gate = json.loads(Path(gate_path).read_text())
    name = str(Path(config_path).resolve().relative_to(repo))
    if (gate['verdict'] != 'FROZEN_REPLICATION_BEFORE_SELECTED_TEST_ACCESS'
            or gate['protocol'] != 'P004_v1' or not gate['fixed_dev_pilot_passed']
            or gate['config_hashes'].get(name) != file_sha(config_path)):
        raise ValueError('Unissued/unbound P004 held-out freeze')
    validate_test_config(json.loads(Path(config_path).read_text()))
    for name, digest in gate['file_hashes'].items():
        if file_sha(repo / name) != digest:
            raise ValueError(f'Frozen P004 source/artifact changed: {name}')
    return gate
