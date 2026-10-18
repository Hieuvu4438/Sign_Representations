from pathlib import Path
from copy import deepcopy
import pytest
from signrepr.config import load_config, validate_config, load_protocol_config
from signrepr.cli import parser


def test_all_task_and_control_configs_validate():
    for path in (Path(__file__).parents[1]/'configs').rglob('*.yaml'):
        if path.name == 'data_protocol.yaml': load_protocol_config(path)
        else: load_config(path)
    choices = parser()._subparsers._group_actions[0].choices
    assert {'make_fixture', 'import_unisign', 'validate_data', 'build_protocol', 'pretrain', 'extract', 'retrieval',
            'train_islr', 'eval_islr', 'train_metric', 'train_cslr', 'train_slt', 'export'} <= set(choices)


def test_invalid_config_fails_explicitly(config):
    cfg = deepcopy(config); cfg['data']['joint_layout'] = 'unknown'
    with pytest.raises(ValueError, match='joint convention'): validate_config(cfg)
    cfg = deepcopy(config); cfg['loss']['latent_objective'] = 'unknown'
    with pytest.raises(ValueError, match='latent objective'): validate_config(cfg)
    cfg = deepcopy(config); cfg['model']['input_channels'] = 4
    with pytest.raises(ValueError, match='input_channels'): validate_config(cfg)
