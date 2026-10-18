from copy import deepcopy
from pathlib import Path
import json
import torch
from signrepr.data.fixture import make_fixture
from signrepr.data.schema import read_manifest, write_manifest
from signrepr.training.pretrain import SSLTrainer
from signrepr.training.checkpoints import save_training
from signrepr.training.islr import train_islr, eval_islr
from signrepr.training.sequence_tasks import train_sequence
from signrepr.cli import main
import yaml


def supervised_config(config, tmp_path):
    cfg = deepcopy(config)
    train = make_fixture(tmp_path/'train', 8, seed=42)
    validation = make_fixture(tmp_path/'validation', 8, seed=75)
    rows = read_manifest(validation)
    for row in rows:
        row['sample_id'] = row['sample_id'].replace('/train/', '/val/')
        row['source_video_id'] += '_validation'
        row['split'] = 'val'
    write_manifest(validation, rows)
    cfg['data'].update(train_manifest=str(train), validation_manifest=str(validation))
    cfg['run']['output_dir'] = str(tmp_path/'out')
    cfg['downstream'] = {'frozen': True, 'epochs': 2, 'batch_size': 4, 'head_lr': .001, 'encoder_lr': .0001,
                         'weight_decay': .01, 'warmup_epochs': 0, 'label_smoothing': .1, 'selection_metric': 'Top1_macro',
                         'patience': 2, 'hidden_dim': 8}
    return cfg


def test_frozen_and_finetuned_islr_end_to_end(config, batch, tmp_path):
    ssl = SSLTrainer(config); ssl.update([batch]); save_training(ssl, tmp_path/'ssl.pt')
    cfg = supervised_config(config, tmp_path)
    result = train_islr(cfg, tmp_path/'ssl.pt')
    state = torch.load(result['checkpoint'], weights_only=False)
    for k, v in ssl.teacher.state_dict().items():
        torch.testing.assert_close(v, state['encoder'][k], atol=0, rtol=0)
    eval_result = eval_islr(result['checkpoint'], cfg['data']['validation_manifest'])
    assert eval_result['metrics']['num_samples'] == 8
    assert eval_result['metrics']['valid_samples'] == 8
    cfg['downstream']['frozen'] = False
    cfg['run']['output_dir'] = str(tmp_path/'finetune')
    result = train_islr(cfg, tmp_path/'ssl.pt')
    fine = torch.load(result['checkpoint'], weights_only=False)
    assert any(not torch.equal(v, fine['encoder'][k]) for k, v in ssl.teacher.state_dict().items())


def test_cslr_complete_frozen_workflow(config, batch, tmp_path):
    ssl = SSLTrainer(config); ssl.update([batch]); save_training(ssl, tmp_path/'ssl.pt')
    cfg = supervised_config(config, tmp_path)
    result = train_sequence(cfg, tmp_path/'ssl.pt', 'cslr')
    assert result['task'] == 'cslr'
    state = torch.load(result['checkpoint'], weights_only=False)
    assert state['validation']['num_utterances'] == 8
    assert state['validation']['reference_tokens'] == 8
    assert (Path(cfg['run']['output_dir'])/'feature_cache_provenance.json').exists()
    import numpy as np
    with np.load(Path(cfg['run']['output_dir'])/'feature_cache/train/00000000.npz') as cache:
        assert int(cache['length']) == len(cache['features'])
        assert bool(cache['timing_known'])
        assert cache['source_frame_indices'].shape == (int(cache['length']), 2)


def test_cslr_finetune_preserves_full_alignment_and_disables_cache(config, batch, tmp_path):
    ssl = SSLTrainer(config); ssl.update([batch]); save_training(ssl, tmp_path/'ssl.pt')
    cfg = supervised_config(config, tmp_path)
    cfg['downstream'].update(frozen=False, epochs=1, encoder_lr=1e-5)
    result = train_sequence(cfg, tmp_path/'ssl.pt', 'cslr')
    state = torch.load(result['checkpoint'], weights_only=False)
    assert not state['cache_provenance']['cache_enabled']
    assert not (Path(cfg['run']['output_dir'])/'feature_cache').exists()
    assert any(not torch.equal(value, state['encoder'][key]) for key, value in ssl.teacher.state_dict().items())
    from signrepr.export.api import export_checkpoint, SignRepresentationModel
    info = export_checkpoint(result['checkpoint'], tmp_path/'fine_bundle')
    assert info['method'] == 'pose_cslr_supervised_encoder'
    model = SignRepresentationModel(tmp_path/'fine_bundle')
    for key, value in state['encoder'].items():
        torch.testing.assert_close(value, model.encoder.state_dict()[key], atol=0, rtol=0)


def test_metric_cli_warmup_joint_validation(config, batch, tmp_path):
    ssl = SSLTrainer(config); ssl.update([batch]); save_training(ssl, tmp_path/'ssl.pt')
    cfg = supervised_config(config, tmp_path)
    cfg['run'].update(stage='metric', supervision='gloss_supervised_metric_adaptation')
    cfg['metric'] = {'head_warmup_steps': 1, 'joint_steps': 1, 'head_lr': 1e-4, 'encoder_lr': 2e-5,
                     'weight_decay': .01, 'ssl_replay': True, 'temperature': .07, 'lr_warmup_steps': 1,
                     'classes_per_batch': 4, 'clips_per_class': 2, 'validation_interval_steps': 2}
    cfg_path = tmp_path/'config.yaml'; cfg_path.write_text(yaml.safe_dump(cfg))
    main(['train_metric', '--config', str(cfg_path), '--init', str(tmp_path/'ssl.pt'), '--device', 'cpu', '--cpu-threads', '1'])
    root = Path(cfg['run']['output_dir'])
    assert (root/'best_validation.pt').exists()
    report = json.loads((root/'validation_00000002.json').read_text())
    assert report['metrics']['num_queries'] == 8
    state = torch.load(root/'last.pt', weights_only=False)
    assert state['step'] == 2 and state['metadata']['stage'] == 'metric'
