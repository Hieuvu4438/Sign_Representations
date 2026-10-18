from pathlib import Path
import torch
from signrepr.cli import main
from signrepr.export.api import export_checkpoint, SignRepresentationModel
from signrepr.training.pretrain import SSLTrainer
from signrepr.training.checkpoints import save_training
from signrepr.data.fixture import fixture_pose
from signrepr.data.normalization import canonicalize_sample
from signrepr.evaluation.extract import extract_full_sequence


def test_bundle_cpu_parity_short_odd_long(config, batch, tmp_path):
    trainer = SSLTrainer(config)
    trainer.update([batch])
    save_training(trainer, tmp_path/'last.pt')
    export_checkpoint(tmp_path/'last.pt', tmp_path/'bundle')
    model = SignRepresentationModel(tmp_path/'bundle')
    for t in (1, 5, 139):
        pose = fixture_pose(t)
        ref = extract_full_sequence(trainer.teacher, canonicalize_sample(pose))
        seq = model.encode_sequence(pose)
        assert seq['length'] == (t+1)//2
        torch.testing.assert_close(ref['features'], seq['features'], atol=0, rtol=0)
        torch.testing.assert_close(ref['raw_embedding'], model.encode_clip(pose)['embedding'], atol=0, rtol=0)
        assert seq['source_frame_indices'][-1, 0] == (t-1)//2*2
    assert model.encode_clip(fixture_pose(5, variant='all_missing'))['status'] == 'normalization_failed'
    import pytest
    with pytest.raises(ValueError, match='not been trained'):
        model.encode_clip(fixture_pose(), embedding='metric')
    assert model.encode_sequence(fixture_pose(), {'joint_layout': 'unknown'})['status'] == 'malformed_input'
    assert model.encode_sequence({'xy': []})['status'] == 'malformed_input'
    assert model.encode_sequence(None)['status'] == 'malformed_input'


def test_cli_help_fixture_and_audit(tmp_path):
    main(['make_fixture', '--output', str(tmp_path/'fixture')])
    main(['validate_data', '--manifest', str(tmp_path/'fixture/manifest.jsonl'), '--report', str(tmp_path/'report.json')])
    assert (tmp_path/'report.json').exists()


def test_extract_and_retrieval_cli_independent_bundle_loads(config, batch, tmp_path):
    import json
    from signrepr.data.fixture import make_fixture
    path = make_fixture(tmp_path/'fixture', 8)
    trainer = SSLTrainer(config); trainer.update([batch])
    save_training(trainer, tmp_path/'last.pt')
    for manifest, output in [(path, 'gallery'), (tmp_path/'fixture/test.jsonl', 'queries')]:
        main(['extract', '--checkpoint', str(tmp_path/'last.pt'), '--manifest', str(manifest), '--output', str(tmp_path/output), '--device', 'cpu'])
    main(['retrieval', '--gallery', str(tmp_path/'gallery'), '--queries', str(tmp_path/'queries'), '--protocol', 'gloss_max_single_relevant', '--output', str(tmp_path/'retrieval.json')])
    assert json.loads((tmp_path/'retrieval.json').read_text())['metrics']['num_queries'] == 8
