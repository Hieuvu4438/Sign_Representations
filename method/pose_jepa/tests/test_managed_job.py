from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import pytest
import torch
import yaml
from signrepr.data.fixture import make_fixture
from signrepr.data.pose_dataset import PoseDataset
from signrepr.training.pretrain import SSLTrainer
from signrepr.training.stream import SampleStream
from signrepr.training.job_support import durable_checkpoint, resume_latest
from signrepr.evaluation.loso import loso_metrics


def test_legacy_failure_adapter_keeps_queries_and_excludes_invalid_gallery():
    import importlib.util
    path = Path(__file__).parents[1]/'scripts/evaluate_aslc3000_job.py'
    spec = importlib.util.spec_from_file_location('managed_eval', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    features = np.array([[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]])
    labels = np.array([0, 1, 0, 1, 2])
    ranks = module.failure_ranks(features, labels, np.array([2, 3, 4]), np.array([0, 1]),
                                 np.array([True, False, True, False, True]), 3)
    np.testing.assert_array_equal(ranks, [1., np.inf, np.inf])
    # All-valid inputs preserve the old evaluator's pessimistic ties.
    ranks = module.failure_ranks(features, labels, np.array([2, 3]), np.array([0, 1]), np.ones(5, bool), 3)
    np.testing.assert_array_equal(ranks, [2., 2.])


def test_loso_different_signer_ties_and_failure_denominator():
    result = loso_metrics([[1.,0.],[1.,0.],[0.,1.],[0.,0.]], ['a','a','b','a'], ['p1','p2','p2','p3'], ['1','2','3','4'])
    assert result['metrics']['num_queries'] == 4
    assert result['metrics']['R@1'] == .5
    assert result['queries'][2]['failure'] == 'no_cross_signer_positive'
    assert result['queries'][3]['failure'] == 'invalid_query_embedding'


def test_checkpoint_corruption_fallback_and_provenance_rejection(config, tmp_path):
    path = make_fixture(tmp_path/'data')
    ds = PoseDataset(path, config['data'])
    stream = SampleStream(ds, 2, 5)
    trainer = SSLTrainer(config)
    metadata = {'code_sha256':'source', 'selection_protocol_sha256':'selection'}
    durable_checkpoint(trainer, stream, tmp_path/'train', metadata)
    trainer.update([stream.next()])
    durable_checkpoint(trainer, stream, tmp_path/'train', metadata)
    (tmp_path/'train/last.pt').write_bytes(b'invalid checkpoint')
    restored = SSLTrainer(config); new_stream = SampleStream(ds, 2, 5)
    _, recovery = resume_latest(restored, new_stream, tmp_path/'train', metadata)
    assert restored.step == 0 and recovery['checkpoint'].endswith('previous.pt')
    assert recovery['fallback_errors']
    with pytest.raises(ValueError, match='code_sha256'):
        resume_latest(restored, new_stream, tmp_path/'train', {**metadata, 'code_sha256':'changed'})
    durable_checkpoint(restored, new_stream, tmp_path/'train', metadata)
    assert torch.load(tmp_path/'train/previous.pt', weights_only=False)['step'] == 0
    stream.close(); new_stream.close()


def test_managed_sigterm_saves_and_resume_continues(config, tmp_path):
    cfg = deepcopy(config)
    manifest = make_fixture(tmp_path/'data', 16)
    cfg['data'].update(train_manifest=str(manifest), validation_manifest=str(tmp_path/'data/val.jsonl'), num_workers=0)
    cfg['run']['output_dir'] = str(tmp_path/'train')
    cfg['optim']['max_steps'] = 10000
    cfg['evaluation'].update(checkpoint_interval_steps=1000, frozen_validation_interval_steps=10000)
    path = tmp_path/'config.yaml'; path.write_text(yaml.safe_dump(cfg))
    command = [sys.executable,'-m','signrepr.training.managed','--config',str(path),'--job-root',str(tmp_path),'--device','cpu','--cpu-threads','1']
    env = os.environ.copy(); env['PYTHONPATH'] = str(Path(__file__).parents[1]/'src')
    with (tmp_path/'log').open('w') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+30
            while time.monotonic() < deadline:
                progress = tmp_path/'progress.json'
                if progress.exists() and json.loads(progress.read_text())['step'] >= 3: break
                if child.poll() is not None: raise AssertionError((tmp_path/'log').read_text())
                time.sleep(.05)
            else: raise AssertionError('Managed trainer did not start')
            child.send_signal(signal.SIGTERM)
            assert child.wait(timeout=20) == 0, (tmp_path/'log').read_text()
        finally:
            if child.poll() is None: child.kill(); child.wait()
    state = torch.load(tmp_path/'train/last.pt', weights_only=False)
    previous_step = state['step']
    assert previous_step >= 3 and json.loads((tmp_path/'progress.json').read_text())['status'] == 'paused'
    # Pause via stop-file after startup. The loaded counter must be preserved,
    # even though no additional optimizer update is allowed to occur.
    (tmp_path/'pause.request').touch()
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
    recovered = json.loads((tmp_path/'train/resume.json').read_text())
    assert recovered['step'] == previous_step
    assert torch.load(tmp_path/'train/last.pt', weights_only=False)['step'] == previous_step


def test_validation_interruption_at_final_step_is_replayed_without_retraining(config, tmp_path, monkeypatch):
    from signrepr.training import managed
    cfg = deepcopy(config)
    manifest = make_fixture(tmp_path/'data')
    cfg['data'].update(train_manifest=str(manifest), validation_manifest=str(tmp_path/'data/val.jsonl'), num_workers=0)
    cfg['run']['output_dir'] = str(tmp_path/'train')
    cfg['optim']['max_steps'] = 1
    cfg['evaluation'].update(checkpoint_interval_steps=1, frozen_validation_interval_steps=1)
    path = tmp_path/'config.yaml'; path.write_text(yaml.safe_dump(cfg))
    def interrupted(*args, **kwargs):
        raise RuntimeError('simulated validation interruption')
    monkeypatch.setattr(managed, 'validate_loso', interrupted)
    with pytest.raises(RuntimeError, match='simulated'):
        managed.run(path, tmp_path, device='cpu', cpu_threads=1)
    before = torch.load(tmp_path/'train/last.pt', weights_only=False)
    assert before['step'] == 1 and not (tmp_path/'train/complete.json').exists()
    monkeypatch.setattr(managed, 'validate_loso', lambda *a, **kw: {'metrics': {'R@1': .5}, 'queries': []})
    result = managed.run(path, tmp_path, device='cpu', cpu_threads=1)
    after = torch.load(tmp_path/'train/last.pt', weights_only=False)
    assert result['status'] == 'complete' and after['step'] == 1
    assert (tmp_path/'train/best_validation.pt').exists()
    for key, value in before['system'].items():
        torch.testing.assert_close(value, after['system'][key], atol=0, rtol=0)
