from copy import deepcopy
import torch
from signrepr.training.pretrain import SSLTrainer
from signrepr.training.checkpoints import save_training, load_training


def test_exact_resume_and_old_center_per_update(config, batch, tmp_path):
    uninterrupted = SSLTrainer(config)
    assert uninterrupted.update([batch])['successful']
    save_training(uninterrupted, tmp_path/'last.pt', {'cursor': 2})
    report = uninterrupted.update([batch, batch])
    resumed = SSLTrainer(config)
    cursor, _ = load_training(resumed, tmp_path/'last.pt')
    assert cursor == {'cursor': 2}
    other = resumed.update([batch, batch])
    report.pop('update_duration_sec'); other.pop('update_duration_sec')
    assert report == other
    for module in ('system', 'teacher', 'criterion'):
        for k, value in getattr(uninterrupted, module).state_dict().items():
            torch.testing.assert_close(value, getattr(resumed, module).state_dict()[k], atol=0, rtol=0)
    assert resumed.step == 2


def test_zero_support_and_nonfinite_skip_leave_state_unchanged(config, batch):
    trainer = SSLTrainer(config)
    empty = deepcopy(batch)
    for stream in empty['parts'].values():
        stream['frame_valid'].fill_(False)
        stream['joint_valid'].fill_(False)
    before = deepcopy(trainer.system.state_dict())
    result = trainer.update([empty])
    assert result['reason'] == 'zero_support' and trainer.step == 0
    assert not trainer.criterion.center.any()
    # Inject overflow in a trainable component; every training state must skip.
    hook = trainer.system.predictor.latent.weight.register_hook(lambda g: g*float('inf'))
    result = trainer.update([batch])
    hook.remove()
    assert result['reason'] == 'nonfinite_gradients' and trainer.step == 0
    for k, value in trainer.system.state_dict().items():
        torch.testing.assert_close(value, before[k], atol=0, rtol=0)
