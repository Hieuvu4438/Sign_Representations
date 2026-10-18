from copy import deepcopy
from signrepr.training.pretrain import SSLTrainer


def test_reconstruction_control_skips_teacher_and_reports_applied_momentum(config, batch):
    cfg = deepcopy(config)
    cfg['loss'].update(latent_weight=0, geometry_weight=1, motion_weight=0)
    trainer = SSLTrainer(cfg)
    def forbidden_teacher_forward(*args, **kwargs):
        raise AssertionError('Reconstruction-only control must not execute a teacher forward')
    trainer.teacher.encode_full = forbidden_teacher_forward
    report = trainer.update([batch])
    assert report['successful']
    assert report['ema_momentum'] == 0
    assert report['raw_losses']['latent'] == 0
    assert not trainer.criterion.center.any()
