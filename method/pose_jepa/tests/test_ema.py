from copy import deepcopy
import torch
from signrepr.models.pose_encoder import PoseEncoder
from signrepr.models.ema import update_encoder_ema


def test_shared_hand_ema_once(config):
    student = PoseEncoder(config['model'])
    teacher = deepcopy(student)
    assert student.tokenizer.frontends['left_hand'] is student.tokenizer.frontends['right_hand']
    assert teacher.tokenizer.frontends['left_hand'] is teacher.tokenizer.frontends['right_hand']
    with torch.no_grad():
        for p in student.parameters(): p.fill_(2)
        for p in teacher.parameters(): p.fill_(0)
    update_encoder_ema(student, teacher, .5)
    for p in teacher.parameters():
        assert p.eq(1).all()


def test_distinct_layer_initialization(config):
    model = PoseEncoder(config['model'])
    assert not torch.equal(model.temporal.layers[0].linear1.weight, model.temporal.layers[1].linear1.weight)
