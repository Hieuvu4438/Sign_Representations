from dataclasses import replace
import numpy as np
import pytest
import torch

from signpath.data.collate import collate_examples
from signpath.losses.reduction import global_group_mean
from signpath.models.teacher_student import ResearchModel
from signpath.training.ema import update_teacher
from model_helpers import batch_from, example, tiny_config


@pytest.fixture(autouse=True)
def thread_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_teacher_copy_and_gradient_policy(bundle):
    torch.manual_seed(4)
    model = ResearchModel(tiny_config()).train()
    assert not model.teacher.training
    for a, b in zip(model.online.parameters(), model.teacher.parameters()):
        assert a is not b and torch.equal(a, b) and not b.requires_grad
    batch = batch_from(bundle)
    output = model(batch)
    assert output.global_semantic_count > 0 and output.global_geometry_count > 0
    output.total.backward()
    assert all(p.grad is None for p in model.teacher.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.online.parameters())
    assert sum(p.grad.abs().sum() for p in model.online.parameters()) > 0
    assert sum(p.grad.abs().sum() for p in model.predictor.geometry_head.parameters()) > 0


def test_hidden_values_cannot_change_student_or_query_but_teacher_changes(bundle):
    torch.manual_seed(4)
    model = ResearchModel(tiny_config()).eval()
    batch = batch_from(bundle)
    streams = tuple(torch.where(batch.visible[:, :, s, None], values, values + 1000) for s, values in enumerate(batch.streams))
    modified = replace(batch, streams=streams, geometry_values=batch.geometry_values + 987)
    with torch.no_grad():
        first = model.online(batch, batch.visible)
        other = model.online(modified, modified.visible)
        torch.testing.assert_close(first.tokens, other.tokens, atol=0, rtol=0)
        torch.testing.assert_close(model.predictor.query_metadata(batch), model.predictor.query_metadata(modified), atol=0, rtol=0)
        predictions = model.predictor(first, batch)
        changed = model.predictor(other, modified)
        torch.testing.assert_close(predictions.semantic, changed.semantic, atol=0, rtol=0)
        torch.testing.assert_close(predictions.geometry, changed.geometry, atol=0, rtol=0)
        assert not torch.allclose(model.teacher(batch, batch.valid).tokens, model.teacher(modified, modified.valid).tokens)


def test_padding_and_dense_pooling_are_invariant(bundle):
    torch.manual_seed(6)
    model = ResearchModel(tiny_config()).eval()
    first = batch_from(bundle)
    padded = batch_from(bundle, pad_to=64)
    # Invalid values are arbitrary and must not influence real output.
    padded = replace(padded, streams=tuple(torch.where(padded.valid[:, :, s, None], x, 12345) for s, x in enumerate(padded.streams)))
    with torch.no_grad():
        a, b = model.extract_embeddings(first), model.extract_embeddings(padded)
        torch.testing.assert_close(a.dense, b.dense[:, :48], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(a.clip, b.clip, atol=1e-6, rtol=1e-6)
        assert not b.valid[:, 48:].any() and not b.dense[:, 48:].any()
        torch.testing.assert_close(model(first).total, model(padded).total, atol=1e-6, rtol=1e-6)


def test_empty_targets_are_differentiable_and_empty_context_is_finite(bundle):
    model = ResearchModel(tiny_config())
    batch = batch_from(bundle)
    empty = replace(batch, visible=torch.zeros_like(batch.visible), semantic_targets=torch.zeros_like(batch.semantic_targets),
                    geometry_targets=torch.zeros_like(batch.geometry_targets))
    result = model(empty)
    assert result.total.requires_grad and float(result.total.detach()) == 0
    assert result.global_semantic_count == result.global_geometry_count == 0
    result.total.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_global_type_loss_matches_hand_calculation():
    errors = torch.tensor([[1., 3., 8., 999.]], requires_grad=True)
    output = global_group_mean(errors, torch.tensor([[0, 0, 1, 2]]), torch.tensor([[True, True, True, False]]), 3)
    assert output.loss.item() == 5  # ((1+3)/2 + 8)/2
    output.loss.backward()
    torch.testing.assert_close(errors.grad, torch.tensor([[.25, .25, .5, 0.]]))
    assert output.counts.tolist() == [2, 1, 0]


def test_ema_updates_parameters_and_buffers_without_predictor():
    online = torch.nn.BatchNorm1d(3)
    teacher = torch.nn.BatchNorm1d(3).requires_grad_(False).eval()
    online.weight.data.fill_(3)
    online.running_mean.fill_(4)
    online.num_batches_tracked.fill_(7)
    update_teacher(online, teacher, .75)
    torch.testing.assert_close(teacher.weight, torch.full((3,), 1.5))
    torch.testing.assert_close(teacher.running_mean, torch.full((3,), 1.))
    assert teacher.num_batches_tracked.item() == 7


def test_cluster_only_avoids_teacher_forward(bundle):
    config = tiny_config()
    config["geometry"]["enabled"] = False
    config["loss"].update(geometry_weight=0., semantic_weight=0., cluster_weight=1.)
    model = ResearchModel(config)
    batch = batch_from(bundle, geometry=False)
    labels = torch.arange(48)[None, :, None].expand(1, 48, 7).remainder(256)
    batch = replace(batch, cluster_ids=labels)
    def forbidden(*args, **kwargs):
        raise AssertionError("Cluster-only baseline must not run EMA teacher")
    model.teacher.forward = forbidden
    output = model(batch)
    output.total.backward()
    assert torch.isfinite(output.total) and output.global_semantic_count > 0


def test_unsupported_variant_fails_explicitly():
    config = tiny_config()
    config["geometry"]["role"] = "input_enabled"
    with pytest.raises(ValueError, match="M10"):
        ResearchModel(config)


def test_paired_config_cannot_be_mislabeled_as_sign_only_runtime():
    config = tiny_config()
    config["experiment"]["supervision_regime"] = "paired"
    with pytest.raises(ValueError, match="own runtime"):
        ResearchModel(config)


def test_static_hand_signal_reaches_encoder_even_when_motion_signature_collides(bundle):
    from dataclasses import replace
    from signpath.geometry.logsig2 import logsig2_reference
    from model_helpers import batch_from
    xy = np.broadcast_to(bundle.pose_xy_body[0], bundle.pose_xy_body.shape).copy()
    other = xy.copy()
    other[:, 7:28, 0] += .3
    other[:, 15, 1] += .2
    np.testing.assert_allclose(logsig2_reference(xy[:, 7]), logsig2_reference(other[:, 7]), atol=1e-10)
    first = batch_from(replace(bundle, pose_xy_body=xy), geometry=False)
    second = batch_from(replace(bundle, pose_xy_body=other), geometry=False)
    torch.manual_seed(7)
    model = ResearchModel(tiny_config()).eval()
    with torch.no_grad():
        assert not torch.allclose(model.extract_embeddings(first).clip, model.extract_embeddings(second).clip, atol=1e-6)
