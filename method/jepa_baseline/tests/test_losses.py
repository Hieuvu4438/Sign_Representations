import pytest
import torch
from signrep_baseline.losses import reconstruction_loss, feature_regularization, adversarial_loss
from signrep_baseline.discriminator import (gram_style, style_pairs, StyleDiscriminator,
                                           MatchEMA, frozen_discriminator, discriminator_loss)
from signrep_baseline.schema import LOSS_WEIGHTS


def test_masked_scaled_smoothl1_oracle_and_zero_gradient():
    predicted = {key: torch.zeros(1, 2, requires_grad=True) for key in LOSS_WEIGHTS}
    targets = {key: torch.zeros(1, 2) for key in LOSS_WEIGHTS}
    masks = {key: torch.zeros(1, 2, dtype=torch.bool) for key in LOSS_WEIGHTS}
    targets["right_pose"][:] = torch.tensor([.25, 900.])
    targets["right_dist"][:] = torch.tensor([.5, float("nan")])
    masks["right_pose"][0, 0] = masks["right_dist"][0, 0] = True
    loss, terms = reconstruction_loss(predicted, targets, masks)
    # hand pose target .25*2=.5: .5*.5^2*10=1.25;
    # hand distance target .5*4=2: (2-.5)*20=30.
    assert loss.item() == 31.25
    assert terms["body_pose"].item() == 0
    loss.backward()
    assert predicted["right_pose"].grad.tolist() == [[-5., 0.]]
    assert predicted["right_dist"].grad.tolist() == [[-20., 0.]]
    assert predicted["body_pose"].grad.abs().sum() == 0


def test_regularization_denominator_sum_and_constant_gradients():
    values = torch.tensor([[0., 0.], [1., 2.], [2., 4.]], requires_grad=True)
    variance, covariance = feature_regularization(values)
    assert variance.item() == 0
    # C01=C10=4/3: count both entries, denominator N=3.
    torch.testing.assert_close(covariance, torch.tensor(32 / 9))
    constant = torch.zeros(4, 3, requires_grad=True)
    var, cov = feature_regularization(constant)
    assert var.item() == 3 and cov.item() == 0
    (var + cov).backward()
    assert torch.isfinite(constant.grad).all()
    with pytest.raises(ValueError):
        feature_regularization(torch.zeros(1, 3))


def test_gram_matrix_manual_and_pair_identity():
    values = torch.tensor([[[1., 2.], [3., 4.]]])
    # X^T X = [[10,14],[14,20]], row means [12,17], divide K*D=4.
    torch.testing.assert_close(gram_style(values), torch.tensor([[3., 4.25]]))
    pos, neg = style_pairs(["a", "a", "b", "b"], torch.Generator().manual_seed(3))
    assert pos.tolist() == [1, 0, 3, 2]
    assert all((a < 2) != (b < 2) for a, b in enumerate(neg.tolist()))
    for ids in [["a", "a"], ["a", "b", "b"], ["a", "a", "a", "b", "b"]]:
        with pytest.raises(ValueError):
            style_pairs(ids)


def test_adversarial_analytic_derivatives_gate_ema_and_state():
    matched = torch.tensor([.8, .3], requires_grad=True)
    unmatched = torch.tensor([.2, .9], requires_grad=True)
    em, eu = torch.tensor(.7, requires_grad=True), torch.tensor(.4, requires_grad=True)
    loss = adversarial_loss(matched, unmatched, em, eu)
    # Only first sample incurs (.8-.4)^2+(.7-.2)^2=.41, mean .205.
    torch.testing.assert_close(loss, torch.tensor(.205))
    loss.backward()
    torch.testing.assert_close(matched.grad, torch.tensor([.4, 0.]))
    torch.testing.assert_close(unmatched.grad, torch.tensor([-.5, 0.]))
    assert em.grad is None and eu.grad is None
    assert adversarial_loss(matched, unmatched, .2, .4).item() == 0
    ema = MatchEMA()
    ema.update(torch.tensor([.8, .6]), torch.tensor([.2, .4]))
    torch.testing.assert_close(ema.matched, torch.tensor(.07))
    torch.testing.assert_close(ema.unmatched, torch.tensor(.03))
    restored = MatchEMA()
    restored.load_state_dict(ema.state_dict(), strict=True)
    assert restored.updates.item() == 1


def test_discriminator_freeze_preserves_content_gradient_and_buffers():
    torch.manual_seed(9)
    model = StyleDiscriminator(dimension=4).train()
    content = torch.randn(4, 4, requires_grad=True)
    style = torch.randn(4, 4, requires_grad=True)
    pos, neg = style_pairs(["a", "a", "b", "b"])
    positive, negative = model.paired_logits(content.detach(), style.detach(), pos, neg)
    discriminator_loss(positive, negative).backward()
    assert content.grad is None and style.grad is None
    model.zero_grad(set_to_none=True)
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    with frozen_discriminator(model):
        positive, negative = model.paired_logits(content, style.detach(), pos, neg)
        adversarial_loss(positive.sigmoid(), negative.sigmoid(), .8, .2).backward()
    assert content.grad is not None and content.grad.abs().sum() > 0
    assert style.grad is None and all(p.grad is None for p in model.parameters())
    for key, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[key], rtol=0, atol=0)
    assert model.training and all(p.requires_grad for p in model.parameters())
