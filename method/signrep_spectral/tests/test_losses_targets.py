import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from method.signrep_spectral.data.priors import build_canonical_targets
from method.signrep_spectral.data.scaler import scaler_start
from method.signrep_spectral.losses.geometric import geometric_term_stats
from method.signrep_spectral.losses.temporal import difference_term_stats


def fake_raw(F=20, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.2, 0.8, size=(F, 2, 21, 2)).astype(np.float32)
    obs = np.ones((F, 2, 21), bool)
    return {"hand_xy": xy, "joint_observed": obs}


class TestTargets(unittest.TestCase):
    def test_joint_major_wrist_origin_and_translation(self):
        raw = fake_raw()
        tb = build_canonical_targets(raw, 2, 16, 640, 480)
        lp = tb["targets"]["left_pose"]
        self.assertEqual(lp.shape, (16, 42))
        self.assertTrue(np.allclose(lp[:, :2], 0))
        u = raw["hand_xy"][2:18, 0].astype(np.float64).copy()
        u[..., 1] *= 480 / 640
        s = np.median(np.linalg.norm(u[:, 9] - u[:, 0], axis=-1))
        self.assertTrue(np.allclose(lp[:, 2 * 5 : 2 * 5 + 2], (u[:, 5] - u[:, 0]) / s, atol=1e-6))
        shifted = {"hand_xy": raw["hand_xy"] + np.float32(0.05), "joint_observed": raw["joint_observed"]}
        tb2 = build_canonical_targets(shifted, 2, 16, 640, 480)
        self.assertTrue(np.allclose(tb2["targets"]["left_pose"], lp, atol=1e-5))
        self.assertFalse(np.allclose(tb2["targets"]["wrist_image_xy"], tb["targets"]["wrist_image_xy"]))
        self.assertTrue(np.allclose(tb["targets"]["wrist_image_xy"][:, 1], raw["hand_xy"][2:18, 0, 0, 1] * 0.75, atol=1e-6))

    def test_missing_scale_and_wrist(self):
        raw = fake_raw()
        raw["joint_observed"][:, 1, 9] = False
        raw["joint_observed"][2:6, 1, 9] = True  # only 4 valid scale frames -> still ok
        tb = build_canonical_targets(raw, 2, 16, 640, 480)
        self.assertTrue(tb["info"]["right_pose_scale_ok"])
        raw["joint_observed"][5, 1, 9] = False  # 3 frames -> unusable for the whole clip
        tb = build_canonical_targets(raw, 2, 16, 640, 480)
        self.assertFalse(tb["info"]["right_pose_scale_ok"])
        self.assertFalse(tb["masks"]["right_pose"].any())
        raw = fake_raw()
        raw["joint_observed"][7, 0, 0] = False  # missing wrist -> whole frame unobserved for that hand
        tb = build_canonical_targets(raw, 2, 16, 640, 480)
        self.assertFalse(tb["masks"]["left_pose"][5].any())
        self.assertFalse(tb["masks"]["wrist_image_xy"][5, :2].any())
        self.assertTrue(tb["masks"]["wrist_image_xy"][5, 2:].all())

    def test_scaler_start_stable(self):
        starts = np.array([5, 1, 3, 9])
        a = scaler_start("abc", 1700, starts)
        self.assertEqual(a, scaler_start("abc", 1700, starts[::-1]))
        self.assertIn(a, starts)


class TestBaseLosses(unittest.TestCase):
    def test_geometric(self):
        t = torch.randn(2, 16, 4)
        v = torch.ones_like(t, dtype=torch.bool)
        self.assertLess(geometric_term_stats(t.clone(), t, v).numerator.item(), 1e-12)
        self.assertGreater(geometric_term_stats(t + 1, t, v).numerator.item(), 1.0)
        v2 = v.clone()
        v2[0, 0, 0] = False
        t2 = t.clone()
        t2[0, 0, 0] = float("nan")
        st = geometric_term_stats(t + 1, t2, v2)
        self.assertEqual(st.count, v.numel() - 1)
        self.assertTrue(torch.isfinite(st.numerator))

    def test_velocity_accel(self):
        t = torch.randn(2, 16, 3, dtype=torch.float64)
        v = torch.ones_like(t, dtype=torch.bool)
        self.assertLess(difference_term_stats(t + 2, t, v, 1, "v").numerator.item(), 1e-10)
        lin = torch.arange(16.0, dtype=torch.float64).view(1, 16, 1).repeat(2, 1, 3)
        self.assertLess(difference_term_stats(t + lin, t, v, 2, "a").numerator.item(), 1e-10)
        v[0, 5, 0] = False
        self.assertEqual(difference_term_stats(t, t, v, 1, "v").count, 2 * 15 * 3 - 2)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 768)

    def num_mask_units(self):
        return 4

    def encode(self, x, mask_ratio=0.0, mask=None):
        f = self.lin(x)
        return {"features": f, "latent": f}


class TestTrainWiring(unittest.TestCase):
    def _setup(self, aux="spectral_logmag_l1", lam=1.0):
        from method.signrep_spectral.config import DEFAULT
        from method.signrep_spectral.data.priors import HEADS
        from method.signrep_spectral.engine.train import LossComputer
        from method.signrep_spectral.model import TemporalDecoder, TrainingModel
        import copy

        cfg = copy.deepcopy(DEFAULT)
        cfg["loss"]["auxiliary"] = aux
        torch.manual_seed(0)
        dec = TemporalDecoder("third_party/SignRep", cfg["model"]["heads"])
        dec.reset_parameters(1)
        model = TrainingModel(TinyEncoder(), dec)
        scaler = {h: {"mean": np.zeros(d, np.float32), "scale": np.ones(d, np.float32)} for h, d in HEADS.items()}
        lc = LossComputer(cfg, scaler, torch.device("cpu"))
        g = torch.Generator().manual_seed(3)
        batch = {}
        for h, d in HEADS.items():
            batch[f"t_{h}"] = torch.randn(2, 16, d, generator=g)
            batch[f"m_{h}"] = torch.ones(2, 16, d, dtype=torch.bool)
        x = torch.randn(2, 8, generator=g)
        return model, lc, batch, x

    def test_spectral_only_backward_reaches_encoder_and_decoder(self):
        model, lc, batch, x = self._setup()
        out = model(x)
        loss, _ = lc.aux(lc.scaled(out["priors"], batch), "spectral_logmag_l1")
        loss.backward()
        self.assertGreater(model.encoder.lin.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.decoder.net.latent_proj[0].weight.grad.abs().sum().item(), 0)
        for v in batch.values():
            self.assertIsNone(v.grad)

    def test_active_lambda_changes_gradient(self):
        model, lc, batch, x = self._setup()
        out = model(x)
        sc = lc.scaled(out["priors"], batch)
        b, _ = lc.base(sc)
        a, _ = lc.aux(sc, "spectral_logmag_l1")
        gb = torch.autograd.grad(b, model.encoder.lin.weight, retain_graph=True)[0]
        gt = torch.autograd.grad(b + 0.5 * a, model.encoder.lin.weight)[0]
        self.assertGreater((gt - gb).abs().max().item(), 0)

    def test_lambda_zero_never_calls_spectral(self):
        """The trainer only calls aux when lambda_eff>0; verify resolve_lambda for the baseline."""
        from method.signrep_spectral.config import DEFAULT
        from method.signrep_spectral.engine import train as tr
        import copy

        cfg = copy.deepcopy(DEFAULT)
        cfg["loss"].update(auxiliary="none", lambda_spec=0.0)
        self.assertEqual(tr.resolve_lambda(cfg), ("none", 0.0))
        with mock.patch.object(tr, "grouped_term_stats", side_effect=AssertionError("spectral called")):
            model, lc, batch, x = self._setup()
            b, _ = lc.base(lc.scaled(model(x)["priors"], batch))
            b.backward()


if __name__ == "__main__":
    unittest.main()


class TestR2PolicyAndConfig(unittest.TestCase):
    def test_extent_and_scale_floor(self):
        raw = fake_raw()
        raw["hand_xy"][5, 0, 8] = raw["hand_xy"][5, 0, 0] + 0.9  # implausibly far fingertip
        tb = build_canonical_targets(raw, 0, 16, 640, 480, 4, 0.02, 4.0)
        self.assertFalse(tb["masks"]["left_pose"][5, 16:18].any())
        self.assertTrue(tb["masks"]["left_pose"][5, 14:16].all())
        collapsed = fake_raw()
        collapsed["hand_xy"][:, 1] = 0.5 + 0.001 * collapsed["hand_xy"][:, 1]  # all joints ~one point
        tb = build_canonical_targets(collapsed, 0, 16, 640, 480, 4, 0.02, 4.0)
        self.assertFalse(tb["info"]["right_pose_scale_ok"])
        self.assertFalse(tb["masks"]["right_pose"].any())

    def test_config_rejects_unknown_and_bad_cross_fields(self):
        import copy
        from method.signrep_spectral.config import DEFAULT, ConfigError, validate

        c = copy.deepcopy(DEFAULT)
        c["loss"]["typo"] = 1
        with self.assertRaises(ConfigError):
            validate(c)
        c = copy.deepcopy(DEFAULT)
        c["loss"].update(auxiliary="none")
        with self.assertRaises(ConfigError):
            validate(c)  # baseline with lambda_spec != 0
        c = copy.deepcopy(DEFAULT)
        c["train"]["ddp"] = True
        with self.assertRaises(ConfigError):
            validate(c)
        validate(copy.deepcopy(DEFAULT))


class TestRegularizers(unittest.TestCase):
    def test_var_cov(self):
        from method.signrep_spectral.losses.regularizers import covariance_loss, variance_loss

        z = torch.randn(512, 8, generator=torch.Generator().manual_seed(0)) * 2
        self.assertLess(variance_loss(z).item(), 1e-6)  # std ~2 > gamma
        self.assertAlmostEqual(variance_loss(torch.zeros(4, 8)).item(), 1 - 1e-2, places=5)
        self.assertLess(covariance_loss(torch.randn(4096, 4)).item(), 1e-2)
        c = torch.randn(256, 1).repeat(1, 3)
        self.assertGreater(covariance_loss(c).item(), 0.5)
        with self.assertRaises(ValueError):
            variance_loss(torch.zeros(1, 8))
