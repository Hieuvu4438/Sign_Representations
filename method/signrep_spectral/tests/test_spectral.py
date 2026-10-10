import unittest

import numpy as np
import torch

from method.signrep_spectral.losses.spectral import grouped_term_stats, spectral_term_stats, window_time_term_stats
from method.signrep_spectral.losses.stats import reduce_loss_stats

WINDOWS = [{"n": 8, "hop": 4}, {"n": 16, "hop": 8}]


def oracle_term(pred, target, valid, n, hop):
    """Independent NumPy oracle with explicit loops and a direct DFT."""
    B, T, D = pred.shape
    num, cnt = 0.0, 0
    w = np.array([0.5 - 0.5 * np.cos(2 * np.pi * k / n) for k in range(n)])
    for b in range(B):
        for d in range(D):
            v = valid[b, :, d]
            c = max(v.sum(), 1)
            mt = sum(target[b, t, d] for t in range(T) if v[t]) / c
            mp = sum(pred[b, t, d] for t in range(T) if v[t]) / c
            for s in range(0, T - n + 1, hop):
                if not v[s : s + n].all():
                    continue
                for k in range(n // 2 + 1):
                    St = sum(w[j] * (target[b, s + j, d] - mt) * np.exp(-2j * np.pi * k * j / n) for j in range(n)) / w.sum()
                    Sp = sum(w[j] * (pred[b, s + j, d] - mp) * np.exp(-2j * np.pi * k * j / n) for j in range(n)) / w.sum()
                    num += abs(np.log1p(abs(Sp)) - np.log1p(abs(St)))
                    cnt += 1
    return num, cnt


class TestSpectral(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.t = torch.randn(3, 16, 5, generator=g, dtype=torch.float64)
        self.p = torch.randn(3, 16, 5, generator=g, dtype=torch.float64)
        self.v = torch.ones(3, 16, 5, dtype=torch.bool)

    def test_window_shapes_and_hann(self):
        x = torch.arange(16.0).view(1, 16, 1)
        u8 = x.unfold(1, 8, 4)
        self.assertEqual(tuple(u8.shape), (1, 3, 1, 8))
        self.assertEqual(u8[0, 1, 0, 0].item(), 4.0)
        self.assertEqual(tuple(x.unfold(1, 16, 8).shape), (1, 1, 1, 16))
        for n, s in ((8, 4.0), (16, 8.0)):
            w = torch.hann_window(n, periodic=True, dtype=torch.float64)
            ref = torch.tensor([0.5 - 0.5 * np.cos(2 * np.pi * k / n) for k in range(n)], dtype=torch.float64)
            self.assertTrue(torch.allclose(w, ref, atol=1e-15))
            self.assertAlmostEqual(w.sum().item(), s, places=12)
        st = spectral_term_stats(self.p, self.t, self.v, 8, 4)
        self.assertEqual(st.count, 3 * 3 * 5 * 5)  # B*M*D*K
        st = spectral_term_stats(self.p, self.t, self.v, 16, 8)
        self.assertEqual(st.count, 3 * 1 * 5 * 9)

    def test_matches_numpy_oracle(self):
        v = self.v.clone()
        v[0, 5, 1] = False
        v[2, 12:, 3] = False
        for n, hop in ((8, 4), (16, 8)):
            st = spectral_term_stats(self.p, self.t, v, n, hop)
            num, cnt = oracle_term(self.p.numpy(), self.t.numpy(), v.numpy(), n, hop)
            self.assertEqual(st.count, cnt)
            self.assertLess(abs(st.numerator.item() - num), 1e-10)

    def test_identity_offset_amplitude_sign(self):
        z = spectral_term_stats(self.t.clone(), self.t, self.v, 8, 4)
        self.assertLess(z.numerator.item(), 1e-12)
        off = spectral_term_stats(self.t + torch.randn(1, 1, 5, dtype=torch.float64), self.t, self.v, 8, 4)
        self.assertLess(off.numerator.item(), 1e-10)
        amp = spectral_term_stats(2 * self.t, self.t, self.v, 8, 4)
        self.assertGreater(amp.numerator.item(), 1e-3)
        q = self.t - self.t.mean(1, keepdim=True)
        inv = spectral_term_stats(-q, q, self.v, 8, 4)
        self.assertLess(inv.numerator.item(), 1e-10)  # magnitude cannot see sign inversion
        from method.signrep_spectral.losses.geometric import geometric_term_stats
        self.assertGreater(geometric_term_stats(-q, q, self.v).numerator.item(), 1e-3)

    def test_full_sequence_reversal_limitation(self):
        q = self.t[0, :, 0] - self.t[0, :, 0].mean()
        a = torch.fft.fft(q).abs()
        b = torch.fft.fft(torch.flip(q, [0])).abs()
        self.assertTrue(torch.allclose(a, b, atol=1e-10))

    def test_parseval(self):
        q = self.t[0, :8, 0] * torch.hann_window(8, periodic=True, dtype=torch.float64)
        full = torch.fft.fft(q, norm="ortho")
        self.assertAlmostEqual(full.abs().pow(2).sum().item(), q.pow(2).sum().item(), places=10)
        one = torch.fft.rfft(q, norm="ortho").abs().pow(2)
        w = torch.ones_like(one)
        w[1:-1] = 2.0  # interior bins doubled, DC/Nyquist once
        self.assertAlmostEqual((w * one).sum().item(), q.pow(2).sum().item(), places=10)

    def test_all_missing(self):
        v = torch.zeros_like(self.v)
        t = self.t.clone()
        t[0, 0, 0] = float("nan")
        p = self.p.clone().requires_grad_(True)
        st = spectral_term_stats(p, t, v, 8, 4)
        self.assertEqual(st.count, 0)
        loss, _ = reduce_loss_stats([st], 1)
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertEqual(p.grad.abs().sum().item(), 0.0)

    def test_partial_missing_rejects_only_affected_windows(self):
        v = self.v.clone()
        v[1, 6, 2] = False  # in windows starting 0 and 4 (N=8), and the N=16 window
        st8 = spectral_term_stats(self.p, self.t, v, 8, 4)
        self.assertEqual(st8.count, (3 * 3 * 5 - 2) * 5)
        st16 = spectral_term_stats(self.p, self.t, v, 16, 8)
        self.assertEqual(st16.count, (3 * 5 - 1) * 9)

    def test_invalid_payload_invariance_and_valid_nan_raises(self):
        v = self.v.clone()
        v[0, 3, 0] = False
        base = spectral_term_stats(self.p, self.t, v, 8, 4)
        for bad in (float("nan"), float("inf"), 1e30):
            t = self.t.clone()
            t[0, 3, 0] = bad
            st = spectral_term_stats(self.p, t, v, 8, 4)
            self.assertEqual(st.count, base.count)
            self.assertAlmostEqual(st.numerator.item(), base.numerator.item(), places=10)
        t = self.t.clone()
        t[0, 4, 0] = float("nan")
        with self.assertRaises(ValueError):
            spectral_term_stats(self.p, t, v, 8, 4)
        p = self.p.clone()
        p[0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            spectral_term_stats(p, self.t, self.v, 8, 4)

    def test_short_clip_raises(self):
        with self.assertRaises(ValueError):
            spectral_term_stats(self.p[:, :7], self.t[:, :7], self.v[:, :7], 8, 4)

    def test_gradcheck(self):
        t = self.t[:1, :, :2]
        v = self.v[:1, :, :2].clone()
        v[0, 9, 1] = False
        p = (t + 0.3 * torch.randn_like(t)).requires_grad_(True)
        f = lambda x: spectral_term_stats(x, t, v, 8, 4).numerator + spectral_term_stats(x, t, v, 16, 8).numerator
        self.assertTrue(torch.autograd.gradcheck(f, (p,), eps=1e-6, atol=1e-5))

    def test_invalid_positions_get_zero_grad(self):
        v = self.v.clone()
        v[0, :, 1] = True
        v[0, 3, 1] = False
        p = self.p.clone().requires_grad_(True)
        spectral_term_stats(p, self.t, v, 8, 4).numerator.backward()
        self.assertEqual(p.grad[0, 3, 1].item(), 0.0)
        self.assertNotEqual(p.grad[0, 10, 1].item(), 0.0)

    def test_fixed_denominator_reduction(self):
        groups = {"a": (self.p, self.t, self.v), "b": (self.p, self.t, torch.zeros_like(self.v))}
        stats = grouped_term_stats(groups, WINDOWS, "spectral")
        self.assertEqual(len(stats), 4)
        loss, logs = reduce_loss_stats(stats, 4)
        manual = (stats[0].mean() + stats[1].mean()) / 4  # empty terms still count in denominator
        self.assertAlmostEqual(loss.item(), manual.item(), places=6)
        with self.assertRaises(ValueError):
            reduce_loss_stats(stats, 3)

    def test_window_time_control(self):
        st = window_time_term_stats(self.t + 1.0, self.t, self.v, 8, 4)
        self.assertLess(st.numerator.item(), 1e-10)
        self.assertGreater(window_time_term_stats(-self.t, self.t, self.v, 8, 4).numerator.item(), 1e-3)


if __name__ == "__main__":
    unittest.main()
