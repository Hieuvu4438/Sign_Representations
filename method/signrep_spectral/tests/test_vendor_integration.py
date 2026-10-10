"""Real vendor/checkpoint tests. Skipped (with reason) when the checkpoint is absent."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
CKPT = REPO / "checkpoints/signrep/ckpt.pt"
SHA = "f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b"


@unittest.skipUnless(CKPT.exists(), "SignRep checkpoint not present on this machine")
class TestVendor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from method.signrep_spectral.vendor import load_vendor_model

        cls.model, cls.audit = load_vendor_model("third_party/SignRep", CKPT, SHA, "cpu")
        cls.model.eval()

    def test_strict_load_and_heads(self):
        self.assertEqual(self.audit["checkpoint"]["strict_load"], {"missing": [], "unexpected": []})
        self.assertEqual(self.audit["head_dims"]["left_pose"], 63)

    def test_adapter_parity_and_shim(self):
        from method.signrep_spectral.model import attention_kernel, encoder_from_vendor_model

        enc = encoder_from_vendor_model(self.model)
        x = torch.randn(1, 3, 16, 224, 224, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            y = self.model(x)
            z = enc.encode(x)
            with attention_kernel("sdpa_4d_reshape"):
                s = enc.encode(x)
        self.assertEqual((y["features"] - z["features"]).abs().max().item(), 0.0)
        self.assertTrue(torch.allclose(s["features"], z["features"], atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(s["latent"], z["latent"], atol=1e-5, rtol=1e-4))
        import models.hiera.hiera_clstoken as hc  # restored after the context
        import torch.nn.functional as TF
        self.assertIs(hc.F, TF)

    def test_explicit_mask_semantics(self):
        from method.signrep_spectral.model import encoder_from_vendor_model, make_explicit_mask

        enc = encoder_from_vendor_model(self.model)
        m = make_explicit_mask(enc.num_mask_units(), 2, 0.6, torch.Generator().manual_seed(0))
        self.assertEqual(m.dtype, torch.bool)
        self.assertEqual(int(m[0].sum()), int(enc.num_mask_units() * 0.4))
        x = torch.randn(2, 3, 16, 224, 224)
        with torch.no_grad():
            out = enc.encode(x, mask=m)
        self.assertEqual(tuple(out["features"].shape), (2, 768))

    def test_decoder_heads_and_init_determinism(self):
        from method.signrep_spectral.model import TemporalDecoder, state_hash

        a = TemporalDecoder("third_party/SignRep", {"left_pose": 42, "right_pose": 42, "wrist_image_xy": 4})
        b = TemporalDecoder("third_party/SignRep", {"left_pose": 42, "right_pose": 42, "wrist_image_xy": 4})
        a.reset_parameters(1702)
        torch.rand(100)  # consuming global RNG must not change the seeded init
        b.reset_parameters(1702)
        self.assertEqual(state_hash(a), state_hash(b))
        out = a(torch.randn(3, 768))
        self.assertEqual(tuple(out["wrist_image_xy"].shape), (3, 16, 4))

    def test_preprocessing_matches_upstream_transform(self):
        import cv2
        from method.signrep_spectral.data.video import normalize_frames
        from method.signrep_spectral.vendor import resolve_vendor_root, vendor_context
        import importlib

        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8) for _ in range(3)]
        with vendor_context(resolve_vendor_root("third_party/SignRep")):
            Transformation = importlib.import_module("augmentation.video.base_video_aug").Transformation
        ref = Transformation().aug_video(frames)  # [T,3,224,224]
        ours = normalize_frames(torch.from_numpy(np.stack([cv2.resize(f, (224, 224), interpolation=cv2.INTER_LINEAR) for f in frames])))
        self.assertTrue(torch.allclose(ours.permute(1, 0, 2, 3), ref, atol=1e-5))


class TestImportCollision(unittest.TestCase):
    def test_foreign_models_package_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "models").mkdir()
            (Path(d) / "models" / "__init__.py").write_text("")
            code = (
                "import sys; sys.path.insert(0, %r); import models\n"
                "from method.signrep_spectral.vendor import load_vendor_model\n"
                "try:\n    load_vendor_model('third_party/SignRep', None)\nexcept ImportError as e:\n    print('REJECTED')\n"
                "import models as m2; assert m2 is sys.modules['models']\n" % d
            )
            env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, env=env)
            self.assertIn("REJECTED", r.stdout, r.stderr)


if __name__ == "__main__":
    unittest.main()


class TestDecodeParity(unittest.TestCase):
    def test_cv2_frames_equal_pyav_rgb24(self):
        import json
        import av
        import cv2
        from method.signrep_spectral.data.video import decode_frames

        man = REPO / "method/signrep_spectral/runtime/data_audit_001/manifest.jsonl"
        if not man.exists():
            self.skipTest("audit manifest not present")
        with open(man) as f:
            r = json.loads(f.readline())
        p = str(Path(r["source_root"]) / r["relative_path"])
        frames, _ = decode_frames(p, 3, 7)
        with av.open(p) as c:
            ref = [cv2.resize(fr.to_ndarray(format="rgb24"), (224, 224), interpolation=cv2.INTER_LINEAR)
                   for i, fr in enumerate(c.decode(c.streams.video[0])) if 3 <= i < 7]
        self.assertEqual(frames.shape, (4, 224, 224, 3))
        self.assertTrue(np.array_equal(frames, np.stack(ref)))
        full, _ = decode_frames(p)
        self.assertEqual(len(full), r["num_frames"])
