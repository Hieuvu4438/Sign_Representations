import importlib
import os
import sys
from pathlib import Path
import torch
from signrep_baseline.model import SignRep
from signrep_baseline.schema import PRIOR_SHAPES, BODY_SOURCE, BODY_DEST, FidelityChoices

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path(os.environ.get("SIGNREP_CHECKPOINT", str(ROOT.parents[1] / "checkpoints/signrep/ckpt.pt")))
if not CHECKPOINT.exists():
    CHECKPOINT = ROOT / "runtime/released/ckpt.pt"
torch.set_num_threads(2)


def test_released_checkpoint_against_original_author_forward():
    sys.path.insert(0, str(ROOT / "vendor/signrep"))
    original = importlib.import_module("models.final_models.FINAL_hiera_latent_model_head_v25_active")
    # Author example's architecture parameters, exactly including bias choices.
    import math
    params = {"backbone_params": {"drop_path_rate": .1, "num_cls_tokens": 0},
              "ckpt_dir": None, "mask_ratio": 0, "out_dim": 768, "init_weights": False,
              "cls_head_name": "models.head_models.dict_head_v6_FINAL_thoughtful",
              "cls_head_params": {"latents_params": {"hidden_dim": 512, "out_dim": 384},
              "total_frames": 16, "projection_params": {k: {"hidden_dim": None, "output_dim": math.prod(v)}
                                                        for k, v in PRIOR_SHAPES.items()}}}
    author = original.Model(**params).eval()
    reproduced = SignRep().eval()
    author.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model"], strict=True)
    reproduced.load_released(CHECKPOINT)
    gen = torch.Generator().manual_seed(17)
    images = torch.randn(1, 3, 16, 224, 224, generator=gen)
    with torch.inference_mode():
        expected, actual = author(images), reproduced(images)
    for key in ["features", "latent", "cls_active"]:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    for key in [*PRIOR_SHAPES, "post_latent", "latent"]:
        torch.testing.assert_close(actual["cls_head"][key], expected["cls_head"][key], rtol=0, atol=0)
    assert actual["tokens"].shape[-1] == 768
    # Pretraining must also preserve author's masking and stochastic-depth path.
    author.mask_ratio = .8
    author.train()
    reproduced.train()
    with torch.no_grad():
        torch.manual_seed(71)
        expected = author(images)
        torch.manual_seed(71)
        actual = reproduced(images)
    for key in ["features", "latent", "cls_active"]:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    for key in PRIOR_SHAPES:
        torch.testing.assert_close(actual["cls_head"][key], expected["cls_head"][key], rtol=0, atol=0)
    assert actual["tokens"].shape[1] < 1568


def test_inflation_exact_kernel_and_same_token_grid():
    model = SignRep().eval()
    old = model.backbone.patch_embed.proj.weight.detach().clone()
    old_grid = model.backbone.tokens_spatial_shape.copy()
    with torch.inference_mode():
        original_tokens = model.encode(torch.zeros(1, 3, 16, 224, 224), mask_ratio=0)["tokens"].shape
    model.inflate_for_recognition()
    kernel = model.backbone.patch_embed.proj.weight.detach()
    for original, new in enumerate([1, 3, 5]):
        torch.testing.assert_close(kernel[:, :, new], old[:, :, original], rtol=0, atol=0)
    assert torch.count_nonzero(kernel[:, :, [0, 2, 4, 6]]) == 0
    assert model.backbone.tokens_spatial_shape == old_grid
    with torch.inference_mode():
        out = model.encode(torch.zeros(1, 3, 64, 224, 224), mask_ratio=0)
    assert out["features"].shape == (1, 768)
    # q_pool=2: 8 temporal ×14×14 spatial output tokens. 392 counts mask units.
    assert out["tokens"].shape == original_tokens == (1, 1568, 768)


def test_distance_default_matches_author_trained_linear_heads():
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model"]
    prefix = "cls_head.dict_component_projections."
    pose_w = state[prefix + "body_pose.layers.0.weight"].reshape(61, 3, 384)
    dist_w = state[prefix + "body_dist.layers.0.weight"].reshape(12, 22, 3, 384)
    expected_w = pose_w[list(BODY_SOURCE)][:, None] - pose_w[list(BODY_DEST)][None]
    pose_b = state[prefix + "body_pose.layers.0.bias"].reshape(61, 3)
    dist_b = state[prefix + "body_dist.layers.0.bias"].reshape(12, 22, 3)
    expected_b = pose_b[list(BODY_SOURCE)][:, None] - pose_b[list(BODY_DEST)][None]
    # Independent checkpoint oracle, including bias, with no self-generated targets.
    assert (dist_w - expected_w).norm() < (dist_w + expected_w).norm()
    assert (dist_b - expected_b).norm() < (dist_b + expected_b).norm()
    assert FidelityChoices().distance == "signed_xyz_source_minus_destination"
