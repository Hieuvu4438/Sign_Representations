"""Single controlled import boundary for the vendored SignRep source.

The vendored code uses absolute top-level imports (``models.*``,
``augmentation.*``). We refuse to load if those names already resolve to a
different package, and we scope the ``sys.path`` change to the load call.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

# Literal constructor verified against third_party/SignRep/example_usage.py.
VENDOR_MODEL_PARAMS: Dict[str, Any] = {
    "ckpt_dir": None,
    "backbone_params": {"drop_path_rate": 0.1, "num_cls_tokens": 0},
    "out_dim": 768,
    "mask_ratio": 0.0,
    "init_weights": False,
    "cls_head_name": "models.head_models.dict_head_v6_FINAL_thoughtful",
    "cls_head_params": {
        "latents_params": {"hidden_dim": 512, "out_dim": 384},
        "total_frames": 16,
        "projection_params": {
            "full_body_angles": {"hidden_dim": None, "output_dim": 22 * 2},
            "full_right_angles": {"hidden_dim": None, "output_dim": 41 * 2},
            "full_left_angles": {"hidden_dim": None, "output_dim": 41 * 2},
            "body_pose": {"hidden_dim": None, "output_dim": 61 * 3},
            "right_pose": {"hidden_dim": None, "output_dim": 21 * 3},
            "left_pose": {"hidden_dim": None, "output_dim": 21 * 3},
            "body_dist": {"hidden_dim": None, "output_dim": 12 * 22 * 3},
            "right_dist": {"hidden_dim": None, "output_dim": 5 * 11 * 3},
            "left_dist": {"hidden_dim": None, "output_dim": 5 * 11 * 3},
        },
    },
}
VENDOR_TOP_LEVEL = ("models", "augmentation")
SOURCE_FILES = (
    "example_usage.py",
    "models/final_models/FINAL_hiera_latent_model_head_v25_active.py",
    "models/head_models/dict_head_v6_FINAL_thoughtful.py",
    "models/hiera/hiera_nodec_mae.py",
    "models/hiera/hiera_clstoken.py",
    "models/hiera/hiera_utils.py",
    "augmentation/video/base_video_aug.py",
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def resolve_vendor_root(vendor_root: str | Path) -> Path:
    p = Path(vendor_root)
    if not p.is_absolute():
        p = REPO_ROOT / p
    p = p.resolve()
    if not (p / "models" / "final_models").is_dir():
        raise FileNotFoundError(f"vendored SignRep not found at {p}")
    return p


def _module_locations(mod) -> list:
    locs = []
    if getattr(mod, "__file__", None):
        locs.append(Path(mod.__file__).resolve())
    for p in list(getattr(mod, "__path__", []) or []):
        locs.append(Path(p).resolve())
    return locs


def _assert_inside(mod, root: Path) -> None:
    locs = _module_locations(mod)
    if not locs or not all(root in loc.parents or loc == root for loc in locs):
        raise ImportError(
            f"module {mod.__name__!r} resolves to {locs}, outside vendor root {root}. "
            "Run the signrep_spectral CLI in a dedicated process."
        )


@contextlib.contextmanager
def vendor_context(vendor_root: Path):
    for name in VENDOR_TOP_LEVEL:
        if name in sys.modules:
            _assert_inside(sys.modules[name], vendor_root)
    saved = list(sys.path)
    sys.path.insert(0, str(vendor_root))
    try:
        yield
    finally:
        sys.path[:] = saved


def source_hashes(vendor_root: Path) -> Dict[str, str]:
    return {rel: sha256_file(vendor_root / rel) for rel in SOURCE_FILES}


def load_vendor_model(
    vendor_root: str | Path,
    checkpoint: str | Path | None,
    expected_sha256: str | None = None,
    device: str = "cpu",
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Construct the original SignRep Model, strictly load the checkpoint, audit heads."""
    root = resolve_vendor_root(vendor_root)
    audit: Dict[str, Any] = {"vendor_root": str(root), "source_hashes": source_hashes(root)}
    with vendor_context(root):
        final_mod = importlib.import_module("models.final_models.FINAL_hiera_latent_model_head_v25_active")
        head_mod = importlib.import_module(VENDOR_MODEL_PARAMS["cls_head_name"])
        for name in VENDOR_TOP_LEVEL[:1]:
            _assert_inside(sys.modules[name], root)
        _assert_inside(final_mod, root)
        _assert_inside(head_mod, root)
        model = final_mod.Model(**copy.deepcopy(VENDOR_MODEL_PARAMS))
    # Constructor has a broad try/except that can silently replace the head.
    if not isinstance(model.cls_head, head_mod.ProjectionModel):
        raise RuntimeError("SignRep constructor skipped the class head (got %r)" % type(model.cls_head))
    for attr in ("fc", "active_head", "backbone"):
        if not hasattr(model, attr):
            raise RuntimeError(f"SignRep model missing {attr}")
    expected = {k: v["output_dim"] for k, v in VENDOR_MODEL_PARAMS["cls_head_params"]["projection_params"].items()}
    got = {k: m.layers[-1].out_features for k, m in model.cls_head.dict_component_projections.items()}
    if got != expected:
        raise RuntimeError(f"head dims mismatch {got} vs {expected}")
    audit["head_dims"] = got
    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = REPO_ROOT / ckpt_path
        ckpt_path = ckpt_path.resolve()
        digest = sha256_file(ckpt_path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise RuntimeError(f"checkpoint sha256 {digest} != expected {expected_sha256}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        audit["checkpoint"] = {
            "path": str(ckpt_path),
            "bytes": ckpt_path.stat().st_size,
            "sha256": digest,
            "root_keys": sorted(state.keys()),
        }
        res = model.load_state_dict(state["model"], strict=True)
        audit["checkpoint"]["strict_load"] = {
            "missing": list(res.missing_keys),
            "unexpected": list(res.unexpected_keys),
        }
    audit["projection_model_class"] = head_mod.ProjectionModel
    model.to(device)
    return model, audit


def get_projection_model_class(vendor_root: str | Path):
    root = resolve_vendor_root(vendor_root)
    with vendor_context(root):
        mod = importlib.import_module(VENDOR_MODEL_PARAMS["cls_head_name"])
        _assert_inside(mod, root)
    return mod.ProjectionModel


def build_vendor_backbone(vendor_root: str | Path):
    """Architecture-only Hiera backbone (no weights) for loading exported encoders."""
    root = resolve_vendor_root(vendor_root)
    with vendor_context(root):
        mod = importlib.import_module("models.hiera.hiera_nodec_mae")
        _assert_inside(mod, root)
        return mod.mae_hiera_base_16x224(num_classes=400, pretrained=False, **VENDOR_MODEL_PARAMS["backbone_params"])
