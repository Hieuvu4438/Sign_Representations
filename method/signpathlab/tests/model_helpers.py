from copy import deepcopy
from pathlib import Path
import numpy as np
import torch

from signpath.config import load_config
from signpath.data.collate import TrainingExample, collate_examples
from signpath.geometry.paths import geometry_from_features
from signpath.masking.closure import ClosureResolver
from signpath.masking.samplers import MaskSampler
from signpath.models.tokenizer import SignTokenizer
from signpath.targets.projection import ProjectionObservation, TargetProjectors

ROOT = Path(__file__).resolve().parents[1]


def tiny_config(visual_dim=8, steps=20):
    c = deepcopy(load_config(ROOT / "configs/smoke.yaml").values)
    c["cache"]["visual_dim"] = visual_dim
    c["model"]["raw_dimensions"][:4] = [visual_dim] * 4
    c["model"]["embed_dim"] = 24
    c["model"]["projection_hidden_dim"] = 24
    c["model"]["predictor"]["embed_dim"] = 24
    c["model"]["encoder"]["layers"] = 1
    c["model"]["encoder"]["return_layers"] = [1]
    c["training"]["max_successful_steps"] = steps
    c["training"]["learning_rate"] = .002
    return c


def example(bundle, seed=17, geometry=True):
    geo = geometry_from_features(bundle) if geometry else ()
    projectors = None
    if geo:
        observations = [ProjectionObservation("train-source", "train", f"{g.group}/{g.scale}", g.descriptor) for g in geo if g.valid]
        projectors = TargetProjectors.fit(observations, allowed_train_sources={"train-source"},
            manifest_sha256="a" * 64, extractor_hash=bundle.meta["extractor_hash"], descriptor_config={"fixture": True})
    tokens = SignTokenizer().build(bundle, geo)
    raw = MaskSampler().sample(tokens, np.random.default_rng(seed))
    plan = ClosureResolver().resolve(tokens, raw)
    return TrainingExample(bundle, tokens, plan, projectors)


def batch_from(bundle, geometry=True, pad_to=None):
    config = tiny_config(bundle.meta["feature_dim"])
    return collate_examples([example(bundle, geometry=geometry)], config["model"]["raw_dimensions"], pad_to=pad_to)
