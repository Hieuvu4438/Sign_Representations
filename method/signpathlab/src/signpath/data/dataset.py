"""A training provider owns all RNG used for sources, segments and masks."""
import math
import numpy as np
from .collate import TrainingExample, collate_examples
from .sampling import SourceSegmentSampler, slice_features
from ..geometry.paths import geometry_from_features
from ..masking.closure import ClosureResolver
from ..masking.samplers import MaskSampler
from ..masking.cross_modal import CrossModalSampler, observation_view
from ..models.tokenizer import SignTokenizer


def geometry_kwargs(config):
    g = config["geometry"]
    return {"window_seconds": g["window_seconds"], "center_stride_samples": g["center_stride_samples"],
            "descriptor": g["descriptor"], "min_quality": config["pose"]["min_body_visibility"],
            "min_distinct": g["min_distinct_observations"]}


def fitting_geometry(bundle, config):
    """Include every center phase that a random contiguous train segment can expose.

    Fitting only centers 0,4,... of a full clip can miss valid group/scale pairs
    whose observed window appears when a crop begins at phase 1,2 or 3.
    """
    stride = config["geometry"]["center_stride_samples"]
    for offset in range(min(stride, len(bundle.timestamps))):
        view = slice_features(bundle, offset, len(bundle.timestamps))
        yield from geometry_from_features(view, **geometry_kwargs(config))


class BatchProvider:
    def __init__(self, config, bundles, source_to_samples, projectors=None, *, seed=None, cluster_targets=None):
        self.config, self.bundles, self.projectors = config, bundles, projectors
        self.cluster_targets = cluster_targets
        seed = config["experiment"]["seed"] if seed is None else seed
        self.source_sampler = SourceSegmentSampler(source_to_samples, seed)
        self.mask_generator = np.random.default_rng(seed + 1000003)
        masking = config["masking"]
        if masking["regime"] == "strict_temporal":
            if masking["strategy"] != "synchronized_temporal":
                raise ValueError("strict_temporal requires synchronized temporal hiding")
            self.masker = MaskSampler(masking["requested_frame_ratio"], tuple(masking["span_seconds"]))
        else:
            if masking["strategy"] == "synchronized_temporal":
                raise ValueError("synchronized_temporal belongs to strict_temporal; select an explicit cross-modal observation strategy")
            self.masker = CrossModalSampler(masking["strategy"], masking["requested_frame_ratio"], tuple(masking["span_seconds"]))
        self.closure = ClosureResolver()
        self.tokenizer = SignTokenizer()
        self.attempts = 0

    def load(self, sample_id):
        return self.bundles[sample_id] if isinstance(self.bundles, dict) else self.bundles(sample_id)

    def next_batch(self):
        examples = []
        config = self.config
        selected = [self.source_sampler.select_sample() for _ in range(config["training"]["microbatch_per_gpu"])]
        prefetch = getattr(self.bundles, "prefetch", None)
        loaded = prefetch(selected) if prefetch else [self.load(name) for name in selected]
        for full_bundle in loaded:
            bundle = self.source_sampler.segment(full_bundle, config["data"]["segment_samples"])
            geo = geometry_from_features(bundle, **geometry_kwargs(config)) if config["geometry"]["enabled"] and config["loss"]["geometry_weight"] else ()
            tokens = self.tokenizer.build(bundle, geo)
            if config["masking"]["regime"] == "cross_modal":
                tokens = observation_view(tokens, bundle.source_frame_index, config["masking"]["strategy"])
            valid_count = sum(t.valid for t in tokens.tokens)
            minimum = config["masking"]["min_visible_raw_tokens"]
            if valid_count < 2 * minimum:
                minimum = max(1, math.ceil(valid_count / 4))
            for _ in range(config["masking"]["retries"]):
                raw = self.masker.sample(tokens, self.mask_generator)
                plan = self.closure.resolve(tokens, raw, geometry_eligibility=config["masking"]["geometry_target_eligibility"])
                self.attempts += 1
                if len(plan.visible) >= minimum and plan.semantic_targets:
                    break
            else:
                # The sample stays in the batch, but cannot contribute an invalid update.
                from dataclasses import replace
                plan = replace(plan, semantic_targets=(), geometry_targets=(), diagnostics={**plan.diagnostics, "skip_reason": "mask_resample_exhausted"})
            cluster = self.cluster_targets(bundle) if self.cluster_targets is not None else None
            examples.append(TrainingExample(bundle, tokens, plan, self.projectors, cluster))
        return collate_examples(examples, config["model"]["raw_dimensions"], config["model"]["predictor"]["geometry_output_dim"],
                                pad_to=config["data"]["segment_samples"])

    def state_dict(self):
        return {"source_sampler": self.source_sampler.state_dict(), "mask_attempts": self.attempts}

    def load_state_dict(self, value):
        self.source_sampler.load_state_dict(value["source_sampler"])
        self.attempts = value["mask_attempts"]
