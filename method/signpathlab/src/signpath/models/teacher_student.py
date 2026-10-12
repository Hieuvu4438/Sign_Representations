"""Raw JEPA with deterministic target-only geometry and explicit teacher ownership."""
from copy import deepcopy
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

from ..geometry.paths import DEFAULT_PATHS
from ..losses.reduction import global_group_mean
from .encoder import SignEncoder
from .pooling import pool_raw_tokens
from .predictor import QueryPredictor


@dataclass(frozen=True)
class LossOutput:
    total: torch.Tensor
    semantic: object
    geometry: object
    cluster: object
    predictions: object
    global_semantic_count: int
    global_geometry_count: int
    mask_diagnostics: tuple
    skip_reasons: tuple[str, ...]
    numerical_diagnostics: dict

    def log_values(self):
        return {name: {"mean": float(getattr(self, name).mean),
                       "sums": getattr(self, name).sums.cpu().tolist(),
                       "counts": getattr(self, name).counts.cpu().tolist()} for name in ("semantic", "geometry", "cluster")}


def validate_model_runtime(config):
    """Reject optional behavior until a corresponding implementation is available."""
    if config["experiment"]["supervision_regime"] != "sign_only_ssl":
        raise ValueError("Paired/supervised stages require their own runtime; this model implements sign_only_ssl")
    if config["geometry"]["role"] != "target_only":
        raise ValueError("Geometry input-enabled encoder is not implemented until M10")
    if config["model"]["encoder"]["causal"] or config["model"]["encoder"]["stochastic_depth"] != 0:
        raise ValueError("Causal/stochastic-depth variants are not implemented until M10")
    teacher = config["model"]["teacher"]
    if (teacher["kind"], teacher["initialization"], teacher["tau_schedule"], teacher["eval_mode"]) != ("ema", "exact_online_copy", "cosine_successful_steps", True):
        raise ValueError("Only exact-copy/eval EMA teacher is implemented")
    if config["model"]["pooling"] != {"dense": "valid_stream_mean", "clip": "valid_frame_mean"}:
        raise ValueError("Unsupported encoder pooling policy")
    for name in ("variance", "covariance", "pose_prior", "temporal_contrast", "visible_dense"):
        if config["loss"][name + "_weight"]:
            raise ValueError(f"{name} objective is unavailable in the M6/M7 runtime")
    if config["loss"]["geometry_weight"] and not config["geometry"]["enabled"]:
        raise ValueError("Geometry loss cannot be enabled with geometry disabled")
    if not (config["loss"]["semantic_weight"] or config["loss"]["cluster_weight"]):
        raise ValueError("Pretraining requires semantic or cluster targets")


class ResearchModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        validate_model_runtime(config)
        self.config = deepcopy(config)
        self.online = SignEncoder(config)
        self.predictor = QueryPredictor(config)
        self.teacher = deepcopy(self.online).requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def encode_full(self, batch, return_layers=False):
        return self.online(batch, batch.valid, return_layers=return_layers)

    def extract_embeddings(self, batch):
        return pool_raw_tokens(self.encode_full(batch), batch)

    def forward(self, batch):
        return self.training_loss(batch)

    def training_loss(self, batch):
        config, weights = self.config, self.config["loss"]
        student = self.online(batch, batch.visible)
        prediction = self.predictor(student, batch, geometry_enabled=bool(weights["geometry_weight"]))
        b, t, streams = batch.valid.shape
        raw_mask = batch.semantic_targets.flatten(1)
        raw_groups = torch.arange(7, device=raw_mask.device).repeat(t)[None].expand(b, -1)
        semantic_errors = prediction.semantic.sum(dim=-1) * 0
        if weights["semantic_weight"]:
            with torch.no_grad():
                teacher = self.teacher(batch, batch.valid)
                targets = F.layer_norm(teacher.tokens.float(), (teacher.tokens.shape[-1],))
            if weights["semantic"] == "normalized_l1":
                semantic_errors = (prediction.semantic.float() - targets).abs().mean(dim=-1)
            else:
                semantic_errors = 1 - F.cosine_similarity(prediction.semantic.float(), targets, dim=-1, eps=1e-8)
        semantic = global_group_mean(semantic_errors, raw_groups, raw_mask, 7)
        geo_errors = F.smooth_l1_loss(prediction.geometry.float(), batch.geometry_values.float(),
            reduction="none", beta=weights["geometry_beta"]).mean(dim=-1)
        num_scales = len(config["geometry"]["window_seconds"])
        geo_groups = batch.geometry_path * num_scales + batch.geometry_scale
        geometry = global_group_mean(geo_errors, geo_groups, batch.geometry_targets, len(DEFAULT_PATHS) * num_scales)
        cluster_errors = semantic_errors * 0
        cluster_mask = torch.zeros_like(raw_mask)
        if weights["cluster_weight"]:
            errors = []
            cluster_mask = (batch.semantic_targets & (batch.cluster_ids != -100)).flatten(1)
            for stream, logits in enumerate(prediction.cluster):
                labels = batch.cluster_ids[:, :, stream]
                if ((labels != -100) & ((labels < 0) | (labels >= 256))).any():
                    raise ValueError("Cluster labels outside registered 256-class codebook")
                errors.append(F.cross_entropy(logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="none"))
            cluster_errors = torch.stack(errors, dim=2).flatten(1)
        cluster = global_group_mean(cluster_errors, raw_groups, cluster_mask, 7)
        # Connected zeros keep dynamic/empty ranks in the same DDP autograd graph.
        connected_zero = sum(p.reshape(-1)[0] * 0 for p in self.parameters() if p.requires_grad)
        total = (weights["semantic_weight"] * semantic.loss + weights["geometry_weight"] * geometry.loss
                 + weights["cluster_weight"] * cluster.loss + connected_zero)
        active_count = cluster.counts.sum() if weights["cluster_weight"] and not weights["semantic_weight"] else semantic.counts.sum()
        semantic_count, geo_count = int(active_count), int(geometry.counts.sum())
        reasons = []
        if not semantic_count:
            reasons.append("empty_global_semantic_or_cluster_targets")
        if weights["geometry_weight"] and not geo_count:
            reasons.append("empty_geometry_targets_semantic_can_train")
        numerical = {"loss_finite": bool(torch.isfinite(total.detach())), "teacher_used": bool(weights["semantic_weight"]),
                     "visible_context_tokens_local": int(batch.visible.sum())}
        return LossOutput(total, semantic, geometry, cluster, prediction, semantic_count, geo_count,
                          batch.diagnostics, tuple(reasons), numerical)
