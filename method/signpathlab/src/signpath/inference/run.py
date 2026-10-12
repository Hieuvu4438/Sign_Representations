"""Resolve selected heads/encoders by hashes without importing training into deployment."""
from dataclasses import dataclass
import json
from pathlib import Path
import torch

from ..cache.hashing import canonical_hash, file_hash
from ..cache.index import CachedCorpus
from ..config import ResolvedConfig
from ..models.heads import AttentiveHead, LinearHead
from .deployment import OfflineEncoderGraph
from .extract import load_encoder


@dataclass(frozen=True)
class InferenceRun:
    directory: Path
    resolved: ResolvedConfig
    model_config: dict
    model: object
    head: object
    input_identity: dict
    model_hash: str
    class_ids: tuple[int, ...]

    def graph(self):
        ds = self.resolved.values["downstream"]
        return OfflineEncoderGraph(self.model.online if self.model is not None else None, self.head,
            head_kind=ds["head"], l2_normalize=ds["l2_normalize"], frozen_visual=self.model is None)


def load_run(directory, device="cpu"):
    directory = Path(directory).resolve()
    pointer = directory / "downstream_selection.json"
    if not (directory / "head.pt").exists() and pointer.exists():
        value = json.loads(pointer.read_text())
        directory = Path(value["run"]).resolve()
        if file_hash(directory / "head.pt") != value["head_sha256"]:
            raise ValueError("Selected downstream pointer/head checksum mismatch")
    recorded = json.loads((directory / "resolved_config.json").read_text())
    c = recorded["config"]
    resolved = ResolvedConfig(c, Path(recorded["project_root"]), tuple(recorded["config_chain"]), canonical_hash(c))
    if (directory / "head.pt").exists():
        path = directory / "head.pt"
        lock = json.loads((directory / "protocol.lock.json").read_text())
        digest = file_hash(path)
        if lock["head_sha256"] != digest:
            raise ValueError("Selected inference head differs from locked protocol")
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value["config"] != c or lock["identity"] != value["identity"]:
            raise ValueError("Selected inference config/protocol mismatch")
        ds, classes = c["downstream"], tuple(value["classes"])
        model, model_config, identity = None, c, value.get("cache_identity")
        if value["encoder_path"]:
            model, identity, model_config, encoder_hash = load_encoder(value["encoder_path"], device)
            if encoder_hash != value["encoder_hash"]:
                raise ValueError("Selected head encoder checkpoint changed")
            if value["online"] is not None:
                model.online.load_state_dict(value["online"], strict=True)
        if identity is None:
            corpus = CachedCorpus(resolved.path(c["data"]["train_manifest"]), resolved.path(c["cache"]["root"]), split="train", config=c)
            if corpus.fingerprint != value["identity"]["train"]:
                raise ValueError("Frozen head cache provenance changed")
            identity = corpus.identity
        head = (AttentiveHead(value["dimension"], len(classes), ds["attentive_heads"])
                if ds["head"] == "attentive" else LinearHead(value["dimension"], len(classes))).to(device)
        head.load_state_dict(value["head"], strict=True)
        head.eval().requires_grad_(False)
        return InferenceRun(directory, resolved, model_config, model, head, identity, digest, classes)
    checkpoint = directory / "checkpoints/best.pt"
    if not checkpoint.exists():
        raise ValueError("Inference requires an explicitly selected best checkpoint or locked downstream head")
    model, identity, model_config, digest = load_encoder(checkpoint, device)
    return InferenceRun(directory, resolved, model_config, model, None, identity, digest, ())
