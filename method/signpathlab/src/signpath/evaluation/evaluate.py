import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from ..cache.hashing import canonical_hash, file_hash
from ..cache.index import CachedCorpus
from ..cli import write_json
from ..config import ResolvedConfig
from ..inference.extract import load_encoder
from ..models.heads import AttentiveHead, LinearHead
from .islr import corpus_embeddings
from .metrics import classification_metrics
from .retrieval import retrieval_metrics
from .bootstrap import classification_bootstrap
from .records import prediction_records, write_jsonl


def evaluate_run(run, split="val", export_predictions=False, device="cpu"):
    if split not in {"val", "test"}:
        raise ValueError("Evaluation requires an explicit held-out val/test split")
    torch.set_num_threads(1)
    run = Path(run).resolve()
    if not (run / "head.pt").exists():
        pointer = json.loads((run / "downstream_selection.json").read_text())
        run = Path(pointer["run"])
        if file_hash(run / "head.pt") != pointer["head_sha256"]:
            raise ValueError("Downstream selection pointer identity mismatch")
    lock = json.loads((run / "protocol.lock.json").read_text())
    head_hash = file_hash(run / "head.pt")
    if head_hash != lock["head_sha256"] or lock["selection_split"] != "val":
        raise ValueError("Evaluation head/protocol is not locked to validation selection")
    value = torch.load(run / "head.pt", map_location="cpu", weights_only=False)
    recorded = json.loads((run / "resolved_config.json").read_text())
    config = value["config"]
    if config != recorded["config"] or value["identity"] != lock["identity"] or lock["downstream_config"] != config["downstream"]:
        raise ValueError("Head/config/protocol identity mismatch")
    resolved = ResolvedConfig(config, Path(recorded["project_root"]), tuple(recorded["config_chain"]), canonical_hash(config))
    root = resolved.path(config["cache"]["root"])
    corpus = CachedCorpus(resolved.path(config["data"][f"{split}_manifest"]), root, split=split, config=config)
    train = CachedCorpus(resolved.path(config["data"]["train_manifest"]), root, split="train", config=config)
    if train.fingerprint != value["identity"]["train"] or (split == "val" and corpus.fingerprint != value["identity"]["val"]):
        raise ValueError("Fitted downstream data membership changed")
    from ..data.audit import audit_records
    audit_members = list(train.records.values()) + list(corpus.records.values())
    if split == "test":
        validation = CachedCorpus(resolved.path(config["data"]["val_manifest"]), root, split="val", config=config)
        if validation.fingerprint != value["identity"]["val"]:
            raise ValueError("Validation selection membership changed before final test")
        audit_members += list(validation.records.values())
    audit = audit_records(audit_members,
        signer_disjoint="signer_disjoint" in config["experiment"]["evaluation_protocol"], verify_files=True)
    if not audit["passed"] or corpus.identity != train.identity:
        raise ValueError("Evaluation split overlap/input-identity audit failed")
    model, encoder_config = None, None
    if value["encoder_path"]:
        model, identity, encoder_config, digest = load_encoder(value["encoder_path"], device)
        if digest != value["encoder_hash"] or identity != corpus.identity:
            raise ValueError("Downstream encoder/feature identity mismatch")
        if value["online"] is not None:
            model.online.load_state_dict(value["online"], strict=True)
    ds, classes = config["downstream"], value["classes"]
    head = (AttentiveHead(value["dimension"], len(classes), ds["attentive_heads"])
            if ds["head"] == "attentive" else LinearHead(value["dimension"], len(classes))).to(device)
    head.load_state_dict(value["head"], strict=True)
    head.eval()
    with torch.no_grad():
        outputs = corpus_embeddings(corpus, model, encoder_config, ds, device)
        clip = torch.stack([o.clip for o in outputs])
        if ds["head"] == "attentive":
            logits = torch.cat([head((F.normalize(o.dense, dim=-1) if ds["l2_normalize"] else o.dense)[None], o.valid[None]) for o in outputs])
        else:
            logits = head(F.normalize(clip, dim=-1) if ds["l2_normalize"] else clip)
        gallery = torch.stack([o.clip for o in corpus_embeddings(train, model, encoder_config, ds, device)])
    labels = [corpus.records[name].label_id for name in corpus.available]
    metric = classification_metrics(logits.cpu().numpy(), labels, classes)
    retrieval = retrieval_metrics(clip.cpu().numpy(), gallery.cpu().numpy(), labels,
        [train.records[name].label_id for name in train.available], [corpus.records[name].source_group_id for name in corpus.available],
        [train.records[name].source_group_id for name in train.available])
    report = {"status": "completed", "split": split, "metrics": metric, "retrieval": retrieval, "head_sha256": head_hash,
        "manifest_sha256": file_hash(corpus.manifest), "protocol_signature": canonical_hash(lock),
        "test_used_for_selection": False, "scope": config["experiment"]["evaluation_protocol"]}
    from ..training.runner import runtime_implementation_hash
    report["evaluation_implementation_hash"] = runtime_implementation_hash()
    predictions, qualities = prediction_records(corpus, logits.cpu().numpy(), classes, head_hash, report["protocol_signature"])
    signers = [r["signer_id"] for r in predictions]
    signer_bootstrap = all(s is not None for s in signers) and len(set(signers)) >= 2
    groups = signers if signer_bootstrap else [r["source_group_id"] for r in predictions]
    report["confidence_intervals"] = classification_bootstrap(
        [r["reference_label_id"] == r["prediction_label_id"] for r in predictions], labels,
        groups, samples=ds["bootstrap_samples"], seed=config["experiment"]["seed"],
        resampling_unit="official_signer_id" if signer_bootstrap else "source_group_id")
    if not signer_bootstrap:
        report["confidence_intervals"]["signer_fallback_reason"] = "missing signer IDs or fewer than two independent signers"
    report["quality_buckets"] = {}
    for bucket in ("high", "medium", "low"):
        subset = [i for i, name in enumerate(corpus.available) if qualities[name]["bucket"] == bucket]
        report["quality_buckets"][bucket] = ({"count": len(subset), **classification_metrics(logits.cpu().numpy()[subset], np.asarray(labels)[subset], classes)}
            if subset else {"count": 0, "metrics": None, "reason": "no_samples"})
    write_json(run / f"evaluation.{split}.json", report)
    metrics_path = run / f"metrics.{split}.json"
    existing = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    write_json(metrics_path, {**existing, **{k: v for k, v in report.items() if k != "scope"}, "evaluation_scope": report["scope"]})
    if export_predictions:
        write_json(run / f"predictions.{split}.json", predictions)
        write_jsonl(run / f"predictions.{split}.jsonl", predictions)
    return report
