"""Train-label probes with fixed LR trials, validation selection and full-clip inputs."""
from copy import deepcopy
from dataclasses import dataclass
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..cache.hashing import canonical_hash, file_hash
from ..cli import write_json
from ..inference.extract import frozen_visual_embedding, load_encoder, temporal_embedding
from ..models.heads import AttentiveHead, LinearHead
from .metrics import classification_metrics
from .protocols import audit_development, development_corpora, require_pipeline_gate


@dataclass(frozen=True)
class ProbeResult:
    head: object
    model_state: dict | None
    class_ids: tuple[int, ...]
    metrics: dict
    trials: tuple[dict, ...]
    predictions: np.ndarray
    logits: np.ndarray
    train_embeddings: np.ndarray
    val_embeddings: np.ndarray


def corpus_embeddings(corpus, model, encoder_config, downstream, device):
    embeddings = []
    for name in corpus.available:
        bundle = corpus.load(name)
        if downstream["representation"] == "frozen_visual":
            output = frozen_visual_embedding(bundle)
            output = type(output)(*(getattr(output, field).to(device) for field in ("dense", "timestamps", "valid", "clip")))
        else:
            output = temporal_embedding(model, bundle, encoder_config, window_samples=downstream["window_samples"],
                overlap_samples=downstream["overlap_samples"], device=device)
        embeddings.append(output)
    return embeddings


def fit_probe(train, val, config, model=None, encoder_config=None, *, device="cpu"):
    ds = config["downstream"]
    if ds["task"] != "islr":
        raise ValueError("This fit protocol implements ISLR; continuous heads have separate runtime interfaces")
    if ds["representation"] == "frozen_visual" and ds["head"] != "linear":
        raise ValueError("E00 frozen visual baseline uses the registered linear probe")
    if ds["representation"] == "temporal" and model is None:
        raise ValueError("Temporal probe requires an explicit encoder checkpoint")
    labels_present = {train.records[name].label_id for name in train.available}
    if not labels_present or None in labels_present:
        raise ValueError("Downstream train labels are required")
    classes = tuple(sorted(labels_present))
    taxonomy = {label: index for index, label in enumerate(classes)}
    if any(val.records[name].label_id not in taxonomy for name in val.available):
        raise ValueError("Validation contains classes absent from the registered train taxonomy")
    y_train = torch.tensor([taxonomy[train.records[n].label_id] for n in train.available], device=device)
    labels = np.array([val.records[n].label_id for n in val.available])
    y_val = torch.tensor([taxonomy[int(label)] for label in labels], device=device)
    if model is not None:
        model.eval()
    with torch.no_grad():
        train_outputs = corpus_embeddings(train, model, encoder_config, ds, device)
        val_outputs = corpus_embeddings(val, model, encoder_config, ds, device)
    train_x = torch.stack([o.clip.detach() for o in train_outputs])
    val_x = torch.stack([o.clip.detach() for o in val_outputs])
    if ds["l2_normalize"]:
        train_x, val_x = F.normalize(train_x, dim=-1), F.normalize(val_x, dim=-1)
    initial_model = deepcopy(model.online.state_dict()) if model is not None else None
    trials, best = [], None
    for learning_rate in ds["learning_rates"]:
        torch.manual_seed(config["experiment"]["seed"])
        if model is not None:
            model.online.load_state_dict(initial_model)
        head = (AttentiveHead(train_x.shape[-1], len(classes), ds["attentive_heads"])
                if ds["head"] == "attentive" else LinearHead(train_x.shape[-1], len(classes))).to(device)
        groups = [{"params": list(head.parameters()), "lr": learning_rate}]
        finetune = ds["head"] == "finetune"
        if finetune:
            groups.append({"params": list(model.online.parameters()), "lr": ds["encoder_learning_rate"]})
        optimizer = torch.optim.AdamW(groups, weight_decay=ds["weight_decay"])
        trial_best, stale = None, 0
        generator = torch.Generator().manual_seed(config["experiment"]["seed"])
        for epoch in range(ds["max_epochs"]):
            head.train()
            if finetune:
                model.online.train()
            order = torch.randperm(len(y_train), generator=generator).tolist()
            for offset in range(0, len(order), ds["batch_size"]):
                ids = order[offset:offset + ds["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                if ds["head"] == "linear":
                    logits = head(train_x[ids])
                else:
                    outputs = ([temporal_embedding(model, train.load(train.available[i]), encoder_config,
                        window_samples=ds["window_samples"], overlap_samples=ds["overlap_samples"], device=device) for i in ids]
                        if finetune else [train_outputs[i] for i in ids])
                    if finetune:
                        values = torch.stack([o.clip for o in outputs])
                        logits = head(F.normalize(values, dim=-1) if ds["l2_normalize"] else values)
                    else:
                        logits = torch.cat([head((F.normalize(o.dense, dim=-1) if ds["l2_normalize"] else o.dense)[None], o.valid[None]) for o in outputs])
                loss = F.cross_entropy(logits, y_train[ids])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite downstream objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0, error_if_nonfinite=True)
                optimizer.step()
            head.eval()
            if model is not None:
                model.online.eval()
            with torch.no_grad():
                if finetune:
                    current = corpus_embeddings(val, model, encoder_config, ds, device)
                    x = torch.stack([o.clip for o in current])
                    predicted = head(F.normalize(x, dim=-1) if ds["l2_normalize"] else x)
                elif ds["head"] == "linear":
                    predicted = head(val_x)
                else:
                    predicted = torch.cat([head((F.normalize(o.dense, dim=-1) if ds["l2_normalize"] else o.dense)[None], o.valid[None]) for o in val_outputs])
                logits_array = predicted.cpu().numpy()
                metrics = classification_metrics(logits_array, labels, classes)
                score = metrics["per_class_top1"]
                val_loss = float(F.cross_entropy(predicted, y_val))
            criterion = (score, -val_loss)
            if trial_best is None or criterion > trial_best["criterion"]:
                trial_best = {"criterion": criterion, "head": deepcopy(head.state_dict()), "metrics": metrics,
                    "logits": logits_array, "epoch": epoch + 1, "model": deepcopy(model.online.state_dict()) if finetune else None}
                stale = 0
            else:
                stale += 1
            if stale >= ds["patience"]:
                break
        trials.append({"learning_rate": learning_rate, "epochs_attempted": epoch + 1, "selected_epoch": trial_best["epoch"],
                       "validation_per_class_top1": trial_best["metrics"]["per_class_top1"], "validation_loss": -trial_best["criterion"][1]})
        if best is None or trial_best["criterion"] > best["criterion"]:
            best = {**trial_best, "head_module": head}
    selected_head = best["head_module"]
    selected_head.load_state_dict(best["head"])
    selected_head.eval()
    if model is not None:
        model.online.load_state_dict(best["model"] if best["model"] is not None else initial_model)
    if ds["head"] == "finetune":
        with torch.no_grad():
            train_x = torch.stack([o.clip for o in corpus_embeddings(train, model, encoder_config, ds, device)])
            val_x = torch.stack([o.clip for o in corpus_embeddings(val, model, encoder_config, ds, device)])
    return ProbeResult(selected_head, best["model"], classes, best["metrics"], tuple(trials),
        np.asarray(classes)[best["logits"].argmax(axis=-1)], best["logits"], train_x.cpu().numpy(), val_x.cpu().numpy())


def downstream_fit(resolved, encoder=None, device="cpu"):
    from ..training.runner import runtime_implementation_hash
    c = resolved.values
    torch.set_num_threads(1)
    train, val = development_corpora(resolved)
    audit = audit_development(resolved, train, val)
    gate_hash = require_pipeline_gate(resolved, train, val)
    model, encoder_config, encoder_hash = None, None, None
    if c["downstream"]["representation"] == "temporal":
        if encoder is None:
            raise ValueError("--encoder is required for temporal downstream fitting")
        model, identity, encoder_config, encoder_hash = load_encoder(encoder, device)
        if identity != train.identity:
            raise ValueError("Encoder and downstream cache feature identities differ")
    identity = {"config_hash": resolved.hash, "encoder_hash": encoder_hash, "train": train.fingerprint,
                "val": val.fingerprint, "implementation_hash": runtime_implementation_hash(), "pipeline_gate_sha256": gate_hash}
    output = resolved.path(c["downstream"]["output_dir"]) / canonical_hash(identity)[:12]
    output.mkdir(parents=True, exist_ok=False)
    from ..cli import doctor
    from ..training.provenance import write_run_provenance
    write_json(output / "environment.json", doctor())
    write_run_provenance(output, resolved, train, val, audit,
        {**identity, "runtime_implementation_hash": identity["implementation_hash"], "cache_identity": train.identity})
    started = time.monotonic()
    result = fit_probe(train, val, c, model, encoder_config, device=device)
    torch.save({"schema_version": 1, "head": result.head.state_dict(), "online": result.model_state,
        "classes": result.class_ids, "dimension": result.train_embeddings.shape[1], "encoder_path": str(Path(encoder).resolve()) if encoder else None,
        "encoder_hash": encoder_hash, "config": c, "identity": identity}, output / "head.pt")
    write_json(output / "resolved_config.json", resolved.provenance())
    write_json(output / "identity.json", identity)
    predictions = [{"sample_id": name, "source_group_id": val.records[name].source_group_id,
        "reference_id": val.records[name].label_id, "predicted_id": int(result.predictions[i]),
        "logits": result.logits[i].tolist(), "model_hash": file_hash(output / "head.pt"), "encoder_initialization_hash": encoder_hash,
        "class_ids": list(result.class_ids)} for i, name in enumerate(val.available)]
    write_json(output / "predictions.val.json", predictions)
    np.savez(output / "embeddings.npz", train=result.train_embeddings, val=result.val_embeddings,
             train_ids=np.array(train.available), val_ids=np.array(val.available))
    centered = result.train_embeddings - result.train_embeddings.mean(axis=0)
    np.savez(output / "feature_spectrum.npz", singular_values=np.linalg.svd(centered, compute_uv=False),
             train_embeddings=result.train_embeddings)
    (output / "logs.jsonl").write_text("".join(json.dumps({"stage": "downstream_lr_trial_summary", **trial}) + "\n" for trial in result.trials))
    (output / "mask_diagnostics.jsonl").write_text(json.dumps({"status": "not_applicable", "reason": "supervised unmasked downstream fitting"}) + "\n")
    report = {"status": "completed", "task": "islr", "split": "val", "metrics": result.metrics,
        "trials": list(result.trials), "head": c["downstream"]["head"], "representation": c["downstream"]["representation"],
        "identity": identity, "test_used_for_selection": False, "output": str(output), "fit_seconds": time.monotonic() - started}
    write_json(output / "metrics.val.json", report)
    write_json(output / "protocol.lock.json", {"schema_version": 1, "head_sha256": file_hash(output / "head.pt"),
        "identity": identity, "selection_split": "val", "downstream_config": c["downstream"], "class_ids": list(result.class_ids)})
    (output / "decision_memo.md").write_text(
        f"# Downstream selection: {c['experiment']['id']} seed{c['experiment']['seed']}\n\n"
        f"Validation PC Top1: {result.metrics['per_class_top1']}. Selected using registered LR trials, "
        "PC Top1 and validation loss. No test selection. Training logs are per-trial summaries; "
        "this single run does not establish a representation contribution.\n")
    if encoder:
        pointer = Path(encoder).resolve().parent.parent / "downstream_selection.json"
        if not pointer.exists():
            write_json(pointer, {"run": str(output.resolve()), "head_sha256": file_hash(output / "head.pt")})
    return report
