"""Frozen linear probe on pooled features: Linear(768, C) + cross-entropy, locked config.

Epoch selection uses the val split only. When eval_split == "test", the epoch is
re-selected on val inside the same run and test is scored once at that epoch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .retrieval import load_features

DEFAULT_PROBE = {"lr": 1e-3, "weight_decay": 1e-4, "epochs": 60, "batch_size": 1024, "seed": 0,
                 "feature_key": "pooled_features", "standardize": False}


def _acc(logits, y):
    pred = logits.argmax(1)
    micro = float((pred == y).float().mean())
    classes = torch.unique(y)
    macro = float(torch.stack([(pred[y == c] == c).float().mean() for c in classes]).mean())
    return micro, macro, pred


def run_probe(ctx, features_dir: Path, eval_split: str, probe_cfg: dict) -> None:
    cfg = dict(DEFAULT_PROBE, **probe_cfg)
    if set(cfg) != set(DEFAULT_PROBE):
        raise ValueError(f"unknown probe keys {set(cfg) - set(DEFAULT_PROBE)}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = load_features(features_dir, "train", cfg["feature_key"])
    va = load_features(features_dir, "val", cfg["feature_key"])
    te = load_features(features_dir, eval_split, cfg["feature_key"]) if eval_split != "val" else None
    C = int(max(tr["label"].max(), va["label"].max()) + 1)
    X = torch.as_tensor(tr["x"], device=dev)
    Y = torch.as_tensor(tr["label"], device=dev)
    Xv, Yv = torch.as_tensor(va["x"], device=dev), torch.as_tensor(va["label"], device=dev)
    mu = X.mean(0, keepdim=True) if cfg["standardize"] else torch.zeros_like(X[:1])
    sd = X.std(0, keepdim=True).clamp_min(1e-6) if cfg["standardize"] else torch.ones_like(X[:1])
    X, Xv = (X - mu) / sd, (Xv - mu) / sd  # train-only statistics
    g = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    lin = torch.nn.Linear(X.shape[1], C).to(dev)
    opt = torch.optim.AdamW(lin.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    hist, best, best_state = [], -1.0, None
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(len(X), generator=g).to(dev)
        lin.train()
        for s in range(0, len(X), cfg["batch_size"]):
            idx = perm[s : s + cfg["batch_size"]]
            loss = F.cross_entropy(lin(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        lin.eval()
        with torch.no_grad():
            micro, macro, _ = _acc(lin(Xv), Yv)
        hist.append({"epoch": ep + 1, "val_top1": micro, "val_macro_top1": macro})
        if micro > best:
            best = micro
            best_state = {k: v.clone() for k, v in lin.state_dict().items()}
            best_ep = ep + 1
    lin.load_state_dict(best_state)
    lin.eval()
    res = {"config": cfg, "selected_epoch": best_ep, "selection": "max val top1", "history": hist, "num_classes": C,
           "train_size": int(len(X)), "label_budget": "all train labels"}
    with torch.no_grad():
        micro, macro, pred = _acc(lin(Xv), Yv)
        res["val"] = {"top1": micro, "macro_top1": macro, "num_queries": int(len(Yv))}
        out_split, out_data, out_pred = "val", va, pred
        if te is not None:
            Xt, Yt = (torch.as_tensor(te["x"], device=dev) - mu) / sd, torch.as_tensor(te["label"], device=dev)
            micro, macro, pred = _acc(lin(Xt), Yt)
            res[eval_split] = {"top1": micro, "macro_top1": macro, "num_queries": int(len(Yt))}
            out_split, out_data, out_pred = eval_split, te, pred
    with open(ctx.dir / f"probe_predictions_{out_split}.jsonl", "x") as f:
        for sid, lab, sig, p in zip(out_data["sample_id"], out_data["label"], out_data["signer"], out_pred.cpu().numpy()):
            f.write(json.dumps({"sample_id": str(sid), "label": int(lab), "signer": str(sig), "pred": int(p)}) + "\n")
    ctx.write_json("probe_metrics.json", res)
    print(json.dumps({k: res[k] for k in res if k in ("val", "test", "selected_epoch")}, indent=2))
