"""Evaluate a selected classifier checkpoint on one split (test only after the configuration is locked).

Fine-tuned model: --checkpoint runs/<ft>/checkpoints/best_val.pt --split test
Linear probe:     --checkpoint runs/<lin>/checkpoints/best_val.pt --embeddings runs/<ssl>/embeddings/test.npz
Unusable queries keep their place in the denominator (uniform logits, class-id tie order).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from asm_jepa.cli.common import run
from asm_jepa.data.dataset import LabeledDataset, PoseStoreSource, SampleBuilder
from asm_jepa.data.manifest import SPLITS, load_split
from asm_jepa.evaluation.classification import classification_metrics
from asm_jepa.evaluation.embeddings import load_embeddings
from asm_jepa.features.stats import load_stats
from asm_jepa.training.checkpoint import load_checkpoint
from asm_jepa.utils import now, resolve, sha256_file, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=SPLITS)
    ap.add_argument("--embeddings", help="frozen embeddings (linear probe only)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="debug only: first N clips (report marked partial)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    ck = load_checkpoint(args.checkpoint)
    report = {"created": now(), "checkpoint": args.checkpoint, "checkpoint_sha256": sha256_file(args.checkpoint),
              "kind": ck["kind"], "split": args.split, "readout": "classifier_ranking (not frozen retrieval)"}
    if ck["kind"] == "linear_probe":
        if not args.embeddings:
            raise ValueError("--embeddings required for a linear probe")
        e = load_embeddings(args.embeddings, ck["config"]["feature_key"])
        if e["meta"]["checkpoint_sha256"] != ck["encoder_meta"]["checkpoint_sha256"]:
            raise ValueError("embeddings come from a different encoder checkpoint than the probe was trained on")
        if e["meta"]["split"] != args.split:
            raise ValueError(f"embeddings split {e['meta']['split']} != --split {args.split}")
        head = torch.nn.Linear(ck["feature_dim"], ck["n_classes"])
        head.load_state_dict(ck["head"])
        with torch.no_grad():
            logits = head(torch.as_tensor(e["features"])).numpy()
        y, usable, signers = e["class_ids"], e["usable"], e["signer_ids"]
        quality = e.get("quality")
        report["encoder_checkpoint_sha256"] = e["meta"]["checkpoint_sha256"]
    elif ck["kind"] == "finetune_islr":
        from asm_jepa.training.classifier import Classifier

        mcfg = ck["model_config"]
        root = resolve(ck["config"]["data"]["processed_root"])
        stats_path = root / mcfg["data"]["stats_path"]
        if sha256_file(stats_path) != ck["stats_sha256"]:
            raise ValueError("feature statistics changed since fine-tuning")
        stats = load_stats(stats_path, len(mcfg["features"]["interval_widths_sec"]))
        rows = load_split(root, args.split)
        if args.limit:
            rows = rows[: args.limit]
            report["partial_limit"] = args.limit
        ds = LabeledDataset(PoseStoreSource(rows, root), SampleBuilder(mcfg, stats), [r["class_id"] for r in rows],
                            "eval")
        model = Classifier(mcfg, ck["n_classes"], ck["config"]["head_dropout"]).to(args.device)
        model.load_state_dict(ck["model"])
        model.eval()
        loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=args.workers)
        out, usable, quality = [], [], []
        with torch.no_grad():
            for batch in loader:
                b = {k: v.to(args.device) for k, v in batch.items()}
                lg, present = model(b)
                out.append(lg.float().cpu().numpy())
                usable.append((b["usable"] & present.any(1)).cpu().numpy())
                quality.append(b["token_weight"][:, :4 * model.backbone.n_patches].float().mean(1).cpu().numpy())
        logits, usable, quality = np.concatenate(out), np.concatenate(usable), np.concatenate(quality)
        y = np.array([r["class_id"] for r in rows])
        signers = [r["signer_id"] for r in rows]
        report["init"] = ck["init"]
    else:
        raise ValueError(f"unknown checkpoint kind {ck['kind']}")
    m, ranks = classification_metrics(logits, y, signers=signers, usable=usable, quality=quality)
    report["metrics"] = m
    write_json(args.output, report)
    print(f"{args.split}: top1 {m['top1_instance_pct']:.2f} top5 {m['top5_instance_pct']:.2f} "
          f"class-macro top1 {m['top1_class_macro_pct']:.2f} coverage {m['coverage']:.4f}")
    return 0


if __name__ == "__main__":
    run(main)
