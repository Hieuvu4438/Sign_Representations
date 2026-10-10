"""Deterministic matched ASL Citizen evaluation corpus; all choices made before features."""
import _bootstrap
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from signrel.data.manifest import write_manifest,sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="/home/dongvk/datasets/ASL_Citizen")
    p.add_argument("--out-root",default="manifests/eval_pilot")
    p.add_argument("--classes",type=int,default=40)
    p.add_argument("--train-per-class",type=int,default=4)
    p.add_argument("--query-per-class",type=int,default=2)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    root=Path(args.root)
    splits={}
    for split in ("train","val","test"):
        grouped=defaultdict(list)
        with (root/"splits"/f"{split}.csv").open(newline="") as f:
            for r in csv.DictReader(f):
                path=root/"videos"/r["Video file"]
                if not path.exists():
                    continue
                if r.get("ASL-LEX Code") in {None,"","NA"}:
                    continue
                grouped[r["Gloss"]].append({"sample_id":f"asl_citizen:{path.stem}","video_path":str(path),
                    "source_video_id":f"asl_citizen:{path.stem}","signer_id":r["Participant ID"],
                    "gloss":r["Gloss"],"asllex":r["ASL-LEX Code"],"split":split})
        splits[split]=grouped
    eligible=sorted(set.intersection(*(set(splits[s]) for s in splits)))
    eligible=[c for c in eligible if len(splits["train"][c])>=args.train_per_class and
              min(len(splits["val"][c]),len(splits["test"][c]))>=args.query_per_class]
    if len(eligible)<args.classes:
        raise ValueError("Not enough classes with complete split and lexical mapping coverage")
    rng=random.Random(args.seed)
    classes=sorted(rng.sample(eligible,args.classes))
    out=Path(args.out_root)
    hashes={}
    signer_sets={}
    for split,groups in splits.items():
        n=args.train_per_class if split=="train" else args.query_per_class
        rows=[r for c in classes for r in rng.sample(groups[c],n)]
        path=out/f"asl_citizen_{split}.jsonl"
        write_manifest(rows,path)
        hashes[split]=sha256(path)
        signer_sets[split]={r["signer_id"] for r in rows}
    for a,b in (("train","val"),("train","test"),("val","test")):
        if signer_sets[a]&signer_sets[b]:
            raise AssertionError("Official ASL Citizen pilot signer split overlaps")
    out.joinpath("protocol.json").write_text(json.dumps({"arguments":vars(args),"classes":classes,
        "eligible_classes":len(eligible),"manifest_sha256":hashes,
        "signers":{k:sorted(v) for k,v in signer_sets.items()},
        "selection":"random lexical classes with train/val/test and ASL-LEX code coverage, before extracting features",
        "scope":"pilot only; not a full benchmark or tuned test-set claim"},indent=2))
    print(json.dumps({"classes":classes,"hashes":hashes}))


if __name__=="__main__":
    main()
