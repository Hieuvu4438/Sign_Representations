import _bootstrap
import argparse
import json
from pathlib import Path
import torch
from signrel.config import load_config
from signrel.data.manifest import read_manifest,write_manifest,sha256
from signrel.data.video_dataset import SSLVideoDataset
from signrel.models.region_pooler import token_membership,DEFAULT_EDGES
from signrel.masks.base import SignMaskGenerator


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--config",required=True)
    p.add_argument("--out-manifest",required=True)
    p.add_argument("--out-report",required=True)
    p.add_argument("--draws",type=int,default=4)
    args=p.parse_args()
    cfg=load_config(args.config)
    dc=cfg["data"]
    rows=read_manifest(args.manifest)
    ds=SSLVideoDataset(args.manifest,num_frames=dc["num_frames"],crop_size=dc["crop_size"],
                       sampling_stride=dc.get("sampling_stride",2),augment=True)
    bc=cfg["model"]["backbone"]
    grid=(dc["num_frames"]//bc.get("tubelet",2),dc["crop_size"]//bc.get("patch",16),dc["crop_size"]//bc.get("patch",16))
    masks={name:SignMaskGenerator(mixture={name:1.},min_visible_ratio=cfg["masking"].get("min_visible_ratio",0.25))
           for name in ("random_multiblock","articulator","relation_break","motion_salient")}
    retained,results=[],[]
    for i,r in enumerate(rows):
        failures=[]
        count=0
        for draw in range(args.draws):
            torch.manual_seed(cfg.get("seed",42)+i*1009+draw)
            try:
                sample=ds[i]
                meta=sample["articulator_meta"]
                membership=token_membership(meta["boxes"][None],meta["confidence"][None],grid,
                            mode=cfg.get("articulators",{}).get("membership","intersection"))
                valid=membership.sum((-1,-2))>0
                pair_valid=torch.stack([valid[:,:,a]&valid[:,:,b] for a,b in DEFAULT_EDGES],-1)
                count+=int(pair_valid.sum())
                if not pair_valid.any():
                    raise AssertionError("No valid relation endpoints in sampled clip")
                motion=meta["motion"][None].reshape(1,grid[0],-1,4).mean(2)
                for name,masker in masks.items():
                    masker.sample(grid,membership,motion)
            except (AssertionError,ValueError,OSError) as e:
                failures.append({"draw":draw,"reason":str(e)})
        if not failures:
            retained.append(r)
        results.append({"sample_id":r["sample_id"],"accepted":not failures,"failures":failures,
                        "mean_valid_relations_per_clip":count/args.draws})
        if (i+1)%10==0:
            print(json.dumps({"processed":i+1,"total":len(rows),"retained":len(retained)}),flush=True)
    if not retained:
        raise AssertionError("No clips pass localization/masking preflight")
    write_manifest(retained,args.out_manifest)
    report={"input_sha256":sha256(args.manifest),"output_sha256":sha256(args.out_manifest),
            "input_rows":len(rows),"retained_rows":len(retained),"draws":args.draws,"config":cfg,
            "selection_basis":"structural validity and feasible masks only; no semantic labels or frozen metrics", "clips":results}
    out=Path(args.out_report)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps({"input":len(rows),"retained":len(retained),"rejected":len(rows)-len(retained)}))


if __name__=="__main__":
    main()
