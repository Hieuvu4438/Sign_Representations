import _bootstrap
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from signrel.data.manifest import read_manifest,sha256
from signrel.data.video_dataset import decode_video
from signrel.data.transforms import transform_clip
from signrel.eval.extract import load_encoder,rgb_embedding
from signrel.config import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint")
    p.add_argument("--config")
    p.add_argument("--manifest",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--precision",choices=["fp32","bf16"],default="bf16")
    p.add_argument("--windows",type=int,default=1)
    p.add_argument("--limit",type=int)
    p.add_argument("--baseline",choices=["signrel","signrep"],default="signrel")
    p.add_argument("--signrep-repo",default="../../third_party/SignRep")
    p.add_argument("--signrep-checkpoint",default="../../checkpoints/signrep/ckpt.pt")
    args = p.parse_args()
    if not args.checkpoint and not args.config and args.baseline=="signrel":
        p.error("--checkpoint or --config required")
    cfg = load_config(args.config) if args.config else None
    if args.checkpoint:
        cfg = torch.load(args.checkpoint,map_location="cpu",weights_only=False)["config"]
    seed = cfg.get("seed",42) if cfg else 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    dc = cfg["data"] if cfg else {"num_frames":16,"crop_size":224,"sampling_stride":2}
    if args.baseline=="signrep":
        from signrel.eval.baselines import SignRepBaseline
        encoder = SignRepBaseline(args.signrep_repo,args.signrep_checkpoint).to(args.device).eval()
        dc = {"num_frames":16,"crop_size":224,"sampling_stride":2}
    else:
        encoder = load_encoder(args.checkpoint,cfg["model"]["backbone"],args.device)
    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    if args.windows<1:
        p.error("--windows must be positive")
    vectors,retained,errors = [],[],[]
    with torch.no_grad():
        for i,row in enumerate(rows):
            try:
                # Uniform windows cover different parts of the licensed local sign video.
                _,_,fps,count = decode_video(row["video_path"],dc["num_frames"],dc.get("sampling_stride",2),
                                             row.get("start_sec"),row.get("end_sec"))
                start = row.get("start_sec") or 0.
                end = row.get("end_sec") or count/fps
                span = (dc["num_frames"]-1)*dc.get("sampling_stride",2)/fps+1/fps
                starts = np.linspace(start,max(start,end-span),args.windows) if args.windows>1 else [None]
                embeddings = []
                for ws in starts:
                    video,_,_,_ = decode_video(row["video_path"],dc["num_frames"],dc.get("sampling_stride",2),
                                               ws if ws is not None else start,
                                               min(end,ws+span) if ws is not None else end)
                    x,_,_ = transform_clip(video,size=dc["crop_size"],crop=row.get("signer_crop") or (0,0,1,1))
                    with torch.autocast(torch.device(args.device).type,dtype=torch.bfloat16,enabled=args.precision=="bf16"):
                        z = rgb_embedding(encoder,x[None].to(args.device))
                    embeddings.append(z.float().cpu().numpy()[0])
                feature = np.mean(embeddings,axis=0)
                feature /= max(np.linalg.norm(feature),1e-12)
                if not np.isfinite(feature).all():
                    raise FloatingPointError("Non-finite frozen embedding")
                vectors.append(feature)
                retained.append({**row,"file_sha256":sha256(row["video_path"])})
            except (ValueError,OSError) as e:
                errors.append({"sample_id":row["sample_id"],"error":str(e)})
            if (i+1)%20==0 or i==0:
                print(json.dumps({"processed":i+1,"total":len(rows),"retained":len(retained),"errors":len(errors)}),flush=True)
    if not vectors:
        raise ValueError("No decodable evaluation videos")
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,features=np.stack(vectors),sample_ids=np.array([r["sample_id"] for r in retained]),
                        rows_json=np.array([json.dumps(r) for r in retained]))
    report = {"arguments":vars(args),"manifest_sha256":sha256(args.manifest),
              "checkpoint_sha256":sha256(args.checkpoint) if args.checkpoint else None,
              "upstream_init":encoder.upstream_init,"restored_checkpoint":getattr(encoder,"restored_checkpoint",None),"total_rows":len(rows),"retained_rows":len(retained),
              "errors":errors,"shape":[len(retained),encoder.dim],"rgb_only":True,
              "pooling":"mean tokens, LayerNorm, L2; mean normalized window embeddings, L2",
              "num_frames":dc["num_frames"],"crop_size":dc["crop_size"]}
    report["seed"] = seed
    out.with_suffix(".report.json").write_text(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
