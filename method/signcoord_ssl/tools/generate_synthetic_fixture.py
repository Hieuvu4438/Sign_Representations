"""Known RGB/pose PTS, holds, sinusoidal coordination and invalid segments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"src"))
from signcoord.provenance import atomic_json, sha256_file
from signcoord.targets.schema import SCHEMA_HASH


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--samples",type=int,default=40)
    args=p.parse_args()
    root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    (root/"videos").mkdir(); (root/"pose").mkdir(); (root/"manifests").mkdir()
    rng=np.random.default_rng(3407)
    manifests={"ssl_train":[],"ssl_dev":[]}; labels={}
    for i in range(args.samples):
        sid=f"fixture_{i:03d}"; split="ssl_train" if i < args.samples*4//5 else "ssl_dev"
        fps=12 if i%17 == 16 else 25
        n=12 if i%19 == 18 else 40
        times=np.arange(n,dtype=np.float64)/fps
        xy=np.broadcast_to(np.array([32.,28.]),(n,133,2)).copy()
        xy[:,5]=[23,30]; xy[:,6]=[41,30]
        omega=2*np.pi*(2+i%4)
        hold=i%13 == 12
        l=np.zeros(n) if hold else 6*np.sin(omega*times)
        r=np.zeros(n) if hold else 5*np.sin(omega*times-(i%5)*.35)
        head=np.zeros(n) if hold else 3*np.sin(omega*times+.4)
        xy[:,9,0]=24+l; xy[:,9,1]=40 if hold else 40+2*np.cos(omega*times)
        xy[:,10,0]=40+r; xy[:,10,1]=40 if hold else 40+2*np.cos(omega*times-.3)
        xy[:,0,0]=32+head; xy[:,0,1]=18
        for j in range(21):
            xy[:,91+j]=xy[:,9]+[(j%5)*.4,-(j//5)*.7]
            xy[:,112+j]=xy[:,10]+[(j%5)*.4,-(j//5)*.7]
        for j in range(68):
            xy[:,23+j]=xy[:,0]+[np.cos(j*2*np.pi/68)*4,np.sin(j*2*np.pi/68)*4]
        conf=np.ones((n,133),dtype=np.float32)*.9
        observed=np.ones((n,133),dtype=bool); status=np.zeros((n,133),dtype=np.uint8)
        if i%11 == 10:
            observed[9:13,9]=False; status[9:13,9]=1
        video=root/"videos"/(sid+".mp4")
        with av.open(str(video),"w") as out:
            stream=out.add_stream("libx264",rate=fps)
            stream.width=stream.height=64; stream.pix_fmt="yuv420p"
            for t in range(n):
                # Instance-specific constant background avoids accidental exact
                # duplicate videos from periodic motion parameters across splits.
                im=Image.new("RGB",(64,64),(20+(i%4)*30,20+(i*3)%200,20)); draw=ImageDraw.Draw(im)
                for j,color in ((9,(240,60,30)),(10,(30,240,60)),(0,(60,30,240))):
                    x,y=xy[t,j]; draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
                frame=av.VideoFrame.from_ndarray(np.asarray(im),format="rgb24")
                for packet in stream.encode(frame): out.mux(packet)
            for packet in stream.encode(): out.mux(packet)
        digest=sha256_file(video)
        meta={"schema_version":"pose_cache_v1","schema_hash":SCHEMA_HASH,"media_sha256":digest,
              "extractor_id":"known_synthetic_generator_v1","coordinate_units":"pixels",
              "score_semantics":"synthetic_bounded_quality","tracking_policy":"one_known_synthetic_signer",
              "tracking_identity_verified":True}
        xy[~observed]=np.nan
        np.savez_compressed(root/"pose"/(sid+".npz"),xy_px=xy.astype(np.float32),confidence=conf,
                            observed=observed,status=status,pts_s=times,metadata=json.dumps(meta),track_ids=np.zeros(n,dtype=np.int64))
        record={"schema_version":"video_manifest_v1","sample_id":sid,"dataset_id":"synthetic_fixture",
              "source_video_id":sid,"source_group_id":sid,"media_path":str(video),"media_sha256":digest,
              "start_ms":0,"end_ms":n/fps*1000,"signer_id":"synthetic_train" if split=="ssl_train" else "synthetic_dev",
              "sign_language":None,"split":split,"pose_cache_key":sid+".npz","license_ref":"synthetic_generated",
              "label_ref":"labels.json","duplicate_group_id":None,"access_status":"available"}
        manifests[split].append(record); labels[sid]=i%4
    for split,records in manifests.items():
        (root/"manifests"/(split+".jsonl")).write_text("".join(json.dumps(r)+"\n" for r in records))
    atomic_json(root/"labels.json",labels)
    base=yaml.safe_load((PROJECT/"configs/base.yaml").read_text())
    base["run_dir"]=str(root/"runs/A7_fixture")
    base["model"].update(backbone="tiny_fixture",initialization_mode="random_init",input_frames=32)
    base["model"]["decoder"].update(width=32,blocks=1,heads=4)
    base["model"]["regularization_projector"].update(hidden=32,out_dim=16)
    base["data"].update(ssl_train_manifest=str(root/"manifests/ssl_train.jsonl"),
       ssl_dev_manifest=str(root/"manifests/ssl_dev.jsonl"),pose_cache_root=str(root/"pose"),
       target_stats_path=str(root/"stats/target_stats.json"),phase_calibration_path=str(root/"stats/phase_calibration.json"))
    base["data"]["sampler"]["spatial_size"]=[32,32]
    base["data"]["augmentation"]["color_jitter"]=0.0
    base["loss"]["relation"].update(warmup_zero_updates=1,ramp_end_update=3)
    base["train"].update(device="cpu",precision="fp32",microbatch_per_gpu=4,accumulation_steps=1,
          max_updates=8,dataloader_workers=0,warmup_updates=2,save_every_updates=4,eval_every_updates=4)
    (root/"fixture_smoke.yaml").write_text(yaml.safe_dump(base,sort_keys=False))
    print(json.dumps({"fixture":str(root),"config":str(root/"fixture_smoke.yaml"),"counts":{k:len(v) for k,v in manifests.items()}}))


if __name__ == "__main__": main()
