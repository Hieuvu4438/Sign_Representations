import _bootstrap
import argparse
import csv
import json
import re
from pathlib import Path
import cv2
from signrel.data.manifest import write_manifest,sha256
from signrel.data.leakage import youtube_id


def probe(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or count <= 0:
            raise ValueError(f"Invalid FPS/frame count {path}")
        return {"fps":fps,"num_frames":count,"duration_sec":count/fps,
                "width":int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),"height":int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()


def main():
    p = argparse.ArgumentParser(description="Create label-free SSL or separate annotated evaluation records")
    p.add_argument("--dataset",required=True,choices=["how2sign","asl_citizen","wlasl","common","video_folder"])
    p.add_argument("--root")
    p.add_argument("--metadata")
    p.add_argument("--purpose",choices=["ssl","eval"],default="ssl")
    p.add_argument("--split",default="train")
    p.add_argument("--limit",type=int)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--probe",action="store_true")
    p.add_argument("--hash-files",action="store_true")
    p.add_argument("--out",required=True)
    args = p.parse_args()
    root = Path(args.root).resolve() if args.root else None
    rows = []
    if args.dataset in {"how2sign","video_folder"}:
        if root is None:
            p.error("--root is required")
        for path in sorted(root.rglob("*.mp4")):
            source = re.match(r"^([A-Za-z0-9_-]{11})_",path.stem) if args.dataset=="how2sign" else None
            sid = source[1] if source else f"{args.dataset}:{path.stem}"
            rows.append({"sample_id":f"{args.dataset}:{path.stem}","video_path":str(path),
                         "source_video_id":sid,"source_url":f"https://www.youtube.com/watch?v={sid}" if source else None})
    elif args.dataset == "asl_citizen":
        root = root or Path("/home/dongvk/datasets/ASL_Citizen")
        metadata = Path(args.metadata) if args.metadata else root/"splits"/f"{args.split}.csv"
        with metadata.open(newline="") as f:
            for row in csv.DictReader(f):
                path = root/"videos"/row["Video file"]
                r = {"sample_id":f"asl_citizen:{path.stem}","video_path":str(path),
                     "source_video_id":f"asl_citizen:{path.stem}"}
                if args.purpose == "eval":
                    r.update(gloss=row["Gloss"],signer_id=row["Participant ID"],asllex=row.get("ASL-LEX Code"))
                rows.append(r)
    elif args.dataset == "wlasl":
        if not args.metadata:
            p.error("--metadata must be the official WLASL_v0.3.json, including source URLs")
        root = root or Path("/home/dongvk/datasets/WLASL2000/WLASL2000")
        data = json.loads(Path(args.metadata).read_text())
        for item in data:
            for instance in item["instances"]:
                if args.split != "all" and instance["split"] != args.split:
                    continue
                vid = instance["video_id"]
                url = instance.get("url")
                r = {"sample_id":f"wlasl:{vid}","video_path":str(root/f"{vid}.mp4"),
                     "source_video_id":youtube_id(url) or f"wlasl:{vid}","source_url":url,
                     "official_frame_start":instance.get("frame_start"),"official_frame_end":instance.get("frame_end"),
                     "interval_policy":"local released clip; no unverified annotation recrop"}
                if args.purpose == "eval":
                    r.update(gloss=item["gloss"],signer_id=str(instance.get("signer_id")),split=instance["split"])
                rows.append(r)
    else:
        if not args.metadata:
            p.error("--metadata required for common manifest import")
        from signrel.data.manifest import read_manifest
        for row in read_manifest(args.metadata):
            if row.get("official_split") != args.split and args.split!="all":
                continue
            path = Path(row["source_root"])/row["relative_path"]
            r = {"sample_id":row["sample_id"],"video_path":str(path),
                 "source_video_id":row.get("parent_source_id") or row["sample_id"]}
            if args.purpose == "eval":
                r.update(gloss=row.get("gloss_raw"),signer_id=row.get("signer_id"),
                         asllex=row.get("lexical_mapping_id"),split=row["official_split"])
            rows.append(r)
    missing = [r for r in rows if not Path(r["video_path"]).exists()]
    # Evaluation source audit retains missing-video records so unavailable clips still exclude sources.
    if args.purpose == "ssl":
        rows = [r for r in rows if Path(r["video_path"]).exists()]
    if args.limit and len(rows)>args.limit:
        import random
        rows = random.Random(args.seed).sample(rows,args.limit)
        rows.sort(key=lambda r:r["sample_id"])
    for r in rows:
        r.setdefault("split","ssl_train" if args.purpose=="ssl" else args.split)
        if args.probe and Path(r["video_path"]).exists():
            r.update(probe(r["video_path"]))
        if args.hash_files and Path(r["video_path"]).exists():
            r["file_sha256"] = sha256(r["video_path"])
    write_manifest(rows,args.out)
    report = {"dataset":args.dataset,"purpose":args.purpose,"split":args.split,"rows":len(rows),
              "missing_files":len(missing),"manifest_sha256":sha256(args.out),
              "missing_samples":[r["sample_id"] for r in missing],"arguments":vars(args)}
    Path(args.out+".report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="missing_samples"}))


if __name__ == "__main__":
    main()
