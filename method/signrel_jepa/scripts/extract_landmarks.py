import _bootstrap
import argparse
import json
from pathlib import Path
from signrel.data.manifest import read_manifest,write_manifest,sha256
from signrel.preprocessing.landmarks import LandmarkExtractor,CACHE_VERSION


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--out-root",default="cache/landmarks")
    p.add_argument("--assets",default="assets")
    p.add_argument("--out-manifest",required=True)
    p.add_argument("--limit",type=int)
    args = p.parse_args()
    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    extractor = LandmarkExtractor(args.assets)
    reports = []
    try:
        for i,row in enumerate(rows):
            # Content-address source path prevents collisions, metadata retains byte hash.
            import hashlib
            key = hashlib.sha256(str(Path(row["video_path"]).resolve()).encode()).hexdigest()
            dest = Path(args.out_root)/f"{key}.npz"
            if dest.exists():
                import numpy as np
                with np.load(dest,allow_pickle=False) as z:
                    meta = json.loads(str(z["metadata"]))
                if meta["source_sha256"]!=sha256(row["video_path"]) or meta["models"]!=extractor.model_hashes:
                    raise AssertionError("Existing landmark cache source/model hash mismatch")
                if meta["version"]!=CACHE_VERSION:
                    raise AssertionError("Existing landmark cache version mismatch; use a new out-root")
                report = {"reused":True,**meta}
            else:
                report = extractor.extract(row["video_path"],dest)
            row["landmark_path"] = str(dest.resolve())
            reports.append({"sample_id":row["sample_id"],**report})
            print(json.dumps({"completed":i+1,"total":len(rows),"sample_id":row["sample_id"]}),flush=True)
    finally:
        extractor.close()
    write_manifest(rows,args.out_manifest)
    Path(args.out_manifest+".landmark_report.json").write_text(json.dumps(reports,indent=2))


if __name__ == "__main__":
    main()
