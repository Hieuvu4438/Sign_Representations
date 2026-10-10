import _bootstrap
import argparse
import json
from pathlib import Path
from signrel.data.manifest import read_manifest,write_manifest,sha256
from signrel.data.leakage import audit_and_filter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain",required=True)
    p.add_argument("--eval",nargs="+",required=True)
    p.add_argument("--key",default="source_video_id",choices=["source_video_id"])
    p.add_argument("--out",required=True)
    p.add_argument("--clean-out")
    p.add_argument("--hash-files",action="store_true")
    args = p.parse_args()
    eval_rows = [r for path in args.eval for r in read_manifest(path)]
    clean,report = audit_and_filter(read_manifest(args.pretrain),eval_rows,args.hash_files)
    report.update(pretrain_sha256=sha256(args.pretrain),eval_sha256={path:sha256(path) for path in args.eval})
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    if args.clean_out:
        write_manifest(clean,args.clean_out)
    print(json.dumps({k:v for k,v in report.items() if k!="overlaps"}))
    if not report["clean"] and not args.clean_out:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
