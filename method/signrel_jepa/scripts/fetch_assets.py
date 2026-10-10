import _bootstrap
import argparse
import json
import subprocess
import urllib.request
from pathlib import Path
from signrel.data.manifest import sha256

VJEPA_COMMIT="204698b45b3712590f06245fbfba32d3be539812"
ASSETS={
    "backbone":("vjepa2_1_vitb_dist_vitG_384.pt",
        "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
        "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d"),
    "pose":("pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",None),
    "hand":("hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",None),
    "face":("face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",None),
    "wlasl":("WLASL_v0.3.json",
        "https://raw.githubusercontent.com/dxli94/WLASL/ac00e6be631c1a2a486621b65f202219f3964d6b/start_kit/WLASL_v0.3.json",None),
}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--assets",default="assets")
    p.add_argument("--which",nargs="+",choices=list(ASSETS),default=list(ASSETS))
    p.add_argument("--repo",default="third_party/vjepa2")
    args=p.parse_args()
    repo=Path(args.repo)
    if not repo.exists():
        repo.mkdir(parents=True)
        for cmd in (["git","init",str(repo)],
                    ["git","-C",str(repo),"remote","add","origin","https://github.com/facebookresearch/vjepa2.git"],
                    ["git","-C",str(repo),"fetch","--depth","1","origin",VJEPA_COMMIT],
                    ["git","-C",str(repo),"checkout","--detach","FETCH_HEAD"]):
            subprocess.run(cmd,check=True)
    commit=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if commit!=VJEPA_COMMIT:
        raise AssertionError("Existing upstream checkout differs from pinned commit")
    # Upstream architecture is used by import only. Verify its explicit license before runtime use.
    if not (repo/"LICENSE").read_text().startswith("MIT License"):
        raise AssertionError("Unexpected V-JEPA license")
    dest=Path(args.assets)
    dest.mkdir(parents=True,exist_ok=True)
    report={"vjepa2_commit":commit,"assets":{}}
    for key in args.which:
        name,url,expected=ASSETS[key]
        out=dest/name
        if not out.exists():
            temp=out.with_suffix(out.suffix+".part")
            print(f"Downloading {url}",flush=True)
            urllib.request.urlretrieve(url,temp)
            if expected and sha256(temp)!=expected:
                raise AssertionError("Downloaded checkpoint hash mismatch")
            temp.replace(out)
        digest=sha256(out)
        if expected and digest!=expected:
            raise AssertionError("Existing checkpoint hash mismatch")
        report["assets"][key]={"url":url,"path":str(out.resolve()),"sha256":digest,"bytes":out.stat().st_size}
    path=Path("reports/assets.json")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=="__main__":
    main()
