"""Upgrade only owned v1 face metadata; raw videos and v1 caches remain unchanged."""
import _bootstrap
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from signrel.data.manifest import read_manifest,write_manifest,sha256
from signrel.preprocessing.landmarks import head_crop,CACHE_VERSION
from signrel.preprocessing.articulator_boxes import boxes_from_landmarks,motion_from_landmarks


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--out-root",default="cache/landmarks_v2")
    p.add_argument("--out-manifest",required=True)
    p.add_argument("--assets",default="assets")
    args=p.parse_args()
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    detector=vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(Path(args.assets)/"face_landmarker.task"),
                                          delegate=mp.tasks.BaseOptions.Delegate.CPU),num_faces=1))
    rows=read_manifest(args.manifest)
    reports=[]
    try:
        for i,row in enumerate(rows):
            source=Path(row["landmark_path"])
            dest=Path(args.out_root)/source.name
            with np.load(source,allow_pickle=False) as z:
                cache={k:z[k].copy() for k in z.files}
            metadata=json.loads(str(cache["metadata"]))
            if metadata["source_sha256"]!=sha256(row["video_path"]):
                raise AssertionError("Source video changed since v1 cache")
            if metadata["version"]!= "mediapipe_tasks_v1_pose_wrist_assignment":
                raise AssertionError("This migration accepts exactly v1 caches")
            if dest.exists():
                with np.load(dest,allow_pickle=False) as z:
                    existing=json.loads(str(z["metadata"]))
                    if existing["version"]!=CACHE_VERSION or existing["source_sha256"]!=metadata["source_sha256"]:
                        raise AssertionError("Existing v2 cache mismatch")
                    reports.append({"sample_id":row["sample_id"],"valid_fraction":(z["box_confidence"]>=0.5).mean(0).tolist()})
                row["landmark_path"]=str(dest.resolve())
                continue
            cap=cv2.VideoCapture(row["video_path"])
            width,height=cache["source_size"].tolist()
            cache["xy"][:,42:520]=0
            cache["confidence"][:,42:520]=0
            try:
                for frame in range(int(cache["num_source_frames"])):
                    ok,image=cap.read()
                    if not ok:
                        raise AssertionError("Video decode truncated during face repair")
                    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    if max(rgb.shape[:2])>640:
                        ratio=640/max(rgb.shape[:2])
                        rgb=cv2.resize(rgb,(round(width*ratio),round(height*ratio)))
                    crop,record=head_crop(rgb,cache["xy"][frame,520:522],cache["confidence"][frame,520:522])
                    result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=crop))
                    if result.face_landmarks:
                        x1,y1,x2,y2=record
                        h,w=rgb.shape[:2]
                        cache["xy"][frame,42:520]=[[(x1+p.x*(x2-x1))/w,(y1+p.y*(y2-y1))/h] for p in result.face_landmarks[0]]
                        cache["confidence"][frame,42:520]=1.
            finally:
                cap.release()
            cache["boxes"],cache["box_confidence"]=boxes_from_landmarks(cache["xy"],cache["confidence"],cache["joint_type"])
            cache["motion"]=motion_from_landmarks(cache["xy"],cache["confidence"],cache["joint_type"])
            cache["global_motion"]=cache["motion"].mean(-1)
            cache["valid_frame"]=(cache["confidence"]>0).any(-1)
            metadata.update(version=CACHE_VERSION,upgraded_from=sha256(source),
                            face_search_region="square crop above confident shoulders; coordinates mapped to original frame")
            cache["version"]=np.array(CACHE_VERSION)
            cache["metadata"]=np.array(json.dumps(metadata))
            dest.parent.mkdir(parents=True,exist_ok=True)
            temp=dest.with_suffix(".tmp.npz")
            np.savez_compressed(temp,**cache)
            temp.replace(dest)
            row["landmark_path"]=str(dest.resolve())
            reports.append({"sample_id":row["sample_id"],"valid_fraction":(cache["box_confidence"]>=0.5).mean(0).tolist()})
            print(json.dumps({"completed":i+1,"total":len(rows),**reports[-1]}),flush=True)
    finally:
        detector.close()
    write_manifest(rows,args.out_manifest)
    Path(args.out_manifest+".repair_report.json").write_text(json.dumps(reports,indent=2))


if __name__=="__main__":
    main()
