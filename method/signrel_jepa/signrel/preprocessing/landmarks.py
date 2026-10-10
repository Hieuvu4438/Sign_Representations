"""MediaPipe Tasks CPU cache; confidence semantics are explicit, no raw heatmap scores."""
import json
from pathlib import Path
import cv2
import numpy as np
from .articulator_boxes import boxes_from_landmarks, motion_from_landmarks
from signrel.data.manifest import sha256

BODY_IDS = [11,12,13,14,15,16,17,18,19,20,23,24]
JOINT_TYPE = np.array([0]*21+[1]*21+[2]*478+[3]*len(BODY_IDS),np.int16)
CACHE_VERSION = "mediapipe_tasks_v2_shoulder_head_crop"


def head_crop(rgb, shoulders, confidence):
    """Non-destructive, square head search region from confident anatomical shoulders."""
    h,w = rgb.shape[:2]
    if (np.asarray(confidence)>=0.5).all():
        points = np.asarray(shoulders)*[w,h]
        shoulder_width = float(np.linalg.norm(points[0]-points[1]))
        if shoulder_width>=20:
            center = points.mean(0)
            center[1] -= 0.6*shoulder_width
            side = 1.35*shoulder_width
            x1,y1 = np.maximum(np.floor(center-side/2).astype(int),0)
            x2,y2 = np.minimum(np.ceil(center+side/2).astype(int),[w,h])
            if x2>x1 and y2>y1:
                return np.ascontiguousarray(rgb[y1:y2,x1:x2]),(x1,y1,x2,y2)
    return rgb,(0,0,w,h)


class LandmarkExtractor:
    def __init__(self, assets):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        self.mp = mp
        base = lambda name: mp.tasks.BaseOptions(model_asset_path=str(Path(assets)/name),
                                                 delegate=mp.tasks.BaseOptions.Delegate.CPU)
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=base("pose_landmarker_lite.task"),num_poses=1))
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base("hand_landmarker.task"),num_hands=2))
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=base("face_landmarker.task"),num_faces=1))
        self.model_hashes = {n:sha256(Path(assets)/n) for n in
                             ("pose_landmarker_lite.task","hand_landmarker.task","face_landmarker.task")}

    def close(self):
        self.pose.close()
        self.hands.close()
        self.face.close()

    def frame(self, rgb):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb))
        pose,hand = self.pose.detect(image),self.hands.detect(image)
        xy = np.zeros((len(JOINT_TYPE),2),np.float32)
        confidence = np.zeros(len(JOINT_TYPE),np.float32)
        pose_points = pose.pose_landmarks[0] if pose.pose_landmarks else None
        if pose_points:
            for j,pid in enumerate(BODY_IDS):
                p = pose_points[pid]
                xy[520+j] = [p.x,p.y]
                confidence[520+j] = min(p.visibility,p.presence)
        used = set()
        for points,handed in zip(hand.hand_landmarks,hand.handedness):
            wrist = np.array([points[0].x,points[0].y])
            slot = None
            if pose_points and min(pose_points[15].visibility,pose_points[16].visibility)>=0.5:
                distances = [np.linalg.norm(wrist-[pose_points[i].x,pose_points[i].y]) for i in (15,16)]
                best = int(np.argmin(distances))
                if distances[best]<0.15 and abs(distances[0]-distances[1])>0.02:
                    slot = best
            # Ambiguous anatomical assignment is invalid, not guessed from image x.
            if slot is None or slot in used:
                continue
            used.add(slot)
            begin = slot*21
            xy[begin:begin+21] = [[p.x,p.y] for p in points]
            # Tasks returns landmarks only after configured presence gate. No per-joint confidence.
            confidence[begin:begin+21] = 1.
        crop,record = head_crop(rgb,xy[520:522],confidence[520:522])
        face_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=crop)
        face = self.face.detect(face_image)
        if face.face_landmarks:
            points = face.face_landmarks[0]
            if len(points)!=478:
                raise AssertionError("Unexpected MediaPipe face joint count")
            x1,y1,x2,y2 = record
            height,width = rgb.shape[:2]
            xy[42:520] = [[(x1+p.x*(x2-x1))/width,(y1+p.y*(y2-y1))/height] for p in points]
            confidence[42:520] = 1.
        return xy,confidence

    def extract(self, path, out):
        path,out = Path(path),Path(out)
        cap = cv2.VideoCapture(str(path))
        xs,cs = [],[]
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            expected_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width,height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps<=0 or expected_count<=0:
                raise ValueError(f"Invalid video: {path}")
            while True:
                ok,frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                # Reduce detector compute; normalized coordinates remain in original full-frame space.
                if max(rgb.shape[:2])>640:
                    ratio = 640/max(rgb.shape[:2])
                    rgb = cv2.resize(rgb,(round(width*ratio),round(height*ratio)))
                x,c = self.frame(rgb)
                xs.append(x)
                cs.append(c)
        finally:
            cap.release()
        if len(xs)!=expected_count:
            raise AssertionError(f"Incomplete decode {len(xs)}/{expected_count}: {path}")
        xy,conf = np.stack(xs),np.stack(cs)
        boxes,box_conf = boxes_from_landmarks(xy,conf,JOINT_TYPE)
        motion = motion_from_landmarks(xy,conf,JOINT_TYPE)
        metadata = {"version":CACHE_VERSION, "mediapipe":self.mp.__version__,
                    "models":self.model_hashes,"source_sha256":sha256(path),"fps":fps,
                    "hand_confidence":"binary gated detection; anatomical slot requires pose wrist match",
                    "face_confidence":"binary gated detection", "body_confidence":"min(visibility,presence)",
                    "coordinates":"original full frame, normalized", "missing_assignment":"invalid"}
        out.parent.mkdir(parents=True,exist_ok=True)
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp,xy=xy,confidence=conf,joint_type=JOINT_TYPE,
                            valid_frame=(conf>0).any(-1),source_size=np.array([width,height],np.int32),
                            version=np.array(metadata["version"]),boxes=boxes,box_confidence=box_conf,
                            motion=motion,global_motion=motion.mean(-1),frame_indices=np.arange(len(xs)),
                            num_source_frames=np.array(len(xs)),metadata=np.array(json.dumps(metadata)))
        tmp.replace(out)
        return {"frames":len(xs),"valid_fraction_per_articulator":(box_conf>=0.5).mean(0).tolist(),**metadata}
