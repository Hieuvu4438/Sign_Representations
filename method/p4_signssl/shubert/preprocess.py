"""CPU worker: author (GitHub SHuBERT quickstart) raw-video preprocessing for one clip.

Replicates third_party/SHuBERT/dataset/{clips_bbox,kpe_mediapipe,crop_face,crop_hands}.py and
features/body_features.py by calling the author's own functions (loaded read-only via importlib)
inside per-frame loops that mirror the author scripts line by line. The only structural change is
that every intermediate video is decoded once sequentially instead of with the author's
`video[i]; video.seek(0)` random access (identical decoded pixels, far less decode work).
Intermediate mp4 files (signer crop, face crop, hand crops) are still written with OpenCV mp4v and
re-read with decord exactly as in the author pipeline, so codec round-trips are preserved; they live
in /dev/shm and are deleted after each clip.

Nothing in this module may initialise CUDA before decord is imported (decord-before-CUDA segfault):
the worker initializer hides GPUs unless --yolo-device cuda is requested, in which case CUDA is
initialised first.
"""
import contextlib
import importlib.util
import io
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path('/home/haipd/Sign_Representations')
AUTHOR = REPO / 'third_party/SHuBERT'
MODELS = REPO / 'checkpoints/shubert_author_hf/models'
FACE_TASK = MODELS / 'face_landmarker_v2_with_blendshapes.task'
HAND_TASK = MODELS / 'hand_landmarker.task'
YOLO_PT = MODELS / 'yolov8n.pt'

_G = {}


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, AUTHOR / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _CachedYOLO:
    """Stands in for `ultralytics.YOLO(path)` inside clips_bbox.crop_clip: the author constructs the
    model per clip; we construct it once per worker. verbose=False only silences per-frame logging."""

    def __init__(self, model, device):
        self.model, self.device = model, device

    def __call__(self, image):
        return self.model(image, verbose=False, device=self.device)


class _Box:
    def __init__(self, cls, xyxy):
        self.cls = cls
        self.xyxy = [np.array(xyxy, dtype=np.float64)]  # .tolist() returns the stored Python floats


class _Result:
    def __init__(self, dets):
        self.boxes = [_Box(c, b) for c, b in dets]


class ReplayYOLO:
    """Stands in for the YOLO model inside the author's crop_clip, returning person boxes precomputed
    on the GPU by the main process (yolo_boxes below) for the same frames, in the same order. crop_clip
    reads only `box.cls` and `box.xyxy[0].tolist()`, so the replay is exact."""

    def __init__(self, per_frame):
        self.per_frame, self.i = per_frame, 0

    def __call__(self, image):
        if self.i >= len(self.per_frame):
            raise RuntimeError('replay YOLO: more frames decoded than boxes precomputed')
        out = [_Result(self.per_frame[self.i])]
        self.i += 1
        return out


def yolo_boxes(model, cv2, video_path, device, batch=1):
    """Main-process GPU YOLO over the frames crop_clip will decode (cv2 sequential read, BGR->RGB as in
    the author). Returns per-frame [(cls, [x1,y1,x2,y2]), ...] exactly as the author code reads them.
    batch=1 = per-frame calls like the author; batched GPU inference changed one signer crop by float
    noise in 41 test clips, per-frame GPU matched CPU on all tested clips."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    out = []
    for s in range(0, len(frames), batch):
        for r in model(frames[s:s + batch], verbose=False, device=device):
            out.append([(float(b.cls), b.xyxy[0].tolist()) for b in r.boxes])
    return out


def init_yolo_worker():
    """GPU YOLO process (replay mode). CUDA is initialised first; this process never imports decord."""
    sys.dont_write_bytecode = True
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import torch
    torch.cuda.init()
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.03)
    import cv2
    cv2.setNumThreads(1)
    from ultralytics import YOLO
    _G.update(yolo=YOLO(str(YOLO_PT)), cv2=cv2, torch=torch)


def yolo_job(job):
    """job = (index, path) -> (index, path, boxes, error). Waits out transient OOM on the shared GPU."""
    index, path = job
    torch, err = _G['torch'], None
    for _ in range(30):
        try:
            return index, path, yolo_boxes(_G['yolo'], _G['cv2'], path, 'cuda:0'), None
        except torch.cuda.OutOfMemoryError as e:
            err = f'{type(e).__name__}: {e}'[:300]
            torch.cuda.empty_cache()
            time.sleep(20)
        except Exception as e:
            return index, path, None, f'{type(e).__name__}: {e}'[:300]
    return index, path, None, err


def init_worker(yolo_device='replay', frame_stride=1, pose_model='pose', cores=None, counter=None, return_landmarks=False):
    sys.dont_write_bytecode = True
    if cores:
        # One core per worker: MediaPipe/TFLite thread pools otherwise oversubscribe the machine
        # (12 unpinned workers ran ~10x slower than pinned ones).
        with counter.get_lock():
            core = cores[counter.value % len(cores)]
            counter.value += 1
        os.sched_setaffinity(0, {core})
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['GLOG_minloglevel'] = '2'
    if yolo_device != 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # workers never touch the GPU unless asked
    import torch
    torch.set_num_threads(1)
    if yolo_device == 'cuda':
        torch.cuda.init()  # must precede the decord import below
        torch.cuda.set_per_process_memory_fraction(0.02)
    import cv2
    cv2.setNumThreads(1)
    import decord  # noqa: F401  (after CUDA init when CUDA is used)
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    mods = {k: _load(v, 'author_' + k) for k, v in {
        'crop': 'dataset/clips_bbox.py', 'pose': 'dataset/kpe_mediapipe.py',
        'face': 'dataset/crop_face.py', 'hands': 'dataset/crop_hands.py',
        'body': 'features/body_features.py'}.items()}
    # Author kpe_mediapipe.__main__ settings (identical to the HF demo HolisticDetector defaults).
    face_opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_TASK)),
        output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=6)
    hand_opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_TASK)),
        num_hands=6, min_hand_detection_confidence=0.05)
    pose = mods['pose']
    pose.face_detector = vision.FaceLandmarker.create_from_options(face_opts)
    pose.hand_detector = vision.HandLandmarker.create_from_options(hand_opts)
    # detect_holistic only reads `results.pose_landmarks` from mp_holistic. The Pose solution's pose
    # landmarks are bit-identical to Holistic's (verified on 30 val clips / 2687 frames, max abs 0.0,
    # incl. visibility and missing-pose frames) at less than half the CPU cost.
    if pose_model == 'holistic':
        pose.mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.1)
    else:
        pose.mp_holistic = mp.solutions.pose.Pose(min_detection_confidence=0.1)
    if yolo_device in ('cpu', 'cuda'):
        from ultralytics import YOLO
        cached = _CachedYOLO(YOLO(str(YOLO_PT)), 'cuda:0' if yolo_device == 'cuda' else 'cpu')
        mods['crop'].YOLO = lambda _path: cached
    _G.update(mods=mods, cv2=cv2, decord=decord, yolo_device=yolo_device, frame_stride=frame_stride, return_landmarks=return_landmarks,
              tmp_root=Path('/dev/shm') / f'shubert_extract_{os.getpid()}')
    _G['tmp_root'].mkdir(parents=True, exist_ok=True)


def _write_mp4(cv2, path, frames, fps):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (224, 224))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _read_224(decord, path):
    # Author features/dinov2_features.py: VideoReader(path, width=224, height=224).get_batch(...)
    vr = decord.VideoReader(str(path), width=224, height=224)
    return vr.get_batch(list(range(len(vr)))).asnumpy()


def process_clip(job):
    """job = (index, video_path, boxes_or_None, error_or_None). Returns dict with status and, if ok,
    uint8 crops + body array."""
    index, video_path, boxes, upstream_error = job
    t0 = time.perf_counter()
    timing = {}
    mods, cv2, decord = _G['mods'], _G['cv2'], _G['decord']
    work = _G['tmp_root'] / f'clip_{index}'
    out = {'index': index, 'status': 'ok', 'error': None}
    sink = io.StringIO()
    try:
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        out['fps'] = float(fps) if fps and np.isfinite(fps) else None
        if not Path(video_path).is_file():
            raise FileNotFoundError(video_path)
        if upstream_error:
            raise RuntimeError('main-process YOLO: ' + upstream_error)
        if _G['yolo_device'] == 'replay':
            if boxes is None:
                raise RuntimeError('replay mode needs precomputed YOLO boxes')
            replay = ReplayYOLO(boxes)
            mods['crop'].YOLO = lambda _path: replay
        # Step 1: signer crop (author crop_clip, YOLO person boxes, mp4v re-encode).
        problem = work / 'problems.txt'
        problem.touch()
        signer = work / 'signer.mp4'
        with contextlib.redirect_stdout(sink):
            mods['crop'].crop_clip(str(video_path), str(problem), str(signer), str(YOLO_PT))
        timing['yolo_crop'] = time.perf_counter() - t0
        if _G['yolo_device'] == 'replay' and replay.i != len(boxes):
            raise RuntimeError(f'replay YOLO consumed {replay.i} of {len(boxes)} frames')
        if problem.read_text().strip() or not signer.exists():
            raise RuntimeError('author crop_clip failed (problem file written / no signer.mp4)')
        vr = decord.VideoReader(str(signer))
        n_src = len(vr)
        frames = vr.get_batch(list(range(n_src))).asnumpy()  # RGB, as author `video[i].asnumpy()`
        del vr
        stride = _G['frame_stride']
        if stride > 1:
            frames = frames[::stride]
        n = len(frames)
        out['n_src_frames'] = n_src
        if n < 1:
            raise RuntimeError('no decodable frames')
        # Step 2: landmarks (author kpe_mediapipe.video_holistic loop). The pose graph tracks across
        # frames; reset per clip so results do not depend on clip order (HF demo builds a fresh one).
        t1 = time.perf_counter()
        mods['pose'].mp_holistic.reset()
        result = {}
        with contextlib.redirect_stdout(sink):
            for i in range(n):
                try:
                    _, result[str(i)] = mods['pose'].detect_holistic(frames[i])
                except Exception:
                    result[str(i)] = None
        timing['mediapipe'] = time.perf_counter() - t1
        # Step 3a: face crops (author crop_face.video_holistic loop).
        t2 = time.perf_counter()
        F = mods['face']
        face_frames, prev_face, prev_res = [], None, None
        res = dict(result)
        for i in range(n):
            frame_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            if res[str(i)] is None:
                if prev_res is not None:
                    res[str(i)] = prev_res
                else:
                    continue
            else:
                prev_res = res[str(i)]
            if res[str(i)]['pose_landmarks'] is None:
                face_frames.append(prev_face if prev_face is not None else np.zeros((224, 224, 3), np.uint8))
                continue
            if res[str(i)]['face_landmarks'] is not None:
                lm = F.select_face(res[str(i)]['pose_landmarks'][0], res[str(i)]['face_landmarks'])
                ff = F.resize_frame(F.cues_on_grey_background(frame_rgb, lm), (224, 224))
                face_frames.append(ff)
                prev_face = ff
            elif prev_face is not None:
                face_frames.append(prev_face)
            else:
                face_frames.append(np.zeros((224, 224, 3), np.uint8))
        # Step 3b: hand crops (author crop_hands.video_holistic loop).
        H = mods['hands']
        h1_frames, h2_frames, prev1, prev2, prev_res = [], [], None, None, None
        res = dict(result)
        observed = np.zeros((n, 4), bool)  # face, left_hand(hand1), right_hand(hand2), pose
        hand_lm = np.full((n, 2, 21, 3), np.nan, np.float32)
        pose_lm = np.full((n, 33, 3), np.nan, np.float32)
        for i in range(n):
            frame_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            raw = result[str(i)]
            if res[str(i)] is None:
                if prev_res is not None:
                    res[str(i)] = prev_res
                else:
                    continue
            else:
                prev_res = res[str(i)]
            r = res[str(i)]
            if r['pose_landmarks'] is None:
                h1_frames.append(prev1 if prev1 is not None else np.zeros((224, 224, 3), np.uint8))
                h2_frames.append(prev2 if prev2 is not None else np.zeros((224, 224, 3), np.uint8))
                continue
            left, right = H.select_hands(r['pose_landmarks'][0], r['hand_landmarks'], frame_rgb.shape)
            if raw is not None:  # observation mask: only genuinely detected (not carried-forward) inputs
                observed[i] = [bool(raw.get('face_landmarks')), left is not None, right is not None, True]
                if _G.get('return_landmarks'):  # p4 CIP teacher (EX09): raw per-frame MediaPipe landmarks
                    if left is not None:
                        hand_lm[i, 0] = np.asarray(left, np.float32)[:, :3]
                    if right is not None:
                        hand_lm[i, 1] = np.asarray(right, np.float32)[:, :3]
                    pose_lm[i] = np.asarray(r['pose_landmarks'][0], np.float32)[:33, :3]
            for lm, frames_out, which in ((left, h1_frames, 1), (right, h2_frames, 2)):
                prev = prev1 if which == 1 else prev2
                if lm is not None:
                    box = H.adjust_bounding_box(H.get_bounding_box(lm, frame_rgb.shape, scale_factor=1.5), frame_rgb.shape)
                    hf = H.resize_frame(H.crop_frame(frame_rgb, box), (224, 224))
                    frames_out.append(hf)
                    if which == 1:
                        prev1 = hf
                    else:
                        prev2 = hf
                elif prev is not None:
                    frames_out.append(prev)
                else:
                    frames_out.append(np.zeros((224, 224, 3), np.uint8))
        del frames
        timing['crops'] = time.perf_counter() - t2
        # Write/re-read through mp4v exactly like the author cue videos -> decord (RGB) for DINO.
        t3 = time.perf_counter()
        streams = {}
        enc_fps = fps if (fps and fps > 0) else 30.0
        for name, fr in (('face', face_frames), ('left_hand', h1_frames), ('right_hand', h2_frames)):
            if len(fr) != n:
                raise RuntimeError(f'author crop loop produced {len(fr)} {name} frames for {n} frames '
                                   '(no landmarks result before first detection)')
            path = work / f'{name}.mp4'
            _write_mp4(cv2, path, fr, enc_fps)
            arr = _read_224(decord, path)
            if arr.shape != (n, 224, 224, 3):
                raise RuntimeError(f'{name} mp4 round trip returned {arr.shape}')
            streams[name] = arr
        timing['codec'] = time.perf_counter() - t3
        # Step 4: body features (author features/body_features.keypoints_to_numpy loop, in memory;
        # JSON round trip of Python floats is exact so it is skipped).
        B = mods['body']
        prev_pose, body = None, []
        for i in range(n):
            r = result[str(i)]
            if r is None:
                fp = prev_pose if prev_pose is not None else np.full((7, 2), -9999)
            elif r['pose_landmarks'] is not None:
                p = B.normalize_pose_keypoints(r['pose_landmarks'][0][0:25])
                fp = np.array([p[j] for j in [0, 11, 12, 13, 14, 15, 16]]).flatten()
                prev_pose = fp
            elif prev_pose is not None:
                fp = prev_pose
            else:
                fp = np.full((7, 2), -9999).flatten()
            body.append(fp)
        body = np.array(body, dtype=np.float32)
        if body.shape != (n, 14) or not np.isfinite(body).all():
            raise RuntimeError(f'body feature shape/finite check failed {body.shape}')
        if _G.get('return_landmarks'):
            out.update(hand_lm=hand_lm, pose_lm=pose_lm)
        out.update(n_frames=n, streams=streams, body=body, observed=observed,
                   missing_result_frames=int(sum(v is None for v in result.values())))
    except Exception as error:
        out.update(status='fail', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc(limit=3),
                   n_frames=0)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    timing['total'] = time.perf_counter() - t0
    out['timing'] = timing
    return out
