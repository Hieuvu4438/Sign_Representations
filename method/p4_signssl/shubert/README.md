# SHuBERT batch feature extractor (ASL Citizen train/val)

Frozen SHuBERT (`checkpoint_836_400000.pt`) per-frame features for dictionary-retrieval readout
comparisons. Test split is refused by the CLI (`--split {train,val}` only).

```
PYTHONDONTWRITEBYTECODE=1 nohup .venv-shubert-native/bin/python \
  method/p4_signssl/shubert/extract.py --split val --workers 8 --yolo-procs 4 --cores 0-7 \
  > method/p4_signssl/runtime/logs/shubert_val.log 2>&1 &
```
Re-running the same command resumes (finished shards are skipped; an interrupted finalize continues).

## Pipeline (what the author code does, and what we run)
Two author code paths exist: the GitHub quickstart (`third_party/SHuBERT`, commit cc19293) and the HF
demo (`third_party/shubert-author-demo`). Both use the same models and per-frame logic; we follow the
GitHub path, which the legacy runner (`artifacts/legacy/scripts/prepare_shubert_*.py`) also used:

1. `clips_bbox.crop_clip`: YOLOv8n person boxes on every frame -> one mean signer box -> crop, written
   with OpenCV mp4v. The YOLO calls run per frame on the GPU in helper processes, and the boxes are
   replayed inside the unmodified author `crop_clip`.
2. `kpe_mediapipe.detect_holistic` on each decoded (RGB) frame: pose landmarks, FaceLandmarker
   (6 faces), HandLandmarker (6 hands, conf 0.05). The author only reads `pose_landmarks` from
   Holistic; we use `mp.solutions.pose.Pose`, which gives bit-identical pose landmarks.
3. `crop_face` / `crop_hands` loops: face = eyes+mouth on grey background; hands = boxes around
   MediaPipe hands whose wrist is within 0.1 of a pose wrist. **Missing landmarks keep the author
   policy**: carry the previous crop forward, black 224x224 crop if none yet; body keypoints carry the
   previous pose, `-9999` if none yet. Crops are written as mp4v and re-read with decord at 224.
4. `body_features.normalize_pose_keypoints` -> 14-d body vector (7 joints x xy, signing-space normalised).
5. Finetuned DINOv2 ViT-S/14-reg (face / hand checkpoints) -> 384-d per stream; SHuBERT on the whole
   clip (batch 1, no padding). Per author `shubert_inference.py`, layer *i* output = `layer_results[i][-1]`
   (FFN branch of block *i*, before the residual add).

Frame rate: all native frames (~30 fps). The author inference code uses stride 1 (`features.py`) and
the GitHub scripts keep input fps; the paper's every-other-frame drop (pretraining) is not applied.
`--frame-stride 2` exists but is off by default.

Implementation: `extract.py` is the main GPU process. It initialises CUDA before any decord import
and runs DINO + SHuBERT, using about 2 GB of GPU memory. It drives two process pools, 12 processes
in total:
- `--yolo-procs` (4) GPU YOLO helpers, about 0.8 GB GPU each, which compute person boxes ahead of
  the workers.
- `--workers` (8) CPU workers (`preprocess.py`), each pinned to one core. They call the author
  functions, loaded read-only via importlib, and keep intermediates in `/dev/shm`.

Pinning matters: MediaPipe and TFLite thread pools oversubscribe the machine, and 12 unpinned workers
made almost no progress. Results go to shards of 1000 clips, written atomically
(`shards/shard_XXXX.npz`, tmp + fsync + rename). A streaming finalize then copies each shard into the
final arrays and deletes it, and can resume through `finalize_state.json`. Transient CUDA OOM on the
shared GPU is retried, not counted as a clip failure.

## Outputs (`runtime/feats/shubert_{split}/`)
`frames_l12.npy` [N,768] fp16, `offsets.npy` [C+1] int64, `layer_means.npy` [C,12,768] fp16 (time
mean of each layer in fp32, then fp16; zeros for failed clips), `frame_valid.npy` [N] bool,
`stream_observed.npy` [N,4] bool (face, left hand, right hand, pose detected on that frame),
`clips.jsonl` (split-file order; failed clips keep a row with `status != "ok"` and 0 frames), `meta.json`.

`frame_valid` = pose AND face detected AND at least one hand matched to a pose wrist on that frame.
Invalid frames still have features (they were computed from carried-forward or blank crops, as in
the author pipeline). This mask is ours; the author pipeline has no such mask.

## Deviations from the author pipeline (also in meta.json)
- Each intermediate video is decoded once in order, not via the author's per-frame `video[i]; seek(0)`
  (same pixels). Per-frame loops mirror the author scripts and call the author functions.
- **MediaPipe Holistic is reset per clip.** Holistic tracks across frames. The author batch script
  reuses one graph across clips, so its pose landmarks depend on which clip came before. Reset gives
  order-independent results, the same as the HF demo, which builds a fresh detector for each video.
- YOLO runs on the GPU in helper processes, which is what the author slurm scripts do; the legacy run
  used the CPU. Boxes are replayed into `crop_clip`. Signer crops from CPU YOLO and per-frame GPU YOLO
  were bit-identical on 30 val clips. Batched GPU YOLO changed 1 of 41 crops, so we keep per-frame
  calls. The model is loaded once per process.
- MediaPipe `Pose` replaces `Holistic`, since only pose landmarks are read. On 30 val clips (2687
  frames) the pose landmarks were bit-identical, including visibility and missing-pose frames, and the
  pose step is more than 2x cheaper. After the YOLO and Pose swaps, all 12 layers on the 4 fidelity
  clips are bit-identical to the CPU-YOLO + Holistic run (max abs 0.0).
- Body features are computed in memory instead of through the JSON file (exact round trip).
- SHuBERT is built from the checkpoint cfg with `strict=True`. The author uses the default cfg with
  `strict=False`; neither has missing or unexpected keys.
- DINO normalisation runs on the GPU with batch size 256. The author uses torchvision ToTensor and
  Normalize with batch 512; the float ops are the same.
- Not followed from the HF demo: it skips the YOLO crop, and its in-memory crops reach DINO as BGR
  (`cv2.COLOR_BGR2RGB` is applied to frames decord already returns as RGB). The GitHub path
  round-trips the crops through `cv2.VideoWriter`, so DINO gets RGB.

## Fidelity check against the legacy runner (`artifacts/legacy/features/shubert_native_common_v1`)
- Encoder half (legacy intermediates -> our DINO+SHuBERT, 5 val clips): layer-12 per-frame cosine
  >= 0.99996, max abs diff <= 0.18 (feature scale ~10). The differences come from batch-size float order.
- End-to-end on the 3 legacy clips that began with a fresh Holistic graph (train clips that started
  each legacy attempt): layer-12 per-frame cosine min 0.9995, mean >= 0.99999. Clip layer-mean cosine
  is >= 0.999996, and the stream-observation masks are identical.
- Other legacy clips (including the 5 val smoke clips) differ: layer-12 per-frame cosine mean
  0.14 to 0.99, and as low as -0.29 on single frames. This is because their legacy Holistic state was
  carried over from the previous clip. Replaying the legacy clip order without reset reproduces the
  legacy pose landmarks exactly (max abs 0.0). Signer crops, face landmarks and hand landmarks already
  match exactly. So SHuBERT features are very sensitive to the body-pose stream, and legacy features
  depend on clip order.
