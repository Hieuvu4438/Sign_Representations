"""Batch SHuBERT feature extraction for ASL Citizen (train / val only; test is refused).

GPU side (this process): finetuned DINOv2 face/hand encoders + SHuBERT (checkpoint_836_400000).
CPU side (<= --workers processes, preprocess.py): author raw-video preprocessing per clip.

Output (per split): frames_l12.npy [N,768] fp16, offsets.npy [C+1] int64, layer_means.npy [C,12,768]
fp16, frame_valid.npy [N] bool, stream_observed.npy [N,4] bool, clips.jsonl, meta.json.
Sharded + resumable: shards/shard_XXXX.npz (atomic tmp+rename) are skipped on restart; finalize
streams shards into the final arrays and deletes each shard after it is durably copied.
See README.md for the design note.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REPO = Path('/home/haipd/Sign_Representations')
AUTHOR = REPO / 'third_party/SHuBERT'
DINO_REPO = REPO / 'third_party/dinov2'
MODELS = REPO / 'checkpoints/shubert_author_hf/models'
CKPT = MODELS / 'checkpoint_836_400000.pt'
DINO_FACE = MODELS / 'dinov2face.pth'
DINO_HAND = MODELS / 'dinov2hand.pth'
ASLC = Path('/home/dongvk/datasets/ASL_Citizen')
OUT_ROOT = REPO / 'method/p4_signssl/runtime/feats'
STREAM_NAMES = ['face', 'left_hand', 'right_hand', 'pose']


def log(msg):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), msg, flush=True)


def sha256(path, cache={}):
    path = str(path)
    if path not in cache:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1 << 22), b''):
                h.update(block)
        cache[path] = h.hexdigest()
    return cache[path]


def split_rows(split):
    rows = []
    with open(ASLC / 'splits' / f'{split}.csv', newline='') as f:
        for r in csv.DictReader(f):
            rows.append({'sample_id': Path(r['Video file']).stem, 'path': str(ASLC / 'videos' / r['Video file']),
                         'gloss': r['Gloss'], 'signer': r['Participant ID'], 'asllex': r['ASL-LEX Code'],
                         'split': split})
    return rows


# ----------------------------------------------------------------------------------------- models
def load_models(device):
    import dataclasses
    import torch
    sys.path.insert(0, str(AUTHOR / 'fairseq'))
    import fairseq  # noqa: F401  (import before examples so registration does not half-fail)
    from examples.shubert.models.shubert import SHubertConfig, SHubertModel
    payload = torch.load(CKPT, map_location='cpu', weights_only=False)
    saved = dict(payload['cfg']['model'])
    known = {f.name for f in dataclasses.fields(SHubertConfig)}
    model = SHubertModel(SHubertConfig(**{k: v for k, v in saved.items() if k in known}))
    res = model.load_state_dict(payload['model'], strict=True)
    del payload
    model.eval().requires_grad_(False).to(device)
    audit = {'shubert_strict_load': True, 'missing': list(res.missing_keys), 'unexpected': list(res.unexpected_keys),
             'ignored_cfg_fields': sorted(set(saved) - known)}
    dinos = {}
    for name, path in (('face', DINO_FACE), ('hand', DINO_HAND)):
        m = torch.hub.load(str(DINO_REPO), 'dinov2_vits14_reg', source='local', pretrained=False)
        m.pos_embed = torch.nn.Parameter(torch.zeros(1, 257, 384))
        sd = torch.load(path, map_location='cpu', weights_only=False)['teacher']
        sd = {k.replace('backbone.', ''): v for k, v in sd.items() if 'dino_head' not in k}
        m.load_state_dict(sd, strict=True)
        dinos[name] = m.eval().requires_grad_(False).to(device)
    return model, dinos, audit


class Encoder:
    def __init__(self, device, dino_batch, hidden='ffn'):
        self.hidden = hidden
        import torch
        self.torch, self.device, self.dino_batch = torch, device, dino_batch
        self.shubert, self.dinos, self.audit = load_models(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def dino(self, model, frames):
        """frames uint8 [n,224,224,3] RGB -> [n,384]. Same arithmetic as torchvision ToTensor +
        Normalize used by the author (float32 /255, -mean, /std)."""
        torch = self.torch
        outs = []
        for s in range(0, len(frames), self.dino_batch):
            x = torch.from_numpy(frames[s:s + self.dino_batch]).to(self.device)
            x = x.permute(0, 3, 1, 2).float().div(255).sub(self.mean).div(self.std)
            outs.append(model(x).float())
        return torch.cat(outs)

    def encode(self, item):
        """Returns (all-layer features [12,T,768] float32 torch, on device)."""
        torch = self.torch
        with torch.inference_mode():
            st = item['streams']
            face = self.dino(self.dinos['face'], st['face'])
            hands = self.dino(self.dinos['hand'], __import__('numpy').concatenate([st['left_hand'], st['right_hand']]))
            n = face.shape[0]
            src = {'face': face, 'left_hand': hands[:n], 'right_hand': hands[n:],
                   'body_posture': torch.from_numpy(item['body']).float().to(self.device)}
            for k in ['face', 'left_hand', 'right_hand', 'body_posture']:
                src['label_' + k] = torch.zeros((n, 1), device=self.device)
            res = self.shubert.extract_features([src], padding_mask=None, kmeans_labels=None, mask=False)
            # Author features/shubert_inference.py: layer[-1] of each layer_results tuple
            # (the FFN-branch output of each transformer block, before residual add).
            # hidden='ffn': author semantics (FFN branch, lr[-1]); hidden='residual': block output x (lr[0]),
            # i.e. the encoder's residual stream (added for p4 EX06b).
            pick = -1 if self.hidden == 'ffn' else 0
            layers = torch.stack([lr[pick].squeeze(1) if lr[pick].dim() == 3 and lr[pick].shape[1] == 1 else lr[pick].transpose(0, 1).squeeze(0) for lr in res['layer_results']])
        if layers.shape != (12, n, 768):
            raise RuntimeError(f'unexpected SHuBERT output {tuple(layers.shape)} for {n} frames')
        if not torch.isfinite(layers).all():
            raise RuntimeError('non-finite SHuBERT features')
        return layers


# ------------------------------------------------------------------------------------------ shards
def write_shard(path, frames, n_frames, layer_means, valid, observed, rows):
    import numpy as np
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        np.savez(f, frames_l12=frames, n_frames=np.asarray(n_frames, np.int64), layer_means=layer_means,
                 frame_valid=valid, stream_observed=observed, rows=np.array(json.dumps(rows)))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def finalize(out, n_clips, shard_size, meta):
    """Stream shards into final arrays; delete each shard once it is copied and the progress state
    is durable (keeps peak disk ~1x the split size). Resumable via finalize_state.json."""
    import numpy as np
    from numpy.lib.format import open_memmap
    sd = out / 'shards'
    n_shards = (n_clips + shard_size - 1) // shard_size
    state_path = out / 'finalize_state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        counts = []
        for k in range(n_shards):
            with np.load(sd / f'shard_{k:04d}.npz', allow_pickle=False) as z:
                counts.append(z['n_frames'].astype(np.int64))
        counts = np.concatenate(counts)
        if len(counts) != n_clips:
            raise RuntimeError(f'shards hold {len(counts)} clips, expected {n_clips}')
        offsets = np.zeros(n_clips + 1, np.int64)
        offsets[1:] = np.cumsum(counts)
        np.save(out / 'offsets.npy.tmp.npy', offsets)
        N = int(offsets[-1])
        open_memmap(out / 'frames_l12.npy.tmp', 'w+', np.float16, (N, 768)).flush()
        open_memmap(out / 'layer_means.npy.tmp', 'w+', np.float16, (n_clips, 12, 768)).flush()
        open_memmap(out / 'frame_valid.npy.tmp', 'w+', np.bool_, (N,)).flush()
        open_memmap(out / 'stream_observed.npy.tmp', 'w+', np.bool_, (N, 4)).flush()
        (out / 'clips.jsonl.tmp').write_text('')
        state = {'next_shard': 0, 'n_frames_total': N, 'clips_lines': 0}
        state_path.write_text(json.dumps(state))
    offsets = np.load(out / 'offsets.npy.tmp.npy')
    fr = open_memmap(out / 'frames_l12.npy.tmp', 'r+')
    lm = open_memmap(out / 'layer_means.npy.tmp', 'r+')
    fv = open_memmap(out / 'frame_valid.npy.tmp', 'r+')
    so = open_memmap(out / 'stream_observed.npy.tmp', 'r+')
    clips_tmp = out / 'clips.jsonl.tmp'
    # Drop any clips lines appended after the last durable state (crash between append and state write).
    lines = clips_tmp.read_text().splitlines(keepends=True)[:state['clips_lines']]
    clips_tmp.write_text(''.join(lines))
    for k in range(state['next_shard'], n_shards):
        p = sd / f'shard_{k:04d}.npz'
        c0, c1 = k * shard_size, min(n_clips, (k + 1) * shard_size)
        with np.load(p, allow_pickle=False) as z:
            a, b = int(offsets[c0]), int(offsets[c1])
            if z['frames_l12'].shape[0] != b - a or z['layer_means'].shape[0] != c1 - c0:
                raise RuntimeError(f'shard {k} size mismatch')
            fr[a:b] = z['frames_l12']
            fv[a:b] = z['frame_valid']
            so[a:b] = z['stream_observed']
            lm[c0:c1] = z['layer_means']
            rows = json.loads(str(z['rows']))
        for m in (fr, lm, fv, so):
            m.flush()
        with open(clips_tmp, 'a') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
            f.flush()
            os.fsync(f.fileno())
        state.update(next_shard=k + 1, clips_lines=state['clips_lines'] + len(rows))
        tmp = state_path.with_name('finalize_state.json.tmp')
        tmp.write_text(json.dumps(state))
        os.replace(tmp, state_path)
        p.unlink()
        log(f'finalize: copied+removed shard {k + 1}/{n_shards}')
    del fr, lm, fv, so
    for name in ['frames_l12.npy', 'layer_means.npy', 'frame_valid.npy', 'stream_observed.npy']:
        os.replace(out / (name + '.tmp'), out / name)
    os.replace(out / 'offsets.npy.tmp.npy', out / 'offsets.npy')
    os.replace(clips_tmp, out / 'clips.jsonl')
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    state_path.unlink()
    try:
        sd.rmdir()
    except OSError:
        pass


# -------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    ap.add_argument('--allow-test', action='store_true', help='test only under a LOCKED protocol (p4 EX10)')
    ap.add_argument('--out', default='', help='default: runtime/feats/shubert_{split}')
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument('--shard-size', type=int, default=1000)
    ap.add_argument('--sample-ids', default='', help='file of sample_ids (smoke subset, split order kept)')
    ap.add_argument('--max-shards', type=int, default=0, help='stop after this many new shards (throughput test)')
    ap.add_argument('--yolo-device', default='replay', choices=['replay', 'cpu', 'cuda'],
                    help='replay: YOLO runs batched on the GPU in this process, boxes are replayed inside the '
                         "author crop_clip in the workers; cpu/cuda: YOLO inside each worker (cuda costs ~0.8 GB "
                         'GPU per worker)')
    ap.add_argument('--yolo-procs', type=int, default=2, help='replay mode: GPU YOLO processes (count toward the 12)')
    ap.add_argument('--pose-model', default='pose', choices=['pose', 'holistic'],
                    help='pose: MediaPipe Pose (pose landmarks bit-identical to Holistic, verified)')
    ap.add_argument('--cores', default='0-11', help="CPU cores to pin workers to, one per worker ('' = no pinning)")
    ap.add_argument('--frame-stride', type=int, default=1, help='1 = author inference code (all frames)')
    ap.add_argument('--dino-batch', type=int, default=256)
    ap.add_argument('--gpu-mem-gb', type=float, default=10.0)
    ap.add_argument('--hidden', default='ffn', choices=['ffn', 'residual'])
    ap.add_argument('--save-all-layers', action='store_true', help='smoke only: keep per-frame 12-layer fp32 in shard')
    a = ap.parse_args()
    if a.split == 'test' and not a.allow_test:
        sys.exit('test split is quarantined; pass --allow-test only under a LOCKED protocol')
    if a.workers + (a.yolo_procs if a.yolo_device == 'replay' else 0) > 12:
        sys.exit('resource rule: <= 12 worker processes (CPU workers + YOLO processes)')
    out = Path(a.out) if a.out else OUT_ROOT / f'shubert_{a.split}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'meta.json').exists():
        log(f'{out} already complete')
        return
    rows = split_rows(a.split)
    if a.sample_ids:
        keep = {l.strip() for l in open(a.sample_ids) if l.strip()}
        rows = [r for r in rows if r['sample_id'] in keep]
    n_clips = len(rows)
    n_shards = (n_clips + a.shard_size - 1) // a.shard_size
    sd = out / 'shards'
    sd.mkdir(exist_ok=True)
    config = {k: getattr(a, k) for k in ['split', 'shard_size', 'frame_stride', 'sample_ids']}
    cfg_path = out / 'run_config.json'
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != config:
        sys.exit(f'run config changed vs existing shards: {cfg_path.read_text()} != {config}')
    cfg_path.write_text(json.dumps(config))
    finalizing = (out / 'finalize_state.json').exists()
    todo = [k for k in range(n_shards) if not finalizing and not (sd / f'shard_{k:04d}.npz').exists()]
    if a.max_shards:
        todo = todo[:a.max_shards]
    log(f'split={a.split} clips={n_clips} shards={n_shards} todo={len(todo)} out={out}')

    import numpy as np
    run_seconds = 0.0
    stats_path = out / 'run_stats.json'
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {'extract_seconds': 0.0, 'runs': []}
    if todo:
        import torch
        torch.cuda.init()  # CUDA before anything that could import decord (segfault pitfall)
        device = torch.device('cuda:0')
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, a.gpu_mem_gb * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory))
        enc = Encoder(device, a.dino_batch, a.hidden)
        log(f'models loaded; gpu_alloc={torch.cuda.memory_allocated() / 2**30:.2f}GB audit={enc.audit}')
        import multiprocessing as mp
        import preprocess
        ctx = mp.get_context('spawn')
        jobs = [(i, rows[i]['path']) for k in todo for i in range(k * a.shard_size, min(n_clips, (k + 1) * a.shard_size))]
        t0 = time.time()
        done = fails = frames_done = 0
        buf = None
        cores = []
        for part in filter(None, a.cores.split(',')):
            lo, _, hi = part.partition('-')
            cores += list(range(int(lo), int(hi or lo) + 1))
        if cores and len(cores) < a.workers:
            sys.exit('need at least one core per worker')
        counter = ctx.Value('i', 0)
        yolo_procs = a.yolo_procs if a.yolo_device == 'replay' else 0
        with ctx.Pool(max(1, yolo_procs), initializer=preprocess.init_yolo_worker if yolo_procs else None) as ypool, \
                ctx.Pool(a.workers, initializer=preprocess.init_worker,
                         initargs=(a.yolo_device, a.frame_stride, a.pose_model, cores, counter)) as pool:
            # replay: GPU YOLO processes compute person boxes ahead of the CPU workers (ordered).
            feed = (ypool.imap(preprocess.yolo_job, jobs, chunksize=1) if yolo_procs
                    else ((i, p, None, None) for i, p in jobs))
            for item in pool.imap(preprocess.process_clip, feed, chunksize=1):
                i = item['index']
                k = i // a.shard_size
                if buf is None or buf['k'] != k:
                    buf = {'k': k, 'frames': [], 'n': [], 'means': [], 'valid': [], 'obs': [], 'rows': [], 'all': []}
                row = {key: rows[i][key] for key in ['sample_id', 'gloss', 'signer', 'asllex', 'split']}
                fps = item.get('fps')
                row['fps'] = (fps / a.frame_stride) if fps else None
                n = 0
                if item['status'] == 'ok':
                    try:
                        for attempt in range(30):  # shared GPU: wait out transient OOM instead of failing
                            try:
                                layers = enc.encode(item)
                                break
                            except torch.cuda.OutOfMemoryError:
                                torch.cuda.empty_cache()
                                if attempt == 29:
                                    raise
                                log(f'OOM on {rows[i]["sample_id"]} (attempt {attempt + 1}); retrying in 20s')
                                time.sleep(20)
                        n = layers.shape[1]
                        buf['frames'].append(layers[-1].half().cpu().numpy())
                        buf['means'].append(layers.mean(1).half().cpu().numpy())
                        obs = item['observed']
                        buf['obs'].append(obs)
                        # frame_valid: pose detected AND face detected AND >=1 hand matched to a pose wrist
                        # on this frame (i.e. not carried forward / blank per the author crop logic).
                        buf['valid'].append(obs[:, 3] & obs[:, 0] & (obs[:, 1] | obs[:, 2]))
                        if a.save_all_layers:
                            buf['all'].append(layers.cpu().numpy())
                        row['status'] = 'ok'
                    except Exception as error:  # GPU-side failure: keep the row, zero frames
                        row['status'] = 'fail_encode'
                        row['error'] = f'{type(error).__name__}: {error}'
                        n = 0
                else:
                    row['status'] = 'fail_preprocess'
                    row['error'] = item['error']
                if n == 0:
                    buf['means'].append(np.zeros((12, 768), np.float16))
                    fails += 1
                    if fails <= 20 or fails % 100 == 0:
                        log(f'FAIL #{fails} {row["sample_id"]}: {row["status"]} {row.get("error")}')
                row['n_frames'] = n
                row['n_src_frames'] = item.get('n_src_frames')
                row['missing_landmark_frames'] = item.get('missing_result_frames')
                buf['n'].append(n)
                buf['rows'].append(row)
                done += 1
                frames_done += n
                c1 = min(n_clips, (k + 1) * a.shard_size)
                if i == c1 - 1:  # shard complete (imap is ordered)
                    cat = lambda xs, shape, dt: np.concatenate(xs) if xs else np.zeros(shape, dt)
                    path = sd / f'shard_{k:04d}.npz'
                    write_shard(path, cat(buf['frames'], (0, 768), np.float16), buf['n'], np.stack(buf['means']),
                                cat(buf['valid'], (0,), bool), cat(buf['obs'], (0, 4), bool), buf['rows'])
                    if a.save_all_layers:
                        np.save(sd / f'shard_{k:04d}_all_layers.npy',
                                np.concatenate(buf['all'], axis=1) if buf['all'] else np.zeros((12, 0, 768), np.float32))
                    el = time.time() - t0
                    log(f'wrote {path.name}: clips={len(buf["rows"])} frames={sum(buf["n"])}')
                    buf = None
                if done % 50 == 0 or done == len(jobs):
                    el = time.time() - t0
                    rate = done / el
                    log(f'progress {done}/{len(jobs)} clips  {rate:.3f} clips/s  {frames_done / el:.1f} frames/s  '
                        f'fails={fails}  eta={(len(jobs) - done) / max(rate, 1e-9) / 3600:.2f}h  '
                        f'gpu_peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GB  last_timing='
                        f'{ {kk: round(v, 2) for kk, v in item["timing"].items()} }')
        run_seconds = time.time() - t0
        stats['extract_seconds'] += run_seconds
        stats['runs'].append({'clips': done, 'seconds': run_seconds, 'clips_per_s': done / max(run_seconds, 1e-9),
                              'workers': a.workers, 'cores': a.cores, 'yolo_device': a.yolo_device,
                              'pose_model': a.pose_model, 'fails': fails, 'gpu_peak_gb': torch.cuda.max_memory_allocated() / 2**30})
        stats_path.write_text(json.dumps(stats, indent=1))
    missing = [k for k in range(n_shards) if not (sd / f'shard_{k:04d}.npz').exists()]
    if missing and not (out / 'finalize_state.json').exists():
        log(f'{len(missing)} shards still missing; not finalizing')
        return
    if a.save_all_layers:
        log('smoke mode (--save-all-layers): skipping finalize so shards stay inspectable')
        return
    meta = build_meta(a, n_clips, stats)
    finalize(out, n_clips, a.shard_size, meta)
    (out / 'run_stats.json').unlink(missing_ok=True)
    log(f'finalized {out}')


def build_meta(a, n_clips, stats):
    import torch
    import cv2
    srcs = {p: sha256(AUTHOR / p) for p in ['dataset/clips_bbox.py', 'dataset/kpe_mediapipe.py', 'dataset/crop_face.py',
                                            'dataset/crop_hands.py', 'features/body_features.py',
                                            'features/dinov2_features.py', 'features/shubert_inference.py']}
    return {
        'split': a.split, 'n_clips': n_clips, 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'extract_seconds_total': stats.get('extract_seconds'), 'runs': stats.get('runs'),
        'checkpoints_sha256': {p.name: sha256(p) for p in [CKPT, DINO_FACE, DINO_HAND, MODELS / 'yolov8n.pt',
                                                           MODELS / 'hand_landmarker.task',
                                                           MODELS / 'face_landmarker_v2_with_blendshapes.task']},
        'code': {'extract.py': sha256(HERE / 'extract.py'), 'preprocess.py': sha256(HERE / 'preprocess.py'),
                 'author_repo': 'third_party/SHuBERT (GitHub, commit cc19293, unmodified)',
                 'author_sources_sha256': srcs, 'dinov2_repo': 'third_party/dinov2 (commit 7764ea0)',
                 'torch': torch.__version__, 'opencv': cv2.__version__},
        'frame_rate': {'frame_stride': a.frame_stride,
                       'note': ('All native frames (author inference code: features.py stride=1; GitHub quickstart '
                                'keeps input fps). The paper drops every other frame (~15 fps) for pretraining; not '
                                'applied here.') if a.frame_stride == 1 else
                               f'Every {a.frame_stride}-th native frame (after signer crop, before landmarks).',
                       'fps_field': 'clips.jsonl fps = OpenCV CAP_PROP_FPS of the source video / frame_stride'},
        'features': {'frames_l12': 'layer_results[11][-1] (author shubert_inference.py semantics: FFN-branch '
                                   'output of the final transformer block before residual add), float16',
                     'layer_means': 'time-mean over all frames of layer_results[i][-1], i=0..11, computed in fp32 then '
                                    'cast to float16; zeros for failed clips',
                     'offsets': 'clip c frames = frames_l12[offsets[c]:offsets[c+1]] (equal offsets for failed clips)',
                     'frame_valid': 'pose detected AND face detected AND >=1 hand (MediaPipe hand whose wrist is '
                                    'within 0.1 of a pose wrist, author select_hands) on that frame; frames failing '
                                    'this still have features (author carry-forward / blank crops feed the model)',
                     'stream_observed': 'columns [face, left_hand(hand1), right_hand(hand2), pose] per frame'},
        'deviations': DEVIATIONS,
    }


DEVIATIONS = [
    'Preprocessing follows the GitHub SHuBERT quickstart (YOLOv8 signer crop -> MediaPipe -> face/hand crops '
    'written as mp4v and re-read with decord at 224 -> finetuned DINOv2 -> SHuBERT), as the legacy runner did. '
    'The HF demo (features.py) skips the YOLO crop and feeds in-memory crops that are BGR (it applies '
    'cv2.COLOR_BGR2RGB to decord RGB frames and never converts back); the GitHub path the DINO/SHuBERT models '
    'were trained on round-trips through cv2.VideoWriter, so DINO sees RGB. We follow the GitHub path.',
    'Intermediate videos are decoded once sequentially instead of the author per-frame `video[i]; video.seek(0)` '
    'access (same decoded pixels); author per-frame loops are mirrored in preprocess.py calling the author functions.',
    'MediaPipe Holistic (a tracking graph) is reset at the start of every clip, so results do not depend on clip '
    'order (the author batch script reuses one graph across clips; the HF demo builds a fresh one per video).',
    'YOLO (yolov8n, author settings, per-frame calls) runs on the GPU in 2 helper processes over the same '
    'cv2-decoded RGB frames; the per-frame person boxes are replayed inside the unmodified author crop_clip in the '
    'workers. Author slurm scripts run it on a GPU; the legacy run used CPU. Signer crops were bit-identical CPU vs '
    'per-frame GPU on 30 val clips (batched GPU inference changed 1 of 41 crops, so it is not used). Model is built '
    'once per process rather than once per clip.',
    'Pose landmarks come from mp.solutions.pose.Pose(min_detection_confidence=0.1) instead of Holistic (author only '
    'reads results.pose_landmarks). Verified bit-identical (x,y,z,visibility and missing-pose frames) on 30 val clips '
    '/ 2687 frames; >2x cheaper. Per-run choice recorded in meta.runs.',
    'Workers are pinned to one CPU core each (performance only).',
    'Body features computed in memory with the author normalize_pose_keypoints loop instead of via the JSON file '
    '(JSON round trip of Python floats is exact).',
    'SHuBERT built from the checkpoint cfg with strict=True load (author: default SHubertConfig, strict=False); '
    'no missing/unexpected keys.',
    'DINO input normalisation done on GPU in batches (same float32 ops as ToTensor+Normalize); batch size differs '
    'from author (512) -> only floating-point-order differences.',
    'Clips whose author crop loop drops frames (no landmarks result before the first detection) or other failures '
    'are recorded with status != ok and zero frames.',
]


if __name__ == '__main__':
    main()
