"""EX09: proposal5-style teacher signals for How2Sign clips, reusing the SHuBERT author
preprocessing: per frame DINOv2 (author hand/face fine-tuned ViT-S/14-reg) appearance of
face / left hand / right hand crops (384-d each) + raw MediaPipe hand landmarks (2x21x3, with
the detector's relative z) + MediaPipe pose landmarks (33x3) + observation masks.

User rule: <= 2 worker processes. One .npz per clip (resumable: existing files are skipped).
Sentence text is never read."""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE.parents[0] / "runtime" / "cip_teacher_how2sign"
H2S = Path("/home/shared_data/sign_language/How2Sign/train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cores", default="0-1")
    a = ap.parse_args()
    import torch  # CUDA before decord (known SIGSEGV otherwise)
    torch.cuda.init()
    import numpy as np
    from extract import Encoder
    import multiprocessing as mp
    import preprocess
    OUT.mkdir(parents=True, exist_ok=True)
    # same 8,000-clip sample as EX08 (seed 2026), first n of it
    rng = np.random.default_rng(2026)
    names = sorted(p.stem for p in (H2S / "train_pose").glob("*.pkl"))
    names = list(rng.choice(names, 8000, replace=False))[: a.n]
    vids = []
    for nm in names:
        v = H2S / "raw_videos" / f"{nm}.mp4"
        if not v.exists():
            v = H2S / "raw_videos" / f"{nm.replace('-rgb_front', '')}.mp4"
        vids.append(v)
    missing = sum(not v.exists() for v in vids)
    print(f"{len(vids)} clips, missing videos {missing}", flush=True)
    jobs = [(i, str(v), None, None) for i, (nm, v) in enumerate(zip(names, vids))
            if v.exists() and not (OUT / f"{nm}.npz").exists()]
    enc = Encoder("cuda", 256)
    ctx = mp.get_context("spawn")
    cores = list(range(int(a.cores.split("-")[0]), int(a.cores.split("-")[1]) + 1))
    counter = ctx.Value("i", 0)
    t0, done, fails = time.time(), 0, 0
    with ctx.Pool(a.workers, initializer=preprocess.init_worker,
                  initargs=("cuda", 1, "pose", cores, counter, True)) as pool:
        for item in pool.imap(preprocess.process_clip, iter(jobs), chunksize=1):
            nm = names[item["index"]]
            done += 1
            if item["status"] != "ok":
                fails += 1
                json.dump({"error": item.get("error")}, open(OUT / f"{nm}.fail.json", "w"))
                continue
            with torch.inference_mode():
                st = item["streams"]
                face = enc.dino(enc.dinos["face"], st["face"])
                lh = enc.dino(enc.dinos["hand"], st["left_hand"])
                rh = enc.dino(enc.dinos["hand"], st["right_hand"])
            app = torch.stack([lh, rh, face], 1).half().cpu().numpy()  # n,3,384 order LH,RH,FACE
            tmp = OUT / f"{nm}.tmp.npz"
            np.savez(tmp, app=app, hand_lm=item["hand_lm"], pose_lm=item["pose_lm"], observed=item["observed"],
                     fps=np.float32(item.get("fps") or 0))
            tmp.rename(OUT / f"{nm}.npz")
            if done % 20 == 0:
                el = time.time() - t0
                print(f"progress {done}/{len(jobs)} {done/el:.3f} clips/s fails={fails} eta={(len(jobs)-done)/(done/el)/3600:.2f}h", flush=True)
    print("DONE", done, fails, flush=True)


if __name__ == "__main__":
    main()
