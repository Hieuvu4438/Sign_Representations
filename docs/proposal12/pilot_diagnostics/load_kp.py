"""Load ASL Citizen COCO-WholeBody keypoints into compact per-clip arrays (diagnostics only)."""
import csv, pickle, sys, os
import numpy as np
from multiprocessing import Pool

KP_DIR = '/home/dongvk/datasets/ASL_Citizen/keypoints'
SPLIT_DIR = '/home/dongvk/datasets/ASL_Citizen/splits'
BODY = [0, 5, 6, 7, 8, 9, 10, 11, 12]            # nose, shoulders, elbows, wrists, hips
FACE = list(range(23, 91))                       # 68 iBUG face points
LH = list(range(91, 112))
RH = list(range(112, 133))
SEL = BODY + FACE + LH + RH                      # 9 + 68 + 21 + 21 = 119


def load(row):
    stem = os.path.splitext(row['Video file'])[0]
    p = os.path.join(KP_DIR, stem + '.pkl')
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:
        d = pickle.load(f)
    kp = np.concatenate([np.asarray(k, np.float32) for k in d['keypoints']], 0)   # T,133,2
    sc = np.concatenate([np.asarray(s, np.float32) for s in d['scores']], 0)      # T,133
    return dict(stem=stem, kp=kp[:, SEL].astype(np.float16), sc=sc[:, SEL].astype(np.float16),
                signer=row['Participant ID'], gloss=row['Gloss'], code=row['ASL-LEX Code'])


def main(split, n_max, out):
    rows = list(csv.DictReader(open(os.path.join(SPLIT_DIR, split + '.csv'))))
    rng = np.random.default_rng(0)
    if n_max and len(rows) > n_max:
        rows = [rows[i] for i in sorted(rng.choice(len(rows), n_max, replace=False))]
    with Pool(8) as pool:
        res = [r for r in pool.map(load, rows, chunksize=64) if r is not None]
    with open(out, 'wb') as f:
        pickle.dump(res, f)
    print(split, len(res))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
