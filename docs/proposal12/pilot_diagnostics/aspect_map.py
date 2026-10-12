"""Read W,H of every ASL Citizen video (train + val) -> JSON {stem: W/H}."""
import csv, os, sys, json, cv2
from multiprocessing import Pool
cv2.setNumThreads(1)
VID = '/home/dongvk/datasets/ASL_Citizen/videos'
def wh(fn):
    cap = cv2.VideoCapture(os.path.join(VID, fn)); w, h = cap.get(3), cap.get(4); cap.release()
    return os.path.splitext(fn)[0], (w / h if h else 4 / 3)
rows = []
for sp in ['train', 'val']:
    rows += [r['Video file'] for r in csv.DictReader(open(f'/home/dongvk/datasets/ASL_Citizen/splits/{sp}.csv'))]
with Pool(8) as p:
    out = dict(p.map(wh, rows, chunksize=64))
json.dump(out, open(sys.argv[1], 'w'))
import collections; print(collections.Counter(round(v, 3) for v in out.values()).most_common(5))
