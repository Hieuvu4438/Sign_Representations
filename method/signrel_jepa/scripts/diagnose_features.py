"""Inspect frozen embeddings without modifying the original evaluation artifacts."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p/"signrel/config.py").exists())
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from signrel.data.manifest import sha256
from signrel.eval.phonology import load_metadata

try:
    from signrel.eval.diagnostics import embedding_health, minimal_pair_retrieval, nearest_neighbors
except ImportError:
    spec = importlib.util.spec_from_file_location("pending_diagnostics",ROOT/"pending_source/signrel/eval/diagnostics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    embedding_health, minimal_pair_retrieval, nearest_neighbors = module.embedding_health,module.minimal_pair_retrieval,module.nearest_neighbors


def load(path):
    with np.load(path,allow_pickle=False) as z:
        return z["features"], [json.loads(v) for v in z["rows_json"].tolist()]


def thumbnail(row, caption, width=200, height=180):
    tile = np.zeros((height,width,3),dtype=np.uint8)
    cap = cv2.VideoCapture(row["video_path"])
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        start = (row.get("start_sec") or 0)*fps
        end = row.get("end_sec")*fps if row.get("end_sec") is not None else count
        cap.set(cv2.CAP_PROP_POS_FRAMES,int((start+end)/2))
        ok, frame = cap.read()
        if ok:
            h,w = frame.shape[:2]
            scale = min(width/w,(height-40)/h)
            frame = cv2.resize(frame,(max(1,int(w*scale)),max(1,int(h*scale))))
            tile[:frame.shape[0],:frame.shape[1]] = frame
        else:
            cv2.putText(tile,"decode unavailable",(4,30),cv2.FONT_HERSHEY_SIMPLEX,.4,(80,80,255),1)
    finally:
        cap.release()
    cv2.putText(tile,str(row["gloss"])[:23],(4,height-24),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
    cv2.putText(tile,caption[:27],(4,height-8),cv2.FONT_HERSHEY_SIMPLEX,.35,(220,220,220),1)
    return tile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features",required=True)
    p.add_argument("--gallery",required=True)
    p.add_argument("--metadata",required=True)
    p.add_argument("--encoding",default="latin-1")
    p.add_argument("--properties",nargs="+",default=["Handshape.2.0","MajorLocation.2.0","Movement.2.0"])
    p.add_argument("--out",required=True)
    p.add_argument("--neighbors",type=int,default=5)
    p.add_argument("--visual-queries",type=int,default=12)
    p.add_argument("--seed",type=int,default=42)
    args = p.parse_args()
    q,qr = load(args.features)
    g,gr = load(args.gallery)
    metadata = load_metadata(args.metadata,args.encoding)
    # Random display selection independent of scores or correctness.
    ids = sorted(np.random.default_rng(args.seed).choice(len(q),min(args.visual_queries,len(q)),replace=False).tolist())
    neighbors = nearest_neighbors(q[ids],g,[qr[i] for i in ids],gr,args.neighbors)
    report = {"feature_sha256":sha256(args.features),"gallery_sha256":sha256(args.gallery),
              "metadata_sha256":sha256(args.metadata),"script_sha256":sha256(__file__),
              "diagnostics_source_sha256":sha256(module.__file__) if "module" in globals() else sha256(ROOT/"signrel/eval/diagnostics.py"),
              "query_embedding_health":embedding_health(q),"gallery_embedding_health":embedding_health(g),
              "minimal_pair_retrieval":minimal_pair_retrieval(q,g,qr,gr,metadata,args.properties),
              "neighbor_selection":{"seed":args.seed,"query_indices":ids,"protocol":"random queries, instance cosine ranking, self/source/content/signer excluded"},
              "neighbors":neighbors}
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    if ids:
        tiles = []
        for index,item in zip(ids,neighbors):
            cells = [thumbnail(qr[index],"query")]
            for neighbor in item["neighbors"]:
                cells.append(thumbnail(gr[neighbor["gallery_index"]],f"cos={neighbor['cosine']:.4f} {'match' if neighbor['same_gloss'] else ''}"))
            cells += [np.zeros_like(cells[0])]*(args.neighbors+1-len(cells))
            tiles.append(np.concatenate(cells,axis=1))
        image_path = out.with_suffix(".neighbors.jpg")
        if not cv2.imwrite(str(image_path),np.concatenate(tiles,axis=0)):
            raise OSError(f"Cannot write {image_path}")
        report["neighbors_image"] = str(image_path)
    out.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({"features":args.features,"health":report["query_embedding_health"],
                      "minimal_pairs":{"Recall@1":report['minimal_pair_retrieval']['Recall@1'],"queries":report['minimal_pair_retrieval']['queries']}}))


if __name__ == "__main__":
    main()
