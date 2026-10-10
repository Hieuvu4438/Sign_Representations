"""Instance retrieval with deterministic ID ties; labels stay in evaluation."""
import numpy as np


def evaluate_retrieval(gallery,queries,gallery_labels,query_labels,gallery_ids,query_ids,groups=None,chunk=256):
    if len(set(gallery_ids)) != len(gallery_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError("duplicate gallery/query IDs")
    if set(gallery_ids)&set(query_ids): raise ValueError("gallery/query self overlap")
    if not np.isfinite(gallery).all() or not np.isfinite(queries).all(): raise ValueError("nonfinite features")
    if not np.allclose(np.linalg.norm(gallery,axis=1),1,atol=1e-5) or not np.allclose(np.linalg.norm(queries,axis=1),1,atol=1e-5):
        raise ValueError("instance retrieval requires normalized clip embeddings")
    rows=[]; ids=np.asarray(gallery_ids)
    for start in range(0,len(queries),chunk):
        sims=queries[start:start+chunk]@gallery.T
        for j,scores in enumerate(sims):
            i=start+j; ranked=np.lexsort((ids,-scores))
            relevant=np.asarray(gallery_labels)[ranked]==query_labels[i]
            hits=np.flatnonzero(relevant); rank=int(hits[0]+1) if len(hits) else None
            rows.append({"query_id":query_ids[i],"label":str(query_labels[i]),"group":str(groups[i]) if groups is not None else query_ids[i],
                         "first_relevant_rank":rank,"rec1":float(rank is not None and rank<=1),
                         "rec5":float(rank is not None and rank<=5),"rec10":float(rank is not None and rank<=10),
                         "rr":1/rank if rank is not None else 0.,"top_gallery_ids":ids[ranked[:10]].tolist()})
    if not rows: raise ValueError("empty queries")
    metrics={k:float(np.mean([r[k] for r in rows])) for k in ("rec1","rec5","rec10","rr")}
    for by in ("label","group"):
        for metric in ("rec1","rr"):
            keys=sorted({r[by] for r in rows})
            metrics[f"macro_{by}_{metric}"]=float(np.mean([np.mean([r[metric] for r in rows if r[by]==key]) for key in keys]))
    return {"protocol":"frozen_instance_cosine_id_ties_v1","gallery_size":len(gallery),"query_count":len(rows),
            "queries_without_gallery_match":sum(r["first_relevant_rank"] is None for r in rows),"metrics":metrics,"per_query":rows}
