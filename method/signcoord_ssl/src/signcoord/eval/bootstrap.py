import numpy as np


def paired_group_bootstrap(a,b,metric="rec1",draws=2000,seed=3407):
    if [r["query_id"] for r in a] != [r["query_id"] for r in b]: raise ValueError("paired query IDs/order mismatch")
    if [r["group"] for r in a] != [r["group"] for r in b]: raise ValueError("paired groups mismatch")
    groups=sorted({r["group"] for r in a})
    diff=np.array([x[metric]-y[metric] for x,y in zip(a,b)])
    indices=[np.array([i for i,r in enumerate(a) if r["group"]==g]) for g in groups]
    rng=np.random.default_rng(seed)
    estimates=[float(diff[np.concatenate([indices[i] for i in rng.integers(len(groups),size=len(groups))])].mean()) for _ in range(draws)]
    return {"difference":float(diff.mean()),"ci95":np.quantile(estimates,[.025,.975]).tolist(),
            "groups":len(groups),"draws":draws,"seed":seed,"few_groups_limitation":len(groups)<10,
            "seed_variation_included":False}
