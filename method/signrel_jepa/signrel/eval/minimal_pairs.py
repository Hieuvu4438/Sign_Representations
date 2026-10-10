import itertools
import numpy as np
from .retrieval import normalize


def minimal_pairs(features,rows,metadata,properties=None):
    properties = properties or ["Handshape.2.0","MajorLocation.2.0","Movement.2.0"]
    z = normalize(features)
    groups = {}
    for i,r in enumerate(rows):
        entry = metadata.get(r.get("asllex"),{})
        signature = tuple(entry.get(p) for p in properties)
        if any(v in {None,"","NA","N/A"} for v in signature):
            continue
        key = r["gloss"]
        if key in groups and groups[key]["signature"]!=signature:
            raise AssertionError("Ambiguous lexical/phonology mapping")
        groups.setdefault(key,{"signature":signature,"rows":[]})["rows"].append(i)
    same,minimal,unrelated = [],[],[]
    per_property = {p:[] for p in properties}
    for name,item in groups.items():
        for i,j in itertools.combinations(item["rows"],2):
            if rows[i].get("signer_id")!=rows[j].get("signer_id"):
                same.append(float(1-z[i]@z[j]))
    for (a,ga),(b,gb) in itertools.combinations(groups.items(),2):
        changed = [i for i,(v,w) in enumerate(zip(ga["signature"],gb["signature"])) if v!=w]
        distances = (1-z[ga["rows"]]@z[gb["rows"]].T).flatten().tolist()
        if len(changed)==1:
            minimal.extend(distances)
            per_property[properties[changed[0]]].extend(distances)
        elif len(changed)==len(properties):
            unrelated.extend(distances)
    def summary(v):
        return {"n":len(v),"mean":float(np.mean(v)) if v else None}
    return {"same_sign_different_signer":summary(same),"minimal_pair":summary(minimal),
            "unrelated":summary(unrelated),"per_property":{p:summary(v) for p,v in per_property.items()},
            "status":"evaluated" if same and minimal and unrelated else "insufficient_coverage",
            "definition":"differ in exactly one selected property; exploratory, other factors not fully controlled"}
