import csv
import numpy as np
from .linear_probe import linear_probe

DEFAULT_PROPERTIES = ["Handshape.2.0","MajorLocation.2.0","MinorLocation.2.0","Movement.2.0",
                      "SelectedFingers.2.0","Contact.2.0","SignType.2.0"]
MISSING = {"","NA","N/A","nan","None"}


def load_metadata(path,encoding="utf-8-sig"):
    with open(path,newline="",encoding=encoding) as f:
        rows = list(csv.DictReader(f))
    result = {r["Code"]:r for r in rows}
    if len(result)!=len(rows):
        raise AssertionError("ASL-LEX Code is not unique; specify an unambiguous mapping")
    return result


def phonological_probes(train_x,train_rows,test_x,test_rows,metadata,properties=None,seed=42,
                       require_signer_disjoint=True,require_gloss_disjoint=False):
    if require_signer_disjoint:
        tr = {r.get("signer_id") for r in train_rows}
        te = {r.get("signer_id") for r in test_rows}
        if None in tr or None in te or tr&te:
            raise AssertionError("Phonology signer-independent protocol needs complete, disjoint signer IDs")
    if require_gloss_disjoint and {r["gloss"] for r in train_rows}&{r["gloss"] for r in test_rows}:
        raise AssertionError("Gloss-aware phonology split overlaps")
    results = {}
    for prop in properties or DEFAULT_PROPERTIES:
        def aligned(rows):
            pairs = [(i,metadata.get(r.get("asllex"),{}).get(prop)) for i,r in enumerate(rows)]
            return [(i,y) for i,y in pairs if y is not None and str(y) not in MISSING]
        train,test = aligned(train_rows),aligned(test_rows)
        if not train or not test or len({y for _,y in train})<2:
            results[prop] = {"status":"insufficient_coverage","train_samples":len(train),"test_samples":len(test)}
            continue
        ti,ty = zip(*train)
        ei,ey = zip(*test)
        results[prop] = {"status":"evaluated",**linear_probe(np.asarray(train_x)[list(ti)],ty,
                                                             np.asarray(test_x)[list(ei)],ey,seed),
                         "signer_independent":require_signer_disjoint,"gloss_disjoint":require_gloss_disjoint}
    return results
