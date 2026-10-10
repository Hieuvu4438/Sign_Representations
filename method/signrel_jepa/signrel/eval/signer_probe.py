"""Signer prediction on known signers with gloss-disjoint training/evaluation."""
import hashlib
import numpy as np
from .linear_probe import linear_probe


def signer_probe(features,rows,seed=42,train_fraction=0.7,min_per_partition=2):
    def training_gloss(gloss):
        key = hashlib.sha256(f"{seed}:{gloss}".encode()).digest()
        return int.from_bytes(key[:8],"big")/2**64 < train_fraction
    tr,te = [],[]
    signers = sorted({str(r["signer_id"]) for r in rows if r.get("signer_id") is not None})
    retained = []
    for signer in signers:
        train = [i for i,r in enumerate(rows) if str(r.get("signer_id"))==signer and training_gloss(r["gloss"])]
        test = [i for i,r in enumerate(rows) if str(r.get("signer_id"))==signer and not training_gloss(r["gloss"])]
        if min(len(train),len(test))>=min_per_partition:
            tr.extend(train)
            te.extend(test)
            retained.append(signer)
    if len(retained)<2:
        return {"status":"insufficient_coverage","retained_signers":retained}
    if {rows[i]["gloss"] for i in tr}&{rows[i]["gloss"] for i in te}:
        raise AssertionError("Signer probe gloss overlap")
    result = linear_probe(np.asarray(features)[tr],[rows[i]["signer_id"] for i in tr],
                          np.asarray(features)[te],[rows[i]["signer_id"] for i in te],seed)
    return {"status":"evaluated",**result,"retained_signers":retained,"gloss_disjoint":True,
            "interpretation":"Lower signer accuracy is useful only at maintained linguistic quality"}
