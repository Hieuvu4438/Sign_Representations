"""Frozen feature diagnostics. Labels here never enter SSL training."""
import itertools
import numpy as np
from signrel.eval.retrieval import normalize


MISSING = {None, "", "NA", "N/A", "nan"}


def embedding_health(features):
    z = normalize(features).astype(np.float64)
    if len(z) < 2:
        raise ValueError("Embedding diagnostics require at least two samples")
    centered = z-z.mean(0)
    eigenvalues = np.linalg.svd(centered, compute_uv=False)**2
    energy = float(eigenvalues.sum())
    if energy > 0:
        p = eigenvalues[eigenvalues > 0]/energy
        rank = float(np.exp(-(p*np.log(p)).sum()))
    else:
        rank = 0.
    # Sum of off-diagonal dot products avoids allocating an N x N matrix.
    mean_cos = float((np.square(z.sum(0)).sum()-np.square(z).sum())/(len(z)*(len(z)-1)))
    return {"samples":len(z), "dimensions":z.shape[1],
            "mean_pairwise_cosine":mean_cos,
            "mean_per_dimension_std":float(centered.std(0).mean()),
            "centered_energy":energy, "effective_rank":rank,
            "leading_component_energy_fraction":float(eigenvalues[0]/energy) if energy > 0 else None,
            "interpretation":"Descriptive geometry, not a stand-alone collapse verdict; compare frozen task quality"}


def lexical_signatures(rows, metadata, properties):
    signatures, excluded = {}, []
    for r in rows:
        entry = metadata.get(r.get("asllex"), {})
        signature = tuple(entry.get(p) for p in properties)
        if any(v in MISSING for v in signature):
            excluded.append(r["sample_id"])
            continue
        gloss = str(r["gloss"])
        if gloss in signatures and signatures[gloss] != signature:
            raise AssertionError("Ambiguous lexical/phonology mapping")
        signatures[gloss] = signature
    return signatures, excluded


def eligible_gallery(query_row, gallery_rows, exclude_same_signer=True):
    if exclude_same_signer and not query_row.get("signer_id"):
        raise ValueError("Signer exclusion requires query signer metadata")
    return np.array([r["sample_id"] != query_row["sample_id"] and
                     not (r.get("source_video_id") and r.get("source_video_id") == query_row.get("source_video_id")) and
                     not (r.get("file_sha256") and r.get("file_sha256") == query_row.get("file_sha256")) and
                     not (exclude_same_signer and (not r.get("signer_id") or r["signer_id"] == query_row["signer_id"]))
                     for r in gallery_rows], dtype=bool)


def minimal_pair_retrieval(query, gallery, q_rows, g_rows, metadata, properties=None,
                           exclude_same_signer=True):
    """Class-max retrieval among same gloss and exactly-one-property distractors.

    Matching is exact on every other selected property. Unselected linguistic
    factors remain uncontrolled and the output explicitly records that limit.
    """
    properties = properties or ["Handshape.2.0", "MajorLocation.2.0", "Movement.2.0"]
    if len(properties) < 2 or len(set(properties)) != len(properties):
        raise ValueError("At least two distinct matching properties are required")
    q, g = normalize(query), normalize(gallery)
    if len(q) != len(q_rows) or len(g) != len(g_rows):
        raise ValueError("Feature/metadata row mismatch")
    signatures, excluded = lexical_signatures(q_rows+g_rows, metadata, properties)
    neighbors = {name:[] for name in signatures}
    pairs = []
    for (a, sa), (b, sb) in itertools.combinations(sorted(signatures.items()), 2):
        changed = [i for i, (v, w) in enumerate(zip(sa, sb)) if v != w]
        if len(changed) == 1:
            p = properties[changed[0]]
            neighbors[a].append((b,p))
            neighbors[b].append((a,p))
            pairs.append({"gloss_a":a, "gloss_b":b, "changed_property":p,
                          "matched_properties":[v for v in properties if v != p]})
    labels = np.array([str(r["gloss"]) for r in g_rows])
    details, skipped = [], []
    for i, row in enumerate(q_rows):
        gloss = str(row["gloss"])
        if gloss not in signatures or not neighbors[gloss]:
            skipped.append({"sample_id":row["sample_id"], "reason":"no_complete_minimal_pair_metadata"})
            continue
        eligible = eligible_gallery(row, g_rows, exclude_same_signer)
        classes = [gloss]+sorted({n for n,_ in neighbors[gloss]})
        scores = q[i]@g.T
        class_scores = {c:float(scores[(labels == c)&eligible].max()) for c in classes if ((labels == c)&eligible).any()}
        if gloss not in class_scores or len(class_scores) < 2:
            skipped.append({"sample_id":row["sample_id"], "reason":"no_eligible_positive_or_distractor"})
            continue
        # Stable lexical tie-breaking, rather than favoring the positive class.
        ranked = sorted(class_scores, key=lambda c:(-class_scores[c], c))
        predicted = ranked[0]
        per_property = []
        for p in properties:
            distractors = [n for n, prop in neighbors[gloss] if prop == p and n in class_scores]
            if distractors:
                best = sorted([gloss]+distractors, key=lambda c:(-class_scores[c],c))[0]
                per_property.append({"property":p, "correct":best == gloss, "distractors":len(distractors)})
        details.append({"sample_id":row["sample_id"], "gloss":gloss, "predicted_gloss":predicted,
                        "rank":ranked.index(gloss)+1, "candidate_classes":len(ranked),
                        "correct":predicted == gloss,
                        "positive_margin":class_scores[gloss]-max(v for c,v in class_scores.items() if c != gloss),
                        "per_property":per_property})
    def summary(items):
        return {"queries":len(items), "Recall@1":float(np.mean([x["correct"] for x in items])) if items else None}
    by_gloss = {}
    for d in details:
        by_gloss.setdefault(d["gloss"], []).append(d["correct"])
    return {"status":"evaluated" if details else "insufficient_coverage", **summary(details),
            "class_macro_Recall@1":float(np.mean([np.mean(v) for v in by_gloss.values()])) if by_gloss else None,
            "queries_total":len(q), "query_coverage":len(details)/len(q) if len(q) else 0.,
            "eligible_glosses":len(by_gloss), "lexical_pairs":pairs, "properties":properties,
            "per_property":{p:summary([v for d in details for v in d["per_property"] if v["property"] == p]) for p in properties},
            "metadata_excluded_sample_ids":excluded, "skipped":skipped, "per_query":details,
            "same_signer_excluded":exclude_same_signer,
            "protocol":"max cosine per gloss; positive and exactly-one-selected-property distractors; self/source/content excluded",
            "limitation":"Unselected phonological factors are uncontrolled; this is an exploratory matched-property diagnostic"}


def nearest_neighbors(query, gallery, q_rows, g_rows, k=5, exclude_same_signer=True):
    q, g = normalize(query), normalize(gallery)
    if len(q) != len(q_rows) or len(g) != len(g_rows) or k < 1:
        raise ValueError("Invalid neighbor inputs")
    out = []
    for i, row in enumerate(q_rows):
        allowed = eligible_gallery(row, g_rows, exclude_same_signer)
        scores = q[i]@g.T
        ranked = np.argsort(-scores, kind="stable")
        ranked = ranked[allowed[ranked]][:k]
        out.append({"query_index":i, "sample_id":row["sample_id"],
                    "neighbors":[{"gallery_index":int(j), "sample_id":g_rows[j]["sample_id"],
                                  "gloss":g_rows[j]["gloss"], "cosine":float(scores[j]),
                                  "same_gloss":g_rows[j]["gloss"] == row["gloss"]} for j in ranked]})
    return out
