"""Frozen gloss-level retrieval with audited query/gallery identities."""
import numpy as np


def score_retrieval(query,query_rows,gallery,gallery_rows,require_signer_disjoint=True):
    query,gallery = np.asarray(query,np.float32),np.asarray(gallery,np.float32)
    if query.ndim!=2 or gallery.ndim!=2 or query.shape[1]!=gallery.shape[1]: raise ValueError('Embedding dimensions mismatch')
    if len(query)!=len(query_rows) or len(gallery)!=len(gallery_rows): raise ValueError('Embedding/index mismatch')
    if not np.isfinite(query).all() or not np.isfinite(gallery).all(): raise ValueError('Non-finite embeddings')
    if len(set(r['video_id'] for r in query_rows))!=len(query_rows) or len(set(r['video_id'] for r in gallery_rows))!=len(gallery_rows):
        raise ValueError('Duplicate query/gallery IDs')
    if require_signer_disjoint:
        qs,gs = {r.get('signer_id') for r in query_rows},{r.get('signer_id') for r in gallery_rows}
        if None in qs|gs or '' in qs|gs or qs&gs: raise ValueError('Known disjoint query/gallery signers required')
    for q in query_rows:
        for g in gallery_rows:
            if q['video_id']==g['video_id'] or (q.get('sha256') and q['sha256']==g.get('sha256')):
                raise ValueError('Exact duplicate query/gallery video')
            if q.get('source_recording_id') and q['source_recording_id']==g.get('source_recording_id'):
                raise ValueError('Query/gallery share a recording')
    qnorm,gnorm = np.linalg.norm(query,axis=1,keepdims=True),np.linalg.norm(gallery,axis=1,keepdims=True)
    if (qnorm<=0).any() or (gnorm<=0).any(): raise ValueError('Zero vector cannot enter normal retrieval')
    query,gallery = query/qnorm,gallery/gnorm
    classes = sorted(set(r['gloss_id'] for r in gallery_rows))
    if not classes: raise ValueError('Empty gallery')
    indices = {c:[i for i,r in enumerate(gallery_rows) if r['gloss_id']==c] for c in classes}
    scores = query@gallery.T
    per_query,excluded = [],[]
    for i,row in enumerate(query_rows):
        if row['gloss_id'] not in indices:
            excluded.append(row['video_id']); continue
        class_scores = np.array([scores[i,indices[c]].max() for c in classes])
        ordered = np.argsort(-class_scores,kind='stable')
        rank = list(ordered).index(classes.index(row['gloss_id']))+1
        per_query.append({'video_id':row['video_id'],'gloss_id':row['gloss_id'],'rank':rank,
                          'r1':float(rank==1),'r5':float(rank<=min(5,len(classes))),'rr':1/rank})
    if not per_query: raise ValueError('No covered queries')
    macro = lambda key: float(np.mean([np.mean([r[key] for r in per_query if r['gloss_id']==c]) for c in sorted(set(r['gloss_id'] for r in per_query))]))
    return {'macro_gloss_r1':macro('r1'),'macro_gloss_r5':macro('r5'),
            'micro_r1':float(np.mean([r['r1'] for r in per_query])),'r5':float(np.mean([r['r5'] for r in per_query])),
            'mrr':float(np.mean([r['rr'] for r in per_query])),'queries_attempted':len(query_rows),
            'queries_scored':len(per_query),'excluded_no_gallery':excluded,'query_coverage':len(per_query)/len(query_rows),
            'gallery_size':len(gallery),'gallery_glosses':len(classes),'per_query':per_query,
            'ranking':'gloss max exemplar cosine; stable gloss_id ties'}
