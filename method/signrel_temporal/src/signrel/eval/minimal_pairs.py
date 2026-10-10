import numpy as np


def score_triplets(vectors,rows):
    measured,missing = [],[]
    for row in rows:
        required = ['triplet_id','anchor_video_id','positive_video_id','minimal_pair_video_id','anchor_gloss_id',
                    'positive_gloss_id','negative_gloss_id','property','verified_by','annotation_version','signer_control','source_control']
        if any(not row.get(k) for k in required): raise ValueError('Minimal pairs require verified annotation and controls')
        if row['anchor_gloss_id']!=row['positive_gloss_id'] or row['anchor_gloss_id']==row['negative_gloss_id']:
            raise ValueError('Invalid positive/negative gloss correspondence')
        ids = [row[k] for k in ('anchor_video_id','positive_video_id','minimal_pair_video_id')]
        if any(k not in vectors for k in ids): missing.append(row['triplet_id']); continue
        z = [np.asarray(vectors[k],dtype=np.float64) for k in ids]
        if any(not np.isfinite(x).all() or np.linalg.norm(x)==0 for x in z): raise ValueError('Invalid triplet embedding')
        z = [x/np.linalg.norm(x) for x in z]
        margin = float(z[0]@z[1]-z[0]@z[2])
        measured.append({'triplet_id':row['triplet_id'],'margin':margin,'property':row['property']})
    return {'status':'evaluated' if measured else 'insufficient_coverage','attempted':len(rows),'scored':len(measured),
            'coverage':len(measured)/len(rows) if rows else 0,'missing_triplets':missing,
            'mean_margin':float(np.mean([r['margin'] for r in measured])) if measured else None,
            'accuracy':float(np.mean([r['margin']>0 for r in measured])) if measured else None,
            'ties':sum(r['margin']==0 for r in measured),'per_triplet':measured}
