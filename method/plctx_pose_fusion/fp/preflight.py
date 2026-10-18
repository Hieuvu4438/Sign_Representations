import json,hashlib,sys
import torch
import numpy as np
from pathlib import Path
from fp.paths import ROOT,ASLC_SPLITS,ASLC_KP
from fp.inference import split_rows
from fp.data.mapped import MappedCache
from fp.utils import sha256,atomic_json


def verify(c):
    cache=MappedCache(c['cache']);rows=split_rows('train')
    if not c['smoke'] and (len(cache)!=40154 or cache.ids.tolist()!=[r['sample_id'] for r in rows]):
        raise ValueError('Full cache IDs/order mismatch')
    if not np.all(np.diff(cache.offs)>0) or cache.xy.shape[1:]!=(103,2):raise ValueError('Malformed cache')
    ids={s:{r['sample_id'] for r in split_rows(s)} for s in ('train','val','test')}
    if any(ids[a]&ids[b] for a,b in [('train','val'),('train','test'),('val','test')]):raise ValueError('Split contamination')
    sg={s:{r['signer'] for r in split_rows(s)} for s in ids}
    if any(sg[a]&sg[b] for a,b in [('train','val'),('train','test'),('val','test')]):raise ValueError('Signer contamination')
    proof={'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'fp').rglob('*.py'))},
           'cache_sha256':sha256(c['cache']),'pl_checkpoint_sha256':sha256(c['pl_checkpoint']),
           'split_sha256':{s:sha256(ASLC_SPLITS/(s+'.csv')) for s in ids},
           'counts':{s:len(v) for s,v in ids.items()},'PL_pretraining':'full160epoch already consumed; additional fusion160epochs'}
    # Fast complete inventory guard for validation/test source files. The training
    # arrays are independently byte-hashed above. This is a stat inventory, not a
    # claim that all original detector/video bytes have been content-hashed.
    h=hashlib.sha256()
    for sid in sorted(set.union(*ids.values())):
        stat=(ASLC_KP/(sid+'.pkl')).stat()
        h.update(f'{sid}:{stat.st_size}:{stat.st_mtime_ns}\n'.encode())
    proof['keypoint_stat_inventory_sha256']=h.hexdigest()
    proof['environment']={'torch':torch.__version__,'numpy':np.__version__,'python':sys.version}
    manifest=Path(c['cache']).with_suffix('.manifest.json')
    if manifest.exists() and json.loads(manifest.read_text())['cache_sha256']!=proof['cache_sha256']:
        raise ValueError('Full cache differs from original provenance')
    if not c['smoke'] and proof['pl_checkpoint_sha256']!='8f8a10687f2df869fd0d7f4703edfeb467c59977d7baaacb2d567c15789e41f9':
        raise ValueError('Unexpected PLctx full160 reference weights')
    return cache,proof
