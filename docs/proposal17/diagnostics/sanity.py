import time, numpy as np
from common import *
t=time.time(); d=load_baseline(); print('load',time.time()-t, d['train']['X'].shape, d['val']['X'].shape, len(d['glosses']))
tr,va=d['train'],d['val']
print('signers train',len(np.unique(tr['signer'])),'val',sorted(np.unique(va['signer'],return_counts=True)[0]), np.unique(va['signer'],return_counts=True)[1])
print('zero vec', (np.linalg.norm(tr['X'],axis=1)==0).sum(), (np.linalg.norm(va['X'],axis=1)==0).sum())
t=time.time(); r=class_max_ranks(va['X'],va['y'],tr['X'],tr['y']); print('rank time',time.time()-t)
print(metrics(r))
