import json
from pathlib import Path
import numpy as np
import torch
import yaml
from signrel.data.synthetic import make_synthetic
from signrel.data.cache import read_cache
from signrel.models.encoder import initialized_model
from signrel.eval.embed import embed_arrays
from signrel.reporting import minimal_pairs,aggregate,build_report
from signrel.utils import write_jsonl
from signrel.config import load_config,ROOT


def test_real_encoder_pool_is_padding_invariant(tmp_path):
    make_synthetic(tmp_path,frames=20)
    arrays,_=read_cache(tmp_path/'caches/synthetic_0000.npz')
    cfg=load_config(ROOT/'configs/smoke_a1.yaml').values
    model=initialized_model(cfg['model'],17).eval()
    first=embed_arrays(arrays,model,32,16)['embedding']
    padded=embed_arrays(arrays,model,64,32)['embedding']
    assert np.allclose(first,padded,atol=2e-6)


def test_minimal_pair_cli_artifact_clusters_and_missing(tmp_path):
    np.save(tmp_path/'z.npy',np.array([[1,0],[1,.1],[0,1]],np.float32))
    write_jsonl(tmp_path/'index.jsonl',[{'video_id':i,'row_index':j,'status':'ok'} for j,i in enumerate(('a','p','n'))])
    row={'triplet_id':'t','anchor_video_id':'a','positive_video_id':'p','minimal_pair_video_id':'n',
         'anchor_gloss_id':'A','positive_gloss_id':'A','negative_gloss_id':'B','property':'synthetic_handshape',
         'verified_by':'synthetic_fixture','annotation_version':'fixture-v1','signer_control':'fixture','source_control':'fixture'}
    write_jsonl(tmp_path/'pairs.jsonl',[row])
    cfg={'schema_version':1,'annotations':str(tmp_path/'pairs.jsonl'),'embeddings':str(tmp_path/'z.npy'),
         'index':str(tmp_path/'index.jsonl'),'output':str(tmp_path/'result.json'),'synthetic':True}
    path=tmp_path/'cfg.yaml';path.write_text(yaml.safe_dump(cfg))
    result=minimal_pairs(path)
    assert result['accuracy']==1 and result['bootstrap_clusters']==1 and result['margin_ci95'][0]>0
    cfg['annotations']=str(tmp_path/'missing.jsonl');path.write_text(yaml.safe_dump(cfg))
    assert minimal_pairs(path)['status']=='BLOCKED_EXTERNAL'
