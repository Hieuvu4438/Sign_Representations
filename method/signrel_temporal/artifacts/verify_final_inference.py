"""Compare final inference implementation to every scored frozen learned export."""
import json
from pathlib import Path
import numpy as np
import torch
from signrel.utils import read_jsonl, atomic_json, sha256, code_hash
from signrel.config import ROOT
from signrel.eval.embed import restore_encoder, embed_arrays, embed_cached
from signrel.data.provenance import inference_expectations
from signrel.data.cache import read_cache
from signrel.train.checkpoint import atomic_checkpoint

status=json.loads(Path('artifacts/matrix_status.json').read_text())
assert status.get('test_evaluated') is True
assert Path('artifacts/postmatrix_validation_patch.json').exists()
torch.set_num_threads(4)
items=[{'run_dir':f'runs/pilot_{v}_seed17','eval_dir':f'results/pilot_{v}_seed17_dev'} for v in ('a1','a4')]
items.extend(r for r in status['rows'] if r['state']=='complete')
caches={r['video_id']:r for r in read_jsonl('data/cache/eval_cache_manifest.jsonl')}
records=[]
portable_root=Path('results/portable_encoders');portable_root.mkdir(exist_ok=True)
for item in items:
    directory=Path(item['eval_dir'])
    vectors=np.load(directory/'embeddings.npy',allow_pickle=False)
    index=read_jsonl(directory/'index.jsonl')
    checkpoint=Path(item['run_dir'])/'checkpoints/last.pt'
    full,cfg,state=restore_encoder(checkpoint)
    expected=inference_expectations(state,cfg)
    dedicated,_,_=restore_encoder(directory/'encoder.pt')
    assert not hasattr(dedicated,'heads')
    maximum_full=maximum_dedicated=0.
    retained=0
    for row in index:
        arrays,_=read_cache(caches[row['video_id']]['cache_path'],expected)
        for model,kind in ((full,'full'),(dedicated,'dedicated')):
            result=embed_arrays(arrays,model,cfg['eval']['window_frames'],cfg['eval']['stride_frames'],cfg['data']['quality_threshold'])
            assert result['status']==row['status']
            if result['status']=='ok':
                error=float(np.max(np.abs(result['embedding']-vectors[row['row_index']])))
                if kind=='full':maximum_full=max(maximum_full,error)
                else:maximum_dedicated=max(maximum_dedicated,error)
        retained+=row['status']=='ok'
    assert maximum_full==maximum_dedicated==0.
    payload=torch.load(directory/'encoder.pt',map_location='cpu',weights_only=False)
    payload.update(expected)
    portable=portable_root/(Path(item['run_dir']).name+'_encoder.pt')
    atomic_checkpoint(portable,payload)
    first=next(r for r in index if r['status']=='ok')
    public=embed_cached(caches[first['video_id']]['cache_path'],portable)
    assert np.array_equal(public['embedding'],vectors[first['row_index']])
    records.append({'run_dir':item['run_dir'],'eval_dir':item['eval_dir'],
                    'attempted':len(index),'retained':retained,
                    'maximum_full_checkpoint_embedding_error':maximum_full,
                    'maximum_dedicated_encoder_embedding_error':maximum_dedicated,
                    'embeddings_sha256':sha256(directory/'embeddings.npy'),
                    'portable_encoder':str(portable),'portable_encoder_sha256':sha256(portable),
                    'public_portable_api_error':0.,'preprocess_fingerprint':expected['preprocess_fingerprint']})
atomic_json('artifacts/final_inference_verification.json',{
    'status':'passed','current_source_sha256':code_hash(ROOT),
    'frozen_training_source_sha256':json.loads(Path('artifacts/code_freeze_pilot.json').read_text())['code_sha256'],
    'exports_checked':len(records),'results':records,
    'scope':'Every scored learned dev/test export compared on all attempted cached videos. Full and dedicated inference use encoder only; no decoder weights are required or evaluated. Compatible vectors/statuses unchanged; stricter provenance rejection covered by tests.'})
print('Every scored learned export reproduced exactly with final full/dedicated inference; portable encoders saved.')
