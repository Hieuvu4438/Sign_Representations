"""Check validation-only changes against immutable real pilot vectors/caches."""
import json
from pathlib import Path
import numpy as np
import torch
import signrel
from signrel.config import ROOT, load_config
from signrel.utils import read_jsonl, atomic_json, sha256, code_hash
from signrel.eval.embed import embed_cached, restore_encoder
from signrel.train.loop import validate_training_inputs
from signrel.data.provenance import inference_expectations

assert 'engineering_validation' in str(ROOT), ROOT
torch.set_num_threads(4)
cfg=load_config('configs/pilot_a1.yaml').values
dataset,expected=validate_training_inputs(cfg)
assert len(dataset.caches)==47
assert expected['preprocess_fingerprint']=='e8b618580dd6913574e24c053128c9e2e314b78e8b95d388f74c5e23f19865e1'
indices=read_jsonl('results/pilot_a1_seed17_dev/index.jsonl')
vectors=np.load('results/pilot_a1_seed17_dev/embeddings.npy',allow_pickle=False)
caches={r['video_id']:r for r in read_jsonl('data/cache/eval_cache_manifest.jsonl')}
chosen=indices[0]
path=caches[chosen['video_id']]['cache_path']
records=[]
for checkpoint in ('runs/pilot_a1_seed17/checkpoints/last.pt','results/pilot_a1_seed17_dev/encoder.pt'):
    result=embed_cached(path,checkpoint)
    error=float(np.max(np.abs(result['embedding']-vectors[chosen['row_index']])))
    assert error==0
    records.append({'checkpoint':checkpoint,'maximum_absolute_error_vs_saved_export':error,'dimension':len(result['embedding'])})
legacy='results/pilot_a1_seed17_dev/encoder.pt'
_,train_cfg,state=restore_encoder(legacy)
state.update(inference_expectations(state,train_cfg))
portable=Path('artifacts/engineering_validation/portable_encoder.pt')
torch.save(state,portable)
# A new inference artifact contains its own teacher/preprocessing binding and
# does not need the source lock to remain at a machine-specific path.
state['config_resolved']['paths']['sources_lock']='nonexistent-portable-asset.json'
torch.save(state,portable)
result=embed_cached(path,portable)
assert np.array_equal(result['embedding'],vectors[chosen['row_index']])
atomic_json('artifacts/staged_inference_verification.json',{
    'status':'passed','staged_source_sha256':code_hash(ROOT),
    'frozen_training_source_sha256':json.loads(Path('artifacts/code_freeze_pilot.json').read_text())['code_sha256'],
    'real_train_caches_checked':47,'preprocess_fingerprint':expected['preprocess_fingerprint'],
    'saved_export_sha256':sha256('results/pilot_a1_seed17_dev/embeddings.npy'),
    'results':records,'portable_without_source_lock':True,
    'dedicated_encoder_has_no_heads':not hasattr(restore_encoder(portable)[0],'heads')})
print('Staged validation accepts all47 real training caches; full/legacy/new portable inference equals immutable pilot export.')
