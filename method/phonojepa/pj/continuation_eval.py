"""Original raw stride2/_grp dictionary protocol, with resumable feature shards."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from pj.frozen_dictionary import Prepared, encode_video, worker_init
from pj.extract import Extractor
from oj.wlasl_retrieval import score_retrieval
from pj.continuation_support import atomic_json, sha256


def report_metrics(gallery, query, gallery_rows, query_rows):
    gy = [r['label'] for r in gallery_rows]; qy = np.array([r['label'] for r in query_rows])
    primary, ranks = score_retrieval(gallery,query,gy,qy,len(set(gy)))
    metrics = {'DCG':primary['DCG'],'MRR':float(np.mean(1/ranks)),
               **{f'Rec@{k}':float(np.mean(ranks<=k)) for k in [1,5,10]},
               'macro_Rec@1':float(np.mean([np.mean(ranks[qy==c]<=1) for c in np.unique(qy)])),
               'queries':len(query_rows),'gallery':len(gallery_rows)}
    return metrics,ranks


def load_original_gallery(job):
    root = Path(job['baseline_dictionary']); rows = job['manifests']['train']
    vectors=[]; ids=[]
    for path in sorted((root/'train').glob('*.npz')):
        with np.load(path,allow_pickle=False) as data:
            vectors.append(data['_grp']); ids.extend(data['ids'].tolist())
    if ids != [r['sample_id'] for r in rows]: raise ValueError('Baseline cached gallery IDs/order changed')
    return np.concatenate(vectors)


@torch.inference_mode()
def extract_features(extractor, rows, split, job, cache_root, checkpoint_sha, stopped, progress):
    folder = Path(cache_root)/split
    identity = {'checkpoint_sha256':checkpoint_sha,'rows_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
                'protocol':'native_PLctx_grp_raw_window64_stride2_mean_before_L2_v1'}
    if folder.exists() and ((folder/'identity.json').exists() and json.loads((folder/'identity.json').read_text()) != identity):
        shutil.rmtree(folder)
    folder.mkdir(parents=True,exist_ok=True); atomic_json(folder/'identity.json',identity)
    metadata = json.loads(Path(job['video_metadata']).read_text())
    matrices=[]; nwin=0; start_time=time.time(); offset=0
    # Resume a verified contiguous prefix, then keep one spawn DataLoader alive
    # through every remaining shard. No fork after CUDA initialization.
    for offset in range(0,len(rows),256):
        subset=rows[offset:offset+256];shard=folder/f'{offset:07d}.npz';receipt=shard.with_suffix('.json')
        reuse=False
        if shard.exists() and receipt.exists():
            try:
                saved=json.loads(receipt.read_text())
                if saved['sha256']==sha256(shard):
                    with np.load(shard,allow_pickle=False) as data:
                        if data['ids'].tolist()==[r['sample_id'] for r in subset] and np.isfinite(data['_grp']).all():
                            matrices.append(data['_grp'].copy());nwin+=int(data['nwin'].sum());reuse=True
            except (OSError,ValueError,EOFError):pass
        if not reuse:break
        progress({'activity':'extract_features','split':split,'completed':min(offset+256,len(rows)),
                  'total':len(rows),'windows':nwin,'elapsed_seconds':time.time()-start_time})
    else:
        return np.concatenate(matrices)
    dataset=Prepared(rows[offset:],Path(job['dataset_root'])/'keypoints',2,metadata)
    workers=job['evaluation_workers']
    loader=torch.utils.data.DataLoader(dataset,batch_size=None,num_workers=workers,worker_init_fn=worker_init,
        prefetch_factor=2 if workers else None,multiprocessing_context='spawn' if workers else None)
    values=[];counts=[];shard_start=offset
    for index,prepared in enumerate(loader,start=offset):
        if stopped():raise InterruptedError('Dictionary feature extraction paused')
        prepared={k:v.numpy() if torch.is_tensor(v) else v for k,v in prepared.items()}
        pooled,count=encode_video(extractor,prepared,job['window_batch'])
        values.append(pooled['_grp']);counts.append(count)
        if len(values)==256 or index+1==len(rows):
            shard=folder/f'{shard_start:07d}.npz';receipt=shard.with_suffix('.json');temporary=shard.with_suffix('.tmp.npz')
            matrix=np.stack(values);count_array=np.array(counts)
            np.savez(temporary,_grp=matrix,nwin=count_array,ids=np.array([r['sample_id'] for r in rows[shard_start:index+1]]))
            os.replace(temporary,shard);atomic_json(receipt,{'sha256':sha256(shard),'clips':len(matrix)})
            matrices.append(matrix);nwin+=int(count_array.sum());values=[];counts=[];shard_start=index+1
            progress({'activity':'extract_features','split':split,'completed':index+1,
                      'total':len(rows),'windows':nwin,'elapsed_seconds':time.time()-start_time})
    return np.concatenate(matrices)


def evaluate_checkpoint(checkpoint, job, directory, split='val', encoder=None, stopped=lambda:False, progress=lambda x:None,
                        reuse_baseline_gallery=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    digest=sha256(checkpoint);output=directory/'metrics.json'
    if output.exists():
        result=json.loads(output.read_text())
        if result['checkpoint_sha256']!=digest:raise ValueError('Evaluation checkpoint identity changed')
        return result
    if encoder is None:
        extractor=Extractor('model',checkpoint,'teacher',job.get('device','cuda'))
    else:
        extractor=Extractor.__new__(Extractor);extractor.kind='model';extractor.dev=str(next(encoder.parameters()).device)
        extractor.hands_only=False;extractor.enc=encoder.eval()
    cache=directory.parent/'feature_cache'
    gallery=load_original_gallery(job) if reuse_baseline_gallery else extract_features(extractor,job['manifests']['train'],'train',job,cache,digest,stopped,progress)
    queries=extract_features(extractor,job['manifests'][split],split,job,cache,digest,stopped,progress)
    if stopped():raise InterruptedError('Dictionary scoring paused')
    progress({'activity':'dictionary_scoring','split':split,'completed':len(queries),'total':len(queries)})
    metrics,ranks=report_metrics(gallery,queries,job['manifests']['train'],job['manifests'][split])
    np.save(directory/'ranks.npy',ranks)
    result={'status':'complete','checkpoint_sha256':digest,'checkpoint':str(checkpoint),'split':split,'metrics':metrics,
            'protocol':'official_train_gallery/raw_window64_stride2_grp_mean_L2_class_max_pessimistic_ties',
            'gallery_zero_vectors':int((np.linalg.norm(gallery,axis=1)<=1e-12).sum()),
            'query_zero_vectors':int((np.linalg.norm(queries,axis=1)<=1e-12).sum()),
            'failure_policy':'historical PLctx scorer: retain zero vectors and finite pessimistic tie ranks; all queries retained',
            'finished_unix':time.time()}
    atomic_json(output,result);return result
