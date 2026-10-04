"""Direct native author CUDA function parity, preserving failed cross-device gate."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'method/frozen_probe'));sys.path.insert(0,str(ROOT))
from cache import file_sha
from gpu_accounting import inspect_effective_budget
from resources import gpu_preflight,scratch_preflight
from run_context import RunContext


def main():
    parser=argparse.ArgumentParser()
    for name in ('config','output-dir'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--seed',required=True,type=int);parser.add_argument('--run-id',required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    with RunContext(args.output_dir,args.run_id,args.config,args.seed) as run:
        for name,digest in cfg['input_hashes'].items():
            if file_sha(REPO/name)!=digest:raise ValueError('Bound native/source/input changed')
        producer=REPO/cfg['input_run'];inputs=json.loads((producer/'input_parity.json').read_text())
        if not inputs['all_exact_parity'] or json.loads((producer/'status.json').read_text())['status']!='completed':
            raise ValueError('All-frame completed input parity required')
        ram=scratch_preflight('/dev/shm',cfg['max_payload_bytes'])
        store=Path('/dev/shm')/('shubert_author_gpu_'+args.run_id);store.mkdir(exist_ok=False)
        for sample in inputs['samples']:
            if sample['split'] not in ('train','val') or file_sha(sample['input_path'])!=sample['source_sha256']:
                raise ValueError('Fixed train/dev source identity changed')
            for media in sample['media']:
                if file_sha(media['adapted_path'])!=media['adapted_sha256']:raise ValueError('Native encoded input changed')
            for path,sha in [('body_adapted_path','body_adapted_sha256'),('metadata_path','metadata_sha256')]:
                if file_sha(sample[path])!=sample[sha]:raise ValueError('Body/metadata changed')
        budget=inspect_effective_budget(cfg);hardware=gpu_preflight()
        physical=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],check=True,capture_output=True,text=True,timeout=10).stdout.splitlines()
        if physical!=[cfg['granted_uuid']] or hardware['uuid']!=cfg['granted_uuid']:raise ValueError('Unique granted physical GPU required')
        run.start_gpu(budget)
        import numpy as np
        import torch
        if torch.cuda.device_count()!=1:raise ValueError('Preserve one visible granted GPU')
        torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(8192/hardware['total_mib'],0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.set_num_threads(1);torch.manual_seed(args.seed)
        sys.path.insert(0,str(REPO/'third_party/SHuBERT/fairseq'))
        from examples.shubert.models.shubert import SHubertConfig,SHubertModel
        spec=importlib.util.spec_from_file_location('author_native_cuda',REPO/cfg['author_source'])
        author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
        prepared=[]
        for i,sample in enumerate(inputs['samples']):
            data={name:value for name,value in np.load(cfg['DINO_inputs'][i],allow_pickle=False).items()}
            data['body_posture']=np.load(sample['body_adapted_path'],allow_pickle=False).astype(np.float32)
            folder=store/str(i);folder.mkdir()
            for name,width in [('face',384),('left_hand',384),('right_hand',384),('body_posture',14)]:
                if data[name].shape!=(sample['frames'],width) or data[name].dtype!=np.float32 or not np.isfinite(data[name]).all():
                    raise ValueError('Native feature input contract failed')
                np.save(folder/(name+'.npy'),data[name])
            prepared.append((folder,data))
        # Author loader is exercised literally, but its permissive strict=False is
        # checked against all checkpoint keys and values before using any output.
        native=author.load_model(str(REPO/cfg['main_checkpoint']))
        checkpoint=torch.load(REPO/cfg['main_checkpoint'],map_location='cpu',weights_only=False)['model']
        state=native.state_dict()
        if set(state)!=set(checkpoint) or any(not torch.equal(state[k].cpu(),v) for k,v in checkpoint.items()):
            raise ValueError('Author permissive loader did not load every published tensor exactly')
        del state,checkpoint
        for folder,data in prepared:
            paths=[str(folder/(name+'.npy')) for name in ('face','left_hand','right_hand','body_posture')]
            features=author.process_sample(native,*paths)
            np.save(folder/'native_author_layers.npy',features)
        del native
        own=SHubertModel(SHubertConfig())
        checkpoint=torch.load(REPO/cfg['main_checkpoint'],map_location='cpu',weights_only=False)['model']
        own.load_state_dict(checkpoint,strict=True);del checkpoint
        own.cuda().eval()
        report=dict(run_id=args.run_id,protocol=cfg['protocol'],hardware=hardware,RAM=ram,samples=[],
            comparison='separately loaded models, direct native author CUDA load_model/process_sample vs own assembly',
            required='bitwise equality all 12 author layer-last arrays, all four full contexts',
            original_CPU_GPU_gate_pass=False,original_tolerance_unchanged=True,
            precision='FP32/TF32off',scientific_pilot_authorized=False,PAPER_READY=False,
            scope='Author CUDA pipeline compatibility only; not restored CPU/GPU parity, natural failures, task quality or paper reproduction')
        run.save('author_gpu_parity.json',report)
        saved=np.load(cfg['first_context_comparison'],allow_pickle=False)
        for i,(sample,(folder,data)) in enumerate(zip(inputs['samples'],prepared)):
            source={name:torch.from_numpy(value).cuda() for name,value in data.items()}
            source.update({f'label_{name}':torch.zeros(sample['frames'],1,device='cuda') for name in data})
            with torch.no_grad():result=own.extract_features([source],padding_mask=None,kmeans_labels=None,mask=False)
            layers=np.stack([row[-1].squeeze(1).cpu().numpy() for row in result['layer_results']]);final=result['x'].cpu().numpy()
            reference=np.load(folder/'native_author_layers.npy',allow_pickle=False)
            equal=bool(np.array_equal(layers,reference))
            if layers.shape!=(12,sample['frames'],768) or not np.isfinite(layers).all() or not np.isfinite(final).all():
                raise ValueError('Full context finite/shape failed')
            metadata=json.loads(Path(sample['metadata_path']).read_text())
            if len(metadata)!=sample['frames']:raise ValueError('Metadata length changed')
            target=folder/'own_features.npz'
            np.savez(target,**data,author_layer_last=layers,final_x=final,
                observed_mask=np.asarray([[row['streams'][name]['observed'] for name in data] for row in metadata],dtype=bool),
                pts_start_s=np.asarray([row['pts_start_s'] for row in metadata]),same_detection=np.asarray([row['same_detection'] for row in metadata],dtype=bool))
            entry=dict(sample_id=sample['sample_id'],split=sample['split'],frames=sample['frames'],
                native_parity_bitwise=equal,max_absolute_error=float(np.max(np.abs(layers-reference))),
                layer_shape=list(layers.shape),final_x_shape=list(final.shape),feature_path=str(target),feature_sha256=file_sha(target),
                native_reference_path=str(folder/'native_author_layers.npy'),native_reference_sha256=file_sha(folder/'native_author_layers.npy'),
                feature_bytes=target.stat().st_size,source_sha256=sample['source_sha256'],metadata_sha256=sample['metadata_sha256'],
                final_x_author_function_reference=False,context_mode='full-clip bidirectional offline')
            if i==0:entry['same_as_saved_failed_run_GPU']=bool(np.array_equal(layers,saved['gpu_author_layer_last']))
            report['samples'].append(entry);run.save('author_gpu_parity.json',report)
            if not equal:raise ValueError('Native author GPU parity failed; no comparison relaxation')
            if sum(p.stat().st_size for p in store.rglob('*') if p.is_file())>cfg['max_payload_bytes']:raise ValueError('RAM payload cap exceeded')
            del result,source
        torch.cuda.synchronize()
        report.update(completed=True,all_bitwise_native_parity=True,total_frames=sum(s['frames'] for s in report['samples']),
            payload_bytes=sum(p.stat().st_size for p in store.rglob('*') if p.is_file()),
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
            author_loader_tensor_coverage=226)
        run.save('author_gpu_parity.json',report);run.status['output_hashes']={'author_gpu_parity.json':file_sha(run.output/'author_gpu_parity.json')}


if __name__=='__main__':main()
