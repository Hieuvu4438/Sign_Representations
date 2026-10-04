"""Four fixed real train/dev streams with strict models and CPU/GPU references."""
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
            if file_sha(REPO/name)!=digest:raise ValueError('Frozen input/model/reference changed')
        producer=REPO/cfg['input_run'];producer_status=json.loads((producer/'status.json').read_text())
        inputs=json.loads((producer/'input_parity.json').read_text())
        if producer_status['status']!='completed' or not inputs['all_exact_parity']:raise ValueError('Completed all-frame native parity required')
        architecture=REPO/cfg['dino_source_run'];source=json.loads((architecture/'source_manifest.json').read_text())
        for member in source['files']:
            if file_sha(architecture/'official_source'/member['path'])!=member['sha256']:raise ValueError('Pinned official architecture changed')
        for sample in inputs['samples']:
            if sample['split'] not in ('train','val'):raise ValueError('Only fixed train/dev sources allowed')
            if file_sha(sample['input_path'])!=sample['source_sha256']:raise ValueError('Real source identity changed')
            for media in sample['media']:
                if file_sha(media['adapted_path'])!=media['adapted_sha256']:raise ValueError('Exact-parity encoded crop changed')
            if file_sha(sample['body_adapted_path'])!=sample['body_adapted_sha256'] or file_sha(sample['metadata_path'])!=sample['metadata_sha256']:
                raise ValueError('Aligned body/mask metadata changed')
        ram=scratch_preflight('/dev/shm',cfg['max_payload_bytes'])
        store=Path('/dev/shm')/('shubert_features_'+args.run_id);store.mkdir(exist_ok=False)
        budget=inspect_effective_budget(cfg);hardware=gpu_preflight()
        physical=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],check=True,capture_output=True,text=True,timeout=10).stdout.splitlines()
        if physical!=[cfg['granted_uuid']] or hardware['uuid']!=cfg['granted_uuid']:raise ValueError('Unique physical granted GPU required')
        run.start_gpu(budget)
        import numpy as np
        import torch
        from torchvision import transforms
        from PIL import Image
        import decord
        if torch.cuda.device_count()!=1:raise ValueError('One scheduler-visible granted device required; visibility preserved')
        torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(8192/hardware['total_mib'],0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.set_num_threads(1);torch.manual_seed(args.seed)
        sys.path.insert(0,str(architecture/'official_source'));sys.path.insert(0,str(REPO/'third_party/SHuBERT/fairseq'))
        from dinov2.hub.backbones import dinov2_vits14_reg
        from examples.shubert.models.shubert import SHubertConfig,SHubertModel
        spec=importlib.util.spec_from_file_location('author_stream_preprocessing',REPO/cfg['author_dino_source'])
        author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        report=dict(run_id=args.run_id,protocol=cfg['protocol'],hardware=hardware,RAM=ram,artifact_root=str(store),
                    samples=[],atol=cfg['atol'],rtol=cfg['rtol'],dtype='float32',TF32=False,
                    scope='Four fixed native train/dev data smoke only; no task head/metric, corpus/failure-rate or paper parity')
        run.save('feature_smoke.json',report)
        arrays=[{} for _ in inputs['samples']]
        dino_parity=[[] for _ in inputs['samples']]
        for kind,names in [('face',['face']),('hand',['left_hand','right_hand'])]:
            model=dinov2_vits14_reg(pretrained=False);model.pos_embed=torch.nn.Parameter(torch.zeros(1,257,384))
            teacher=torch.load(REPO/cfg['dino_checkpoints'][kind],map_location='cpu',weights_only=False)['teacher']
            model.load_state_dict({k.removeprefix('backbone.'):v for k,v in teacher.items() if 'dino_head' not in k},strict=True)
            del teacher
            cpu_reference=model.cpu().eval()
            for index,sample in enumerate(inputs['samples']):
                media={m['stream']:m for m in sample['media']}
                for name in names:
                    reader=decord.VideoReader(media[name]['adapted_path'],ctx=decord.cpu(0),width=224,height=224,num_threads=1)
                    if len(reader)!=sample['frames']:raise ValueError('Encoded real stream length changed')
                    frame=reader[0].asnumpy()
                    own=transform(Image.fromarray(frame));native=author.preprocess_frame(frame)
                    if not torch.equal(own,native):raise ValueError('Real author preprocessing parity differs')
                    with torch.no_grad():reference=cpu_reference(native.unsqueeze(0)).numpy()
                    # Same weights CPU/GPU; no partial reload, selection or fit.
                    model.cuda().eval();features=[]
                    for start in range(0,len(reader),cfg['batch_size']):
                        pixels=reader.get_batch(range(start,min(start+cfg['batch_size'],len(reader)))).asnumpy()
                        tensors=torch.stack([author.preprocess_frame(p) for p in pixels]).cuda()
                        with torch.no_grad():features.append(model(tensors).cpu().numpy())
                    output=np.concatenate(features)
                    if output.shape!=(sample['frames'],384) or not np.isfinite(output).all():raise ValueError('Real DINO output shape/finite differs')
                    equal=bool(np.allclose(output[:1],reference,atol=cfg['atol'],rtol=cfg['rtol']))
                    dino_parity[index].append(dict(stream=name,first_frame_CPU_GPU_parity=equal,
                        max_absolute_error=float(np.max(np.abs(output[:1]-reference)))))
                    if not equal:raise ValueError('Real first-frame DINO numerical parity failed; no tolerance widening')
                    arrays[index][name]=output
                    model.cpu();cpu_reference=model
            del cpu_reference,model
        model=SHubertModel(SHubertConfig());container=torch.load(REPO/cfg['main_checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict(container['model'],strict=True);del container
        model.cpu().eval()
        if cfg.get('diagnostic_mode',False):
            for index,sample in enumerate(inputs['samples']):
                np.savez(store/('DINO_inputs_'+str(index)+'.npz'),**arrays[index])
            run.save('DINO_stage.json',dict(samples=dino_parity,
                scope='All fixed first-frame DINO comparisons; no accepted full-context SHuBERT features'))
        for index,sample in enumerate(inputs['samples']):
            arrays[index]['body_posture']=np.load(sample['body_adapted_path'],allow_pickle=False).astype(np.float32)
            if arrays[index]['body_posture'].shape!=(sample['frames'],14):raise ValueError('Aligned body shape differs')
            source_input={name:torch.from_numpy(value) for name,value in arrays[index].items()}
            source_input.update({f'label_{name}':torch.zeros(sample['frames'],1) for name in ('face','left_hand','right_hand','body_posture')})
            with torch.no_grad():cpu=model.extract_features([source_input],padding_mask=None,kmeans_labels=None,mask=False)
            cpu_layers=np.stack([row[-1].squeeze(1).numpy() for row in cpu['layer_results']]);cpu_final=cpu['x'].numpy()
            del cpu
            model.cuda().eval();gpu_input={name:value.cuda() for name,value in source_input.items()}
            with torch.no_grad():gpu=model.extract_features([gpu_input],padding_mask=None,kmeans_labels=None,mask=False)
            gpu_layers=np.stack([row[-1].squeeze(1).cpu().numpy() for row in gpu['layer_results']]);gpu_final=gpu['x'].cpu().numpy()
            if gpu_layers.shape!=(12,sample['frames'],768) or not np.isfinite(gpu_layers).all():raise ValueError('Author layer-last real contract failed')
            parity=bool(np.allclose(gpu_layers,cpu_layers,atol=cfg['atol'],rtol=cfg['rtol']) and np.allclose(gpu_final,cpu_final,atol=cfg['atol'],rtol=cfg['rtol']))
            if cfg.get('diagnostic_mode',False):
                comparisons=[]
                for layer in range(12):
                    difference=np.abs(gpu_layers[layer]-cpu_layers[layer])
                    bound=cfg['atol']+cfg['rtol']*np.abs(cpu_layers[layer])
                    bad=difference>bound
                    comparisons.append(dict(layer=layer+1,failed_elements=int(bad.sum()),total_elements=int(bad.size),
                        max_absolute_error=float(difference.max()),max_tolerance_ratio=float((difference/bound).max()),
                        max_reference_magnitude=float(np.abs(cpu_layers[layer]).max()),
                        largest_violation_index=list(map(int,np.unravel_index(np.argmax(difference/bound),bad.shape)))))
                target=store/'first_context_comparison.npz'
                np.savez(target,cpu_author_layer_last=cpu_layers,gpu_author_layer_last=gpu_layers,
                         cpu_final_x=cpu_final,gpu_final_x=gpu_final,**arrays[index])
                diagnostic=dict(sample_id=sample['sample_id'],frames=sample['frames'],DINO_parity=dino_parity,
                    layers=comparisons,full_context_pass=parity,final_x_pass=bool(np.allclose(gpu_final,cpu_final,atol=cfg['atol'],rtol=cfg['rtol'])),
                    final_x_max_absolute_error=float(np.abs(gpu_final-cpu_final).max()),atol=cfg['atol'],rtol=cfg['rtol'],
                    comparison_path=str(target),comparison_sha256=file_sha(target),payload_bytes=sum(p.stat().st_size for p in store.iterdir()),
                    scientific_pilot_authorized=False,scope='First failed clip forensic arrays only; original smoke gate unchanged')
                run.save('numerical_diagnostic.json',diagnostic)
                if diagnostic['payload_bytes']>cfg['max_payload_bytes']:raise ValueError('Diagnostic RAM cap exceeded')
                run.status['output_hashes']={name:file_sha(run.output/name) for name in
                    ('feature_smoke.json','DINO_stage.json','numerical_diagnostic.json')}
                return
            if not parity:raise ValueError('Real full-context CPU/GPU parity failed; no tolerance widening')
            metadata=json.loads(Path(sample['metadata_path']).read_text())
            masks=np.asarray([[row['streams'][name]['observed'] for name in ('face','left_hand','right_hand','body_posture')] for row in metadata],dtype=bool)
            target=store/(str(index)+'.npz')
            np.savez(target,**arrays[index],author_layer_last=gpu_layers,final_x=gpu_final,
                     observed_mask=masks,pts_start_s=np.asarray([row['pts_start_s'] for row in metadata]),
                     same_detection=np.asarray([row['same_detection'] for row in metadata],dtype=bool))
            blank={}
            for name in ('face','left_hand','right_hand'):
                positions=[i for i,row in enumerate(metadata) if row['streams'][name]['fill_kind']=='initial_blank']
                if positions:
                    norms=np.linalg.norm(arrays[index][name][positions],axis=1)
                    blank[name]=dict(frames=len(positions),minimum_feature_L2=float(norms.min()),maximum_feature_L2=float(norms.max()),
                                    semantics='Black pixel fill does not imply a zero feature or observed detector')
            report['samples'].append(dict(sample_id=sample['sample_id'],split=sample['split'],frames=sample['frames'],
                source_sha256=sample['source_sha256'],input_parity_sha256=file_sha(producer/'input_parity.json'),
                feature_path=str(target),feature_sha256=file_sha(target),feature_bytes=target.stat().st_size,
                spatial_shapes={name:list(value.shape) for name,value in arrays[index].items()},
                author_layer_last_shape=list(gpu_layers.shape),final_x_shape=list(gpu_final.shape),
                DINO_parity=dino_parity[index],full_context_CPU_GPU_parity=parity,
                full_context_max_absolute_error=float(np.max(np.abs(gpu_layers-cpu_layers))),
                observed_counts=dict(zip(('face','left_hand','right_hand','body_posture'),masks.sum(axis=0).astype(int).tolist())),
                same_detection_frames=sum(row['same_detection'] for row in metadata),initial_blank_features=blank,
                context_mode='full-clip bidirectional offline',layer_recipe='author layer_results[-1] separate from final_x',
                correct_anatomy=False,natural_failure_labels=False))
            run.save('feature_smoke.json',report)
            del gpu,gpu_input,source_input,cpu_layers,cpu_final,gpu_layers,gpu_final
            model.cpu()
            if sum(p.stat().st_size for p in store.iterdir())>cfg['max_payload_bytes']:raise ValueError('RAM feature cap exceeded')
        torch.cuda.synchronize()
        report.update(completed=True,total_frames=sum(s['frames'] for s in report['samples']),all_parity=True,
            payload_bytes=sum(p.stat().st_size for p in store.iterdir()),peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),scientific_pilot_authorized=False)
        run.save('feature_smoke.json',report)
        run.status['output_hashes']={'feature_smoke.json':file_sha(run.output/'feature_smoke.json')}


if __name__=='__main__':main()
