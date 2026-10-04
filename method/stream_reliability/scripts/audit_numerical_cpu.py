"""CPU parameter/input precision promotion on a preserved failed real context."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'method/frozen_probe'));sys.path.insert(0,str(ROOT))
from cache import file_sha
from resources import scratch_preflight
from run_context import RunContext


def main():
    parser=argparse.ArgumentParser()
    for name in ('config','output-dir'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--seed',required=True,type=int);parser.add_argument('--run-id',required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    with RunContext(args.output_dir,args.run_id,args.config,args.seed) as run:
        for name,digest in cfg['input_hashes'].items():
            if file_sha(REPO/name)!=digest:raise ValueError('Bound forensic/source/checkpoint changed')
        diagnostic=json.loads((REPO/cfg['diagnostic_report']).read_text())
        if file_sha(diagnostic['comparison_path'])!=diagnostic['comparison_sha256']:raise ValueError('Saved failed context changed')
        scratch_preflight('/dev/shm',cfg['max_payload_bytes'])
        work=Path('/dev/shm')/('shubert_cpu_precision_'+args.run_id);work.mkdir(exist_ok=False)
        import numpy as np
        import torch
        sys.path.insert(0,str(REPO/'third_party/SHuBERT/fairseq'))
        from examples.shubert.models.shubert import SHubertConfig,SHubertModel
        torch.set_num_threads(1);torch.manual_seed(args.seed)
        if torch.cuda.is_initialized():raise ValueError('CPU-only precision probe initialized CUDA')
        saved=np.load(diagnostic['comparison_path'],allow_pickle=False)
        model=SHubertModel(SHubertConfig());container=torch.load(REPO/cfg['checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict(container['model'],strict=True);del container
        model.cpu().double().eval()
        source={name:torch.from_numpy(saved[name]).double() for name in ('face','left_hand','right_hand','body_posture')}
        source.update({f'label_{name}':torch.zeros(diagnostic['frames'],1,dtype=torch.float64) for name in ('face','left_hand','right_hand','body_posture')})
        with torch.no_grad():output=model.extract_features([source],padding_mask=None,kmeans_labels=None,mask=False)
        promoted=np.stack([row[-1].squeeze(1).numpy() for row in output['layer_results']]);final=output['x'].numpy()
        if promoted.shape!=saved['cpu_author_layer_last'].shape or not np.isfinite(promoted).all():raise ValueError('Promoted CPU diagnostic invalid')
        layers=[];bad_points=[]
        for layer in range(12):
            cpu=saved['cpu_author_layer_last'][layer];gpu=saved['gpu_author_layer_last'][layer]
            bad=np.abs(gpu-cpu)>(diagnostic['atol']+diagnostic['rtol']*np.abs(cpu))
            for t,d in zip(*np.where(bad)):
                bad_points.append(dict(layer=layer+1,frame=int(t),channel=int(d),cpu_FP32=float(cpu[t,d]),gpu_FP32=float(gpu[t,d]),
                    promoted_mixed_precision=float(promoted[layer,t,d]),
                    CPU32_error_to_promoted=float(abs(cpu[t,d]-promoted[layer,t,d])),
                    GPU32_error_to_promoted=float(abs(gpu[t,d]-promoted[layer,t,d]))))
            layers.append(dict(layer=layer+1,original_failed_elements=int(bad.sum()),
                CPU32_vs_promoted_relative_L2=float(np.linalg.norm(cpu-promoted[layer])/np.linalg.norm(promoted[layer])),
                GPU32_vs_promoted_relative_L2=float(np.linalg.norm(gpu-promoted[layer])/np.linalg.norm(promoted[layer])),
                CPU_GPU_relative_L2=float(np.linalg.norm(cpu.astype(np.float64)-gpu)/np.linalg.norm(cpu.astype(np.float64)))))
        target=work/'promoted_cpu_reference.npz';np.savez(target,author_layer_last=promoted,final_x=final)
        if target.stat().st_size>cfg['max_payload_bytes']:raise ValueError('CPU precision payload cap exceeded')
        report=dict(run_id=args.run_id,GPU_hours=0,sample_id=diagnostic['sample_id'],frames=diagnostic['frames'],
            layerwise=layers,failed_coordinates=bad_points,original_full_context_pass=diagnostic['full_context_pass'],
            original_final_x_pass=diagnostic['final_x_pass'],reference_path=str(target),reference_sha256=file_sha(target),reference_bytes=target.stat().st_size,
            precision='FP64 parameter/input promotion; original upstream GELU/softmax FP32 casts retained, mixed-precision diagnostic not an ideal FP64 oracle',
            original_gate_unchanged=True,scientific_pilot_authorized=False,CUDA_initialized=torch.cuda.is_initialized(),
            scope='Numerical sensitivity investigation only; no corrected tolerance, accepted full features or task metric')
        if report['CUDA_initialized']:raise ValueError('CPU-only diagnostic initialized CUDA')
        run.save('cpu_precision_diagnostic.json',report);run.status['output_hashes']={'cpu_precision_diagnostic.json':file_sha(run.output/'cpu_precision_diagnostic.json')}


if __name__=='__main__':main()
