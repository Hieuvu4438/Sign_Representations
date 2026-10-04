"""CPU-only strict SHuBERT coverage and synthetic interface, after asset equality."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import resource
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'method/frozen_probe'))
sys.path.insert(0,str(ROOT))
from cache import file_sha
from run_context import RunContext


def main():
    parser=argparse.ArgumentParser()
    for name in ('config','output-dir'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--seed',required=True,type=int)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    with RunContext(args.output_dir,args.run_id,args.config,args.seed) as run:
        producer=REPO/config['asset_run'];status=json.loads((producer/'status.json').read_text())
        equality=json.loads((producer/'asset_equality.json').read_text())
        if (status['status']!='completed' or not equality['all_equal']
                or file_sha(producer/'asset_equality.json')!=status['output_hashes']['asset_equality.json']):
            raise ValueError('Published author asset equality required')
        for name,digest in config['upstream_hashes'].items():
            if file_sha(REPO/name)!=digest:raise ValueError('Pinned model/reference source changed')
        sys.path.insert(0,str(REPO/'third_party/SHuBERT/fairseq'))
        import torch
        import numpy as np
        from examples.shubert.models.shubert import SHubertConfig,SHubertModel
        torch.set_num_threads(1);torch.manual_seed(args.seed)
        if torch.cuda.is_initialized():raise ValueError('This audit must not allocate CUDA')
        dimensions=('encoder_layers','encoder_embed_dim','encoder_ffn_embed_dim','channels_embed_dim',
                    'channels_pose_embed_dim','intermediate_dim','codebook_size')
        default=SHubertConfig();model=SHubertModel(default).cpu().eval()
        checkpoint=REPO/config['checkpoint']
        if file_sha(checkpoint)!=config['checkpoint_sha256']:raise ValueError('Published checkpoint identity changed')
        # Trusted author bytes verified above; training container may include OmegaConf.
        container=torch.load(checkpoint,map_location='cpu',weights_only=False)
        state=container['model'] if 'model' in container else container
        cfg=container.get('cfg',{})
        if hasattr(cfg,'get'):cfg=cfg.get('model',{})
        architecture={k:cfg.get(k) for k in dimensions} if hasattr(cfg,'get') else {}
        expected=model.state_dict();missing=sorted(set(expected)-set(state));unexpected=sorted(set(state)-set(expected))
        mismatches={k:dict(expected=list(expected[k].shape),actual=list(state[k].shape))
                    for k in set(expected)&set(state) if expected[k].shape!=state[k].shape}
        report=dict(run_id=args.run_id,checkpoint_sha256=config['checkpoint_sha256'],
                    checkpoint_top_keys=sorted(container),state_keys=len(state),default_model_keys=len(expected),
                    default_architecture={k:getattr(default,k) for k in dimensions},checkpoint_architecture=architecture,
                    missing_keys=missing,unexpected_keys=unexpected,shape_mismatches=mismatches,
                    strict_coverage_pass=not(missing or unexpected or mismatches),
                    python=sys.version,packages={p:importlib.metadata.version(p) for p in ('torch','numpy','omegaconf')},
                    upstream_hashes=config['upstream_hashes'],asset_equality_sha256=file_sha(producer/'asset_equality.json'),
                    scope='CPU strict-load/software interface only; no real video stream extraction or task quality',GPU_hours=0)
        run.save('load_audit.json',report)
        if not report['strict_coverage_pass']:raise ValueError('Default author model differs from published checkpoint; no partial load')
        model.load_state_dict(state,strict=True)
        del container,state,expected
        source={name:torch.randn(config['synthetic_frames'],dim) for name,dim in
                (('face',384),('left_hand',384),('right_hand',384),('body_posture',14))}
        source.update({f'label_{name}':torch.zeros(config['synthetic_frames'],1)
                       for name in ('face','left_hand','right_hand','body_posture')})
        with torch.no_grad():output=model.extract_features([source],padding_mask=None,kmeans_labels=None,mask=False)
        layers=[row[-1].squeeze(1).cpu().numpy() for row in output['layer_results']]
        pooled=np.stack(layers)
        if pooled.shape!=(default.encoder_layers,config['synthetic_frames'],default.encoder_embed_dim) or not np.isfinite(pooled).all():
            raise ValueError('Author layer[-1] synthetic feature contract invalid')
        np.savez(run.output/'synthetic_reference.npz',layer_last=pooled,final_x=output['x'].cpu().numpy(),
                 **{f'input_{k}':v.numpy() for k,v in source.items()})
        dino=[]
        for asset in equality['assets'][1:]:
            path=REPO/asset['local_path']
            if file_sha(path)!=asset['local_sha256']:raise ValueError('Published DINO weights changed')
            data=torch.load(path,map_location='cpu',weights_only=False)
            teacher=data.get('teacher',{})
            selected={k:list(v.shape) for k,v in teacher.items()
                      if any(n in k for n in ('pos_embed','register_tokens','cls_token','patch_embed.proj.weight'))}
            dino.append(dict(path=asset['local_path'],container_keys=sorted(data),teacher_key_count=len(teacher),
                             architecture_tensor_shapes=selected,strict_DINO_model_load='not yet performed; official architecture dependency pending'))
            del data,teacher
        snapshot=run.output/'upstream_source_snapshot';snapshot.mkdir()
        for name in config['upstream_hashes']:
            dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,dest)
        report.update(strict_load_completed=True,eval_mode=not model.training,synthetic_shape=list(pooled.shape),
                      final_x_shape=list(output['x'].shape),synthetic_finite=True,
                      output_recipe='author shubert_inference.py layer_results row[-1].squeeze(1); not interchangeable with final_x',
                      DINO_containers=dino,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      CUDA_initialized=torch.cuda.is_initialized())
        if report['CUDA_initialized']:raise ValueError('CPU-only audit unexpectedly initialized CUDA')
        run.save('load_audit.json',report)
        run.status['output_hashes']={name:file_sha(run.output/name) for name in ('load_audit.json','synthetic_reference.npz')}


if __name__=='__main__':main()
