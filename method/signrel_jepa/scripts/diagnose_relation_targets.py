"""Test target predictability and sensitivity to mismatched relation endpoints."""
import argparse
import json
import sys
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'signrel/config.py').exists())
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from signrel.data.manifest import sha256
from signrel.data.video_dataset import SSLVideoDataset
from signrel.models.backbone_adapter import make_backbone
from signrel.models.region_pooler import RegionPooler,token_membership,DEFAULT_EDGES,ARTICULATORS
from signrel.models.relation_projector import RelationProjector


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--limit',type=int,default=95)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out',required=True)
    args = p.parse_args()
    if args.limit < 2:
        p.error('At least two clips are needed for cross-clip endpoint shuffling')
    state = torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    cfg = state['config']
    bc = dict(cfg['model']['backbone'])
    bc.pop('checkpoint',None)
    bc['checkpointing'] = False
    encoder = make_backbone(bc).to(args.device).eval()
    encoder.load_state_dict(state['teacher_encoder'],strict=True)
    projector = RelationProjector(encoder.dim,cfg['model']['relation_dim'],cfg['model'].get('projector','concat_diff_product')).eval()
    projector.load_state_dict(state['teacher_relation_projector'],strict=True)
    edges = tuple(tuple(ARTICULATORS.index(v) if isinstance(v,str) else v for v in e) for e in cfg['model'].get('relation_edges',DEFAULT_EDGES))
    u,v = zip(*edges)
    dc = cfg['data']
    ds = SSLVideoDataset(args.manifest,num_frames=dc['num_frames'],sampling_stride=dc.get('sampling_stride',2),crop_size=dc['crop_size'],augment=False)
    pooler = RegionPooler()
    ac = cfg.get('articulators',{})
    nodes,valid,ids = [],[],[]
    with torch.no_grad():
        for index in range(min(args.limit,len(ds))):
            item = ds[index]
            with torch.autocast(torch.device(args.device).type,dtype=torch.bfloat16,enabled=torch.device(args.device).type == 'cuda'):
                output = encoder.forward_tokens(item['video'][None].to(args.device))
            grid = output['grid_shape']
            membership = token_membership(item['articulator_meta']['boxes'][None],item['articulator_meta']['confidence'][None],grid,
                                          ac.get('confidence_threshold',.5),ac.get('membership','center'))
            n,ok = pooler(output['tokens'].float().cpu().reshape(1,*grid,encoder.dim),membership)
            nodes.append(n[0]); valid.append(ok[0]); ids.append(item['sample_id'])
        nodes,valid = torch.stack(nodes),torch.stack(valid)
        relations = projector(nodes[:,:,u],nodes[:,:,v])
        rel_valid = valid[:,:,u]&valid[:,:,v]
        # A cyclic donor permutation guarantees every clip's second endpoint
        # comes from another clip, without labels or accidental fixed points.
        donor = torch.arange(len(nodes)).roll(1)
        shuffled = projector(nodes[:,:,u],nodes[donor][:,:,v])
        both = rel_valid&valid[donor][:,:,v]
        target = F.layer_norm(relations,(relations.shape[-1],))
        shuffled = F.layer_norm(shuffled,(relations.shape[-1],))
        per_edge = {}
        for e,(a,b) in enumerate(edges):
            y = target[:,:,e]
            ok = rel_valid[:,:,e]
            total_count = ok.sum()
            if total_count < 2:
                per_edge[f'{a}_{b}'] = {'status':'insufficient_coverage'}
                continue
            total_sum = (y*ok[...,None]).sum((0,1))
            mean = total_sum/total_count
            residual = y-mean
            variance_energy = float(residual[ok].square().sum())
            total_energy = float(y[ok].square().sum())
            # Train a constant predictor from other clips only; evaluate each
            # held-out clip. This is a predictability diagnostic, not SSL loss.
            counts = total_count-ok.sum(1)
            loco = (total_sum-(y*ok[...,None]).sum(1))/counts.clamp_min(1)[:,None]
            loco = F.layer_norm(loco,(y.shape[-1],))[:,None].expand_as(y)
            loco_valid = ok&(counts[:,None] > 0)
            mask = both[:,:,e]
            raw_cos = F.cosine_similarity(y,shuffled[:,:,e],dim=-1)
            centered_cos = F.cosine_similarity(residual,shuffled[:,:,e]-mean,dim=-1)
            per_edge[f'{a}_{b}'] = {'status':'evaluated','valid_queries':int(total_count),
                'mean_per_dimension_std':float(y[ok].std(0,unbiased=False).mean()),
                'centered_energy_fraction':variance_energy/total_energy if total_energy else None,
                'leave_one_clip_out_constant_L1':float((loco-y).abs()[loco_valid].mean()),
                'leave_one_clip_out_constant_cosine_loss':float((1-F.cosine_similarity(loco,y,dim=-1))[loco_valid].mean()),
                'mismatched_endpoint_queries':int(mask.sum()),
                'mismatched_endpoint_cosine':float(raw_cos[mask].mean()) if mask.any() else None,
                'mismatched_endpoint_centered_cosine':float(centered_cos[mask].mean()) if mask.any() else None}
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    cache = Path('features/target_diagnostics')/f'{out.stem}.npz'
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,nodes=nodes.numpy(),node_valid=valid.numpy(),relations=relations.numpy(),relation_valid=rel_valid.numpy(),sample_ids=np.array(ids))
    report = {'arguments':vars(args),'manifest_sha256':sha256(args.manifest),'checkpoint_sha256':sha256(args.checkpoint),
              'source_sha256':sha256(__file__),'cache':str(cache),'cache_sha256':sha256(cache),'clips':len(nodes),
              'per_edge':per_edge,'target_normalization':'LayerNorm per vector, matching latent objective',
              'protocol':'Fixed center RGB clips and restored EMA teacher/projector; leave-one-clip-out constant by edge; cyclic cross-clip second-endpoint donor',
              'interpretation':'High raw cosine can reflect a shared mean. Centered cosine and residual energy measure sensitivity beyond that mean. Diagnostic losses are not directly comparable to masked, augmented training batches.'}
    out.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'clips':len(nodes),'per_edge':per_edge}))


if __name__ == '__main__':
    main()
