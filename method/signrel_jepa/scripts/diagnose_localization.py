"""Measure relation-target sensitivity and mask feasibility under box jitter."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p/"signrel/config.py").exists())
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from signrel.data.manifest import sha256
from signrel.data.video_dataset import SSLVideoDataset
from signrel.models.backbone_adapter import make_backbone
from signrel.models.region_pooler import RegionPooler,token_membership,ARTICULATORS,DEFAULT_EDGES
from signrel.models.relation_projector import RelationProjector
from signrel.masks.base import SignMaskGenerator
try:
    from signrel.eval.localization import jitter_boxes,box_iou
except ImportError:
    spec = importlib.util.spec_from_file_location("pending_localization",ROOT/"pending_source/signrel/eval/localization.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    jitter_boxes,box_iou = module.jitter_boxes,module.box_iou


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--sigmas",nargs="+",type=float,default=[0,.05,.1,.2,.4])
    p.add_argument("--limit",type=int,default=95)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out",required=True)
    args = p.parse_args()
    if min(args.sigmas) < 0 or len(set(args.sigmas)) != len(args.sigmas):
        p.error("Noise levels must be distinct and nonnegative")
    state = torch.load(args.checkpoint,map_location="cpu",weights_only=False,mmap=True)
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
    dataset = SSLVideoDataset(args.manifest,num_frames=dc['num_frames'],sampling_stride=dc.get('sampling_stride',2),crop_size=dc['crop_size'],augment=False)
    pooler = RegionPooler()
    ac = cfg.get('articulators',{})
    member = lambda boxes,conf,grid: token_membership(boxes[None],conf[None],grid,ac.get('confidence_threshold',.5),ac.get('membership','center'))
    data = {s:[] for s in sorted(args.sigmas)}
    with torch.no_grad():
        for index in range(min(args.limit,len(dataset))):
            item = dataset[index]
            boxes = item['articulator_meta']['boxes']
            conf = item['articulator_meta']['confidence']
            with torch.autocast(torch.device(args.device).type,dtype=torch.bfloat16,enabled=torch.device(args.device).type == 'cuda'):
                output = encoder.forward_tokens(item['video'][None].to(args.device))
            grid = output['grid_shape']
            tokens = output['tokens'].float().cpu().reshape(1,*grid,encoder.dim)
            base_membership = member(boxes,conf,grid)
            nodes,valid = pooler(tokens,base_membership)
            relations = projector(nodes[:,:,u],nodes[:,:,v])
            rel_valid = valid[:,:,u]&valid[:,:,v]
            for sigma in sorted(args.sigmas):
                jb,jc = jitter_boxes(boxes,conf,sigma,args.seed+index)
                membership = member(jb,jc,grid)
                jnodes,jvalid = pooler(tokens,membership)
                jrelations = projector(jnodes[:,:,u],jnodes[:,:,v])
                both = valid&jvalid
                rboth = rel_valid&jvalid[:,:,u]&jvalid[:,:,v]
                cos = F.cosine_similarity(nodes,jnodes,dim=-1)
                rcos = F.cosine_similarity(relations,jrelations,dim=-1)
                per_edge = {f'{a}_{b}':{'n':int(rboth[:,:,e].sum()),'cosine_sum':float(rcos[:,:,e][rboth[:,:,e]].sum())} for e,(a,b) in enumerate(edges)}
                feasibility = {}
                for strategy in ('random_multiblock','articulator','relation_break','motion_salient'):
                    mask_cfg = dict(cfg['masking'])
                    mask_cfg['mixture'] = {strategy:1.}
                    masker = SignMaskGenerator(edges=edges,**mask_cfg)
                    motion = item['articulator_meta']['motion'][None].reshape(1,grid[0],-1,4).mean(2)
                    try:
                        masker.sample(grid,membership,motion,rng=torch.Generator().manual_seed(args.seed+index))
                        feasibility[strategy] = True
                    except AssertionError:
                        feasibility[strategy] = False
                valid_boxes = conf >= ac.get('confidence_threshold',.5)
                data[sigma].append({'sample_id':item['sample_id'], 'valid_nodes_baseline':int(valid.sum()),
                    'valid_nodes_jitter':int(jvalid.sum()),'valid_nodes_common':int(both.sum()),
                    'node_cosine_sum':float(cos[both].sum()), 'valid_relations_baseline':int(rel_valid.sum()),
                    'valid_relations_common':int(rboth.sum()), 'relation_cosine_sum':float(rcos[rboth].sum()),
                    'valid_boxes':int(valid_boxes.sum()),'box_iou_sum':float(box_iou(boxes,jb)[valid_boxes].sum()),
                    'mask_feasibility':feasibility,'per_edge':per_edge})
            if (index+1)%10 == 0:
                print(json.dumps({'processed':index+1}),flush=True)
    curve = []
    for sigma,rows in data.items():
        total = lambda key: sum(r[key] for r in rows)
        ratio = lambda a,b: total(a)/total(b) if total(b) else None
        curve.append({'sigma_relative_box_size':sigma,'clips':len(rows),
            'mean_box_iou':ratio('box_iou_sum','valid_boxes'), 'node_target_cosine':ratio('node_cosine_sum','valid_nodes_common'),
            'relation_target_cosine':ratio('relation_cosine_sum','valid_relations_common'),
            'node_retained_fraction':ratio('valid_nodes_common','valid_nodes_baseline'),
            'relation_retained_fraction':ratio('valid_relations_common','valid_relations_baseline'),
            'mask_feasibility':{k:sum(r['mask_feasibility'][k] for r in rows)/len(rows) for k in rows[0]['mask_feasibility']},
            'per_edge':{k:sum(r['per_edge'][k]['cosine_sum'] for r in rows)/sum(r['per_edge'][k]['n'] for r in rows)
                         if sum(r['per_edge'][k]['n'] for r in rows) else None for k in rows[0]['per_edge']}})
    report = {'arguments':vars(args),'manifest_sha256':sha256(args.manifest),'checkpoint_sha256':sha256(args.checkpoint),
        'source_sha256':sha256(__file__),'upstream_init':state['upstream_init'],'curve':curve,'per_clip':data,
        'protocol':'fixed complete-RGB EMA teacher tokens; seeded slot translation correlated across frames; clipping to image; no confidence inflation',
        'interpretation':'Training-target localization sensitivity and feasible masks. This does not establish robustness of a retrained representation or RGB occlusion transfer.'}
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    for key in ('node_target_cosine','relation_target_cosine'):
        axes[0].plot([r['sigma_relative_box_size'] for r in curve],[r[key] for r in curve],marker='o',label=key.replace('_',' '))
    for key in ('articulator','relation_break','motion_salient'):
        axes[1].plot([r['sigma_relative_box_size'] for r in curve],[r['mask_feasibility'][key] for r in curve],marker='o',label=key)
    for ax in axes:
        ax.set_xlabel('Box translation sigma / box size')
        ax.set_ylim(0,1.02)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Cosine to unperturbed target')
    axes[1].set_ylabel('Fraction of clips with feasible mask')
    fig.tight_layout()
    fig.savefig(out.with_suffix('.png'),dpi=180)
    fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)
    print(json.dumps({'curve':curve}))


if __name__ == '__main__':
    main()
