"""Frozen held-out retrieval under fixed RGB corruption, with a clean gallery."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'signrel/config.py').exists())
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from signrel.config import load_config
from signrel.data.manifest import read_manifest,sha256
from signrel.data.video_dataset import decode_video
from signrel.data.transforms import transform_clip
from signrel.eval.extract import load_encoder,rgb_embedding
from signrel.eval.retrieval import retrieval
try:
    from signrel.eval.occlusion import occlude_clip
except ImportError:
    spec = importlib.util.spec_from_file_location('pending_occlusion',ROOT/'pending_source/signrel/eval/occlusion.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    occlude_clip = module.occlude_clip


def load(path):
    with np.load(path,allow_pickle=False) as z:
        return z['features'],[json.loads(v) for v in z['rows_json'].tolist()]


def main():
    p = argparse.ArgumentParser()
    init = p.add_mutually_exclusive_group(required=True)
    init.add_argument('--checkpoint')
    init.add_argument('--config')
    p.add_argument('--manifest',required=True)
    p.add_argument('--gallery',required=True)
    p.add_argument('--clean-features',required=True)
    p.add_argument('--fractions',nargs='+',type=float,default=[0,.05,.1,.2,.4])
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out',required=True)
    args = p.parse_args()
    if not args.fractions or min(args.fractions) < 0 or max(args.fractions) > 1 or len(set(args.fractions)) != len(args.fractions):
        p.error('Fractions must be distinct in [0,1]')
    state = torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True) if args.checkpoint else None
    cfg = state['config'] if state else load_config(args.config)
    encoder = load_encoder(args.checkpoint,cfg['model']['backbone'],args.device)
    dc = cfg['data']
    clean,qr = load(args.clean_features)
    gallery,gr = load(args.gallery)
    clean_meta = json.loads(Path(args.clean_features).with_suffix('.report.json').read_text())
    gallery_meta = json.loads(Path(args.gallery).with_suffix('.report.json').read_text())
    manifest = read_manifest(args.manifest)
    assert [r['sample_id'] for r in manifest] == [r['sample_id'] for r in qr], 'Query coverage differs from clean extraction'
    assert clean_meta['manifest_sha256'] == sha256(args.manifest)
    assert clean_meta['num_frames'] == dc['num_frames'] and clean_meta['crop_size'] == dc['crop_size']
    assert clean_meta['arguments']['windows'] == gallery_meta['arguments']['windows'] == 1
    provenance_notes = []
    if args.checkpoint:
        assert clean_meta['checkpoint_sha256'] == gallery_meta['checkpoint_sha256'] == sha256(args.checkpoint)
    else:
        origin = clean_meta['upstream_init']
        assert origin == gallery_meta['upstream_init']
        assert all(encoder.upstream_init.get(k) == v for k,v in origin.items()), 'Frozen backbone origin differs'
        missing = set(encoder.upstream_init)-set(origin)
        assert missing in (set(),{'upstream_commit'}), 'Unrecorded backbone contract fields'
        if missing:
            assets = json.loads((ROOT/'reports/assets.json').read_text())
            assert assets['vjepa2_commit'] == encoder.upstream_init['upstream_commit']
            assert assets['assets']['backbone']['sha256'] == origin['sha256']
            provenance_notes.append('Original frozen feature report predates upstream_commit field; pinned asset provenance and all recorded backbone fields match')
    fractions = sorted(args.fractions)
    vectors = {f:[] for f in fractions if f != 0}
    areas = {f:[] for f in fractions}
    rectangles = []
    precision = clean_meta['arguments']['precision']
    with torch.no_grad():
        for index,row in enumerate(qr):
            video,_,_,_ = decode_video(row['video_path'],dc['num_frames'],dc.get('sampling_stride',2),row.get('start_sec'),row.get('end_sec'))
            x,_,_ = transform_clip(video,size=dc['crop_size'],crop=row.get('signer_crop') or (0,0,1,1))
            for fraction in fractions:
                masked,info = occlude_clip(x,fraction,args.seed+index,max(fractions))
                areas[fraction].append(info['actual_area_fraction'])
                rectangles.append({'sample_id':row['sample_id'],'fraction':fraction,**info})
                if fraction == 0:
                    continue
                with torch.autocast(torch.device(args.device).type,dtype=torch.bfloat16,enabled=precision == 'bf16'):
                    z = rgb_embedding(encoder,masked[None].to(args.device)).float().cpu().numpy()[0]
                vectors[fraction].append(z)
            if (index+1)%20 == 0:
                print(json.dumps({'processed':index+1}),flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    curve = []
    for fraction in fractions:
        z = clean if fraction == 0 else np.stack(vectors[fraction])
        metrics = retrieval(z,gallery,qr,gr,exclude_same_signer=True,class_max=True)
        metric_path = Path('features/occlusion')/f'{out.stem}_area{fraction:.3f}.npz'
        metric_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(metric_path,features=z,rows_json=np.array([json.dumps(r) for r in qr]))
        curve.append({'requested_area_fraction':fraction,'actual_area_fraction_mean':float(np.mean(areas[fraction])),
                      'retrieval':metrics,'features':str(metric_path),'feature_sha256':sha256(metric_path)})
    report = {'arguments':vars(args),'manifest_sha256':sha256(args.manifest),'clean_feature_sha256':sha256(args.clean_features),
              'gallery_sha256':sha256(args.gallery),'checkpoint_sha256':sha256(args.checkpoint) if args.checkpoint else None,
              'upstream_init':state['upstream_init'] if state else encoder.upstream_init,'backbone_config':cfg['model']['backbone'],
              'script_sha256':sha256(__file__),'curve':curve,'rectangles':rectangles,
              'provenance_notes':provenance_notes,
              'protocol':'clean gallery; seeded nested square tube occluder on normalized RGB query; same center/location and exact query rows at every level',
              'interpretation':'Descriptive robustness on a fixed small held-out pilot; no pose input at inference; no recipe selection or retraining using corrupted evaluation data'}
    out.write_text(json.dumps(report,indent=2,allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(6,4))
    for key in ('Recall@1','Recall@5','mAP_instance'):
        ax.plot([r['actual_area_fraction_mean'] for r in curve],[r['retrieval'][key] for r in curve],marker='o',label=key)
    ax.set_xlabel('Occluded query image fraction')
    ax.set_ylabel('Frozen retrieval score')
    ax.set_ylim(0,1)
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout();fig.savefig(out.with_suffix('.png'),dpi=180);fig.savefig(out.with_suffix('.pdf'));plt.close(fig)
    print(json.dumps({'curve':curve}))


if __name__ == '__main__':
    main()
