"""Original/transformed landmark QA with token masks, edges and temporal spans."""
import argparse
import json
import sys
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p/"signrel/config.py").exists())
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
import cv2
import numpy as np
import torch
from signrel.config import load_config
from signrel.data.manifest import read_manifest,sha256,ssl_record
from signrel.data.video_dataset import decode_video
from signrel.data.transforms import transform_clip
from signrel.models.region_pooler import token_membership,DEFAULT_EDGES,ARTICULATORS
from signrel.masks.base import SignMaskGenerator
from visualize_geometry import draw,COLORS


def rgb_frame(video,frame,normalized):
    rgb = video[:,frame].permute(1,2,0).numpy()
    if normalized:
        rgb = rgb*np.array([.229,.224,.225])+np.array([.485,.456,.406])
    return cv2.cvtColor((rgb.clip(0,1)*255).astype(np.uint8),cv2.COLOR_RGB2BGR)


def annotate(image,boxes,conf,points,scores,types):
    h,w = image.shape[:2]
    for ar,color in enumerate(COLORS):
        if conf[ar] >= .5:
            x1,y1,x2,y2 = boxes[ar].numpy()
            cv2.rectangle(image,(int(x1*w),int(y1*h)),(int(x2*w),int(y2*h)),color,2)
            cv2.putText(image,['LH','RH','Face','Body'][ar],(int(x1*w),max(12,int(y1*h))),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
    for point,score,kind in zip(points,scores,types):
        if score >= .5 and bool(((point >= 0)&(point <= 1)).all()):
            cv2.circle(image,(int(point[0]*w),int(point[1]*h)),1,COLORS[int(kind)],-1)
    return image


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--out-root',default='reports/augmentation_qa')
    p.add_argument('--limit',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    args = p.parse_args()
    cfg = load_config(args.config)
    dc,bc = cfg['data'],cfg['model']['backbone']
    grid = (dc['num_frames']//bc.get('tubelet',2),dc['crop_size']//bc.get('patch',16),dc['crop_size']//bc.get('patch',16))
    edges = tuple(tuple(ARTICULATORS.index(v) if isinstance(v,str) else v for v in edge) for edge in cfg['model'].get('relation_edges',DEFAULT_EDGES))
    rows = [ssl_record(r) for r in read_manifest(args.manifest)][:args.limit]
    out = Path(args.out_root)
    out.mkdir(parents=True,exist_ok=True)
    tiles,records = [],[]
    strategies = ['random_multiblock','articulator','relation_break','motion_salient']
    for index,row in enumerate(rows):
        torch.manual_seed(args.seed+index)
        video,frames,_,count = decode_video(row['video_path'],dc['num_frames'],dc.get('sampling_stride',2),row.get('start_sec'),row.get('end_sec'))
        with np.load(row['landmark_path'],allow_pickle=False) as cache:
            positions = np.searchsorted(cache['frame_indices'],frames)
            if int(cache['num_source_frames']) != count or (positions >= len(cache['frame_indices'])).any() or not np.array_equal(cache['frame_indices'][positions],frames):
                raise AssertionError('Exact landmark/video frame alignment required')
            boxes = torch.from_numpy(cache['boxes'][positions].copy())
            conf = torch.from_numpy(cache['box_confidence'][positions].copy())
            points = torch.from_numpy(cache['xy'][positions].copy())
            scores = torch.from_numpy(cache['confidence'][positions].copy())
            types = torch.from_numpy(cache['joint_type'].copy())
            motion = torch.from_numpy(cache['motion'][positions].copy())
        force_flip = bool((index//4)%2)
        transformed,meta,record = transform_clip(video,boxes,conf,size=dc['crop_size'],augment=True,
                                                crop=row.get('signer_crop') or (0,0,1,1),force_flip=force_flip)
        transformed_points = record.apply_points(points)
        transformed_types = types.clone()
        if record.flip:
            transformed_types[types == 0] = 1
            transformed_types[types == 1] = 0
            motion = motion[:,[1,0,2,3]]
        membership = token_membership(meta['boxes'][None],meta['confidence'][None],grid,
                          cfg.get('articulators',{}).get('confidence_threshold',.5),cfg.get('articulators',{}).get('membership','intersection'))
        strategy = strategies[index%4]
        mask_cfg = dict(cfg['masking'])
        mask_cfg['mixture'] = {strategy:1.}
        masker = SignMaskGenerator(edges=edges,**mask_cfg)
        entry = {'index':index,'sample_id':row['sample_id'],'source_frames':frames.tolist(),
                 'crop':record.crop,'flip':record.flip,'mask_strategy_requested':strategy}
        try:
            mask = masker.sample(grid,membership,motion[None].reshape(1,grid[0],-1,4).mean(2),rng=torch.Generator().manual_seed(args.seed+index))
            info = mask.metadata[0]
            entry.update(status='rendered',mask=info,visible_fraction=float(mask.visible_tokens.float().mean()))
            token_time = info.get('start',grid[0]//2)
            frame = min(dc['num_frames']-1,token_time*(dc['num_frames']//grid[0]))
            masked = draw(transformed,meta['boxes'],meta['confidence'],grid,mask,info)
            if 'edge' in info:
                a,b = edges[info['edge']]
                # A tubelet can be valid when its first raw frame is missing a
                # detection. Draw pooled membership centers, never zero boxes.
                yy,xx = torch.meshgrid((torch.arange(grid[1])+.5)/grid[1],(torch.arange(grid[2])+.5)/grid[2],indexing='ij')
                weights = membership[0,token_time]
                centers = torch.stack(((weights*xx).sum((-1,-2)),(weights*yy).sum((-1,-2))),-1)/weights.sum((-1,-2))[:,None].clamp_min(1e-6)*dc['crop_size']
                cv2.line(masked,tuple(centers[a].int().tolist()),tuple(centers[b].int().tolist()),(255,255,255),2)
        except AssertionError as e:
            frame = dc['num_frames']//2
            entry.update(status='infeasible_mask',reason=str(e))
            masked = rgb_frame(transformed,frame,True)
        original = annotate(rgb_frame(video,frame,False),boxes[frame],conf[frame],points[frame],scores[frame],types)
        augmented = annotate(rgb_frame(transformed,frame,True),meta['boxes'][frame],meta['confidence'][frame],transformed_points[frame],scores[frame],transformed_types)
        panels = []
        for name,panel in [('original',original),(f'aug flip={record.flip}',augmented),(strategy,masked)]:
            panel = cv2.resize(panel,(320,240))
            panel = cv2.copyMakeBorder(panel,0,36,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
            cv2.putText(panel,f'{index:03d} {name}',(4,255),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
            cv2.putText(panel,f'frame {int(frames[frame])}; {entry["status"]}',(4,270),cv2.FONT_HERSHEY_SIMPLEX,.35,(210,210,210),1)
            panels.append(panel)
        # A strip shows temporal span boundaries in the exact transformed clip.
        span = entry.get('mask',{})
        selected = sorted(set([0,frame,min(dc['num_frames']-1,span.get('end',grid[0])*bc.get('tubelet',2)-1),dc['num_frames']-1]))
        strip = []
        for f in selected:
            small = cv2.resize(rgb_frame(transformed,f,True),(240,120))
            cv2.putText(small,f'source frame {int(frames[f])}',(4,115),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            strip.append(small)
        strip += [np.zeros_like(strip[0])]*(4-len(strip))
        image = np.concatenate([np.concatenate(panels,axis=1),np.concatenate(strip,axis=1)],axis=0)
        image_path = out/f'{index:03d}.jpg'
        if not cv2.imwrite(str(image_path),image):
            raise OSError(f'Cannot write {image_path}')
        entry['image'] = str(image_path)
        records.append(entry)
        tiles.append(cv2.resize(image,(480,198)))
    for start in range(0,len(tiles),20):
        page = tiles[start:start+20]
        page += [np.zeros_like(page[0])]*(20-len(page))
        cv2.imwrite(str(out/f'contact_{start//20:02d}.jpg'),np.concatenate([np.concatenate(page[j:j+2],axis=1) for j in range(0,20,2)],axis=0))
    report = {'manifest_sha256':sha256(args.manifest),'script_sha256':sha256(__file__),'seed':args.seed,
              'clips':len(rows),'rendered':sum(r['status'] == 'rendered' for r in records), 'records':records,
              'protocol':'alternating forced flip within each mask strategy; shared integer-rounded crop record transforms pixels, boxes and landmarks; exact frame join'}
    (out/'index.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'clips':report['clips'],'rendered':report['rendered']}))


if __name__ == '__main__':
    main()
