"""Visual coordinate hypothesis audit; raw backend scores remain uncalibrated.

Produces diagnostic candidate overlays, not approved canonical caches.
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"src"))
from signcoord.data.video_io import decode_video
from signcoord.provenance import atomic_json, execution_stamp, sha256_file


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",required=True); p.add_argument("--samples",type=int,default=50)
    a=p.parse_args(); root=Path(a.output); root.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(3407)
    aslc=Path('/home/dongvk/datasets/ASL_Citizen')
    names=[Path(r['Video file']).stem for r in csv.DictReader((aslc/'splits/train.csv').open())]
    rng.shuffle(names)
    sources=[("asl_citizen",aslc/'videos',aslc/'keypoints',names[:a.samples//2])]
    h2s=Path('/home/shared_data/sign_language/How2Sign/train')
    names=sorted(p.stem for p in (h2s/'train_pose').glob('*.pkl')); rng.shuffle(names)
    sources.append(("how2sign",h2s/'raw_videos',h2s/'train_pose',names[:a.samples-a.samples//2]))
    rows=[]; all_tiles=[]
    colors={'body':(255,255,0),'face':(80,240,80),'L':(255,80,80),'R':(80,140,255)}
    refs=[]
    for ref in ['method/signrep_spectral/data/pose_reader.py','method/signrep_spectral/records/discovery.md',
                'method/p4_signssl/p4/cip_states.py','method/p4_signssl/scripts/explore_pose_motion.py',
                'third_party/SignRep/example_usage.py']:
        path=PROJECT.parents[1]/ref; refs.append({'path':ref,'sha256':sha256_file(path)})
    for dataset,vr,pr,names in sources:
        for stem in names:
            try:
                d=pickle.load((pr/(stem+'.pkl')).open('rb'))
                video,pts=decode_video(vr/(stem+'.mp4'))
                if len(video) != len(d['keypoints']): raise ValueError('source-frame count mismatch')
                counts=[len(v) for v in d['keypoints']]
                usable=[t for t,n in enumerate(counts) if n==1]
                if not usable: raise ValueError('no single-person frame')
                t=usable[len(usable)//2]; xy=np.asarray(d['keypoints'][t])[0]; scores=np.asarray(d['scores'][t])[0]
                h,w=video.shape[1:3]
                thumbnails=[]
                for policy,multiplier in [('xy_times_W_H',(w,h)),('xy_times_W_W',(w,w)),('xy_times_H_H',(h,h))]:
                    im=Image.fromarray(video[t]); draw=ImageDraw.Draw(im)
                    points=xy*np.array(multiplier)
                    for j in list(range(17))+list(range(23,133)):
                        x,y=points[j]
                        color=colors['body' if j<17 else 'face' if j<91 else 'L' if j<112 else 'R']
                        if np.isfinite([x,y]).all(): draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
                    im.thumbnail((260,190)); thumb=Image.new('RGB',(260,212),(15,15,15)); thumb.paste(im,(0,22))
                    ImageDraw.Draw(thumb).text((4,4),policy,fill='white'); thumbnails.append(thumb)
                row=Image.new('RGB',(780,237),(15,15,15))
                ImageDraw.Draw(row).text((4,4),f'{dataset}: {stem[:85]} frame={t}',fill='white')
                for col,thumb in enumerate(thumbnails): row.paste(thumb,(col*260,25))
                all_tiles.append(row)
                parts={'body':np.arange(17),'face':np.arange(23,91),'L':np.arange(91,112),'R':np.arange(112,133)}
                rows.append({'dataset':dataset,'sample_id':stem,'video_sha256':sha256_file(vr/(stem+'.mp4')),
                             'pose_sha256':sha256_file(pr/(stem+'.pkl')),'source_frames':len(video),
                             'single_person_fraction':counts.count(1)/len(counts),'video_size':[w,h],
                             'score_quantiles_per_part':{part:np.quantile(scores[idx],[0,.1,.5,.9,1]).tolist() for part,idx in parts.items()},
                             'xy_W_H_inside_fraction':float(((xy>=0)&(xy<=1)).all(-1).mean())})
            except Exception as exc:
                rows.append({'dataset':dataset,'sample_id':stem,'error':str(exc)})
    for i in range(0,len(all_tiles),5):
        sheet=Image.new('RGB',(780,237*len(all_tiles[i:i+5])))
        for j,tile in enumerate(all_tiles[i:i+5]): sheet.paste(tile,(0,j*237))
        sheet.save(root/f'overlay_sheet_{i//5:02d}.jpg',quality=92)
    report={**execution_stamp(),'status':'candidate_coordinate_audit','code_references':refs,'samples':rows,
            'coordinate_hypotheses':['x/W,y/H','x/W,y/W','x/H,y/H'],
            'extractor_identity':'unknown_in_previous_method_records_and_local_cache',
            'confidence_semantics':'raw_backend_scores_not_probabilities; no sigmoid/clipping approved',
            'scientific_results':False}
    atomic_json(root/'trace_report.json',report)
    print(json.dumps({'output':str(root),'samples':len(rows),'overlay_samples':len(all_tiles)}))


if __name__ == '__main__': main()
