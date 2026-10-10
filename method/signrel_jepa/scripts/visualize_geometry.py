import _bootstrap
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from signrel.data.video_dataset import SSLVideoDataset
from signrel.models.region_pooler import token_membership
from signrel.masks.base import SignMaskGenerator
from signrel.config import load_config

COLORS=[(255,100,50),(50,180,255),(120,255,100),(255,100,255)]


def draw(video,boxes,confidence,grid,mask,info):
    mean=np.array([0.485,0.456,0.406])
    std=np.array([0.229,0.224,0.225])
    c,t,h,w=video.shape
    tt,hh,ww=grid
    token_time=info.get("start",tt//2)
    frame=min(t-1,token_time*(t//tt))
    rgb=((video[:,frame].permute(1,2,0).numpy()*std+mean).clip(0,1)*255).astype(np.uint8)
    image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    overlay=image.copy()
    hidden=mask.hidden_tokens[0].reshape(tt,hh,ww)[token_time].cpu().numpy()
    for y in range(hh):
        for x in range(ww):
            a,b=int(x*w/ww),int(y*h/hh)
            d,e=int((x+1)*w/ww),int((y+1)*h/hh)
            if hidden[y,x]:
                cv2.rectangle(overlay,(a,b),(d,e),(30,30,220),-1)
            cv2.rectangle(image,(a,b),(d,e),(80,80,80),1)
    image=cv2.addWeighted(image,0.65,overlay,0.35,0)
    for a,color in enumerate(COLORS):
        if confidence[frame,a]<0.5:
            continue
        x1,y1,x2,y2=boxes[frame,a].numpy()
        cv2.rectangle(image,(int(x1*w),int(y1*h)),(int(x2*w),int(y2*h)),color,2)
        cv2.putText(image,["LH","RH","Face","Body"][a],(int(x1*w),max(12,int(y1*h))),cv2.FONT_HERSHEY_SIMPLEX,0.35,color,1)
    return image


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--config",required=True)
    p.add_argument("--out-root",default="reports/geometry_qa")
    p.add_argument("--limit",type=int,default=100)
    args=p.parse_args()
    cfg=load_config(args.config)
    dc=cfg["data"]
    ds=SSLVideoDataset(args.manifest,num_frames=dc["num_frames"],crop_size=dc["crop_size"],
                        sampling_stride=dc.get("sampling_stride",2),augment=False)
    bc=cfg["model"]["backbone"]
    grid=(dc["num_frames"]//bc.get("tubelet",2),dc["crop_size"]//bc.get("patch",16),dc["crop_size"]//bc.get("patch",16))
    masker=SignMaskGenerator(**cfg["masking"])
    dest=Path(args.out_root)
    dest.mkdir(parents=True,exist_ok=True)
    records,tiles=[],[]
    torch.manual_seed(cfg.get("seed",42))
    for i in range(min(len(ds),args.limit)):
        item=ds[i]
        meta=item["articulator_meta"]
        member=token_membership(meta["boxes"][None],meta["confidence"][None],grid,mode="intersection")
        try:
            spec=masker.sample(grid,member,meta["motion"][None].reshape(1,grid[0],-1,4).mean(2))
            info=spec.metadata[0]
            image=draw(item["video"],meta["boxes"],meta["confidence"],grid,spec,info)
            path=dest/f"{i:03d}.jpg"
            cv2.imwrite(str(path),image)
            records.append({"index":i,"sample_id":item["sample_id"],"mask":info,"status":"rendered","image":str(path)})
        except AssertionError as e:
            image=np.zeros((dc["crop_size"],dc["crop_size"],3),np.uint8)
            records.append({"index":i,"sample_id":item["sample_id"],"status":"invalid_regions","reason":str(e)})
        tile=cv2.resize(image,(224,224))
        tile=cv2.copyMakeBorder(tile,0,30,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
        cv2.putText(tile,f"{i:03d} {records[-1]['status']}",(4,243),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
        tiles.append(tile)
    for start in range(0,len(tiles),25):
        page=tiles[start:start+25]
        page += [np.zeros_like(page[0])]*(25-len(page))
        sheet=np.concatenate([np.concatenate(page[j:j+5],axis=1) for j in range(0,25,5)],axis=0)
        cv2.imwrite(str(dest/f"contact_{start//25:02d}.jpg"),sheet)
    (dest/"index.json").write_text(json.dumps(records,indent=2))
    print(json.dumps({"clips":len(records),"rendered":sum(r["status"]=="rendered" for r in records)}))


if __name__=="__main__":
    main()
