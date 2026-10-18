"""Metadata adapters for local WLASL and MS-ASL; pose files remain read-only."""
import argparse
import json
from pathlib import Path
from pp.train import atomic_json
from pp.fit_stats import fingerprint


def wlasl(root,vocab=2000,annotations=None):
    root=Path(root);path=root/'preprocess'/f'nslt_{vocab}.json'
    nslt=json.loads(path.read_text());names={int(l.split(maxsplit=1)[0]):l.split(maxsplit=1)[1].strip() for l in (root/'preprocess/wlasl_class_list.txt').read_text().splitlines()}
    signers={}
    if annotations:
        for record in json.loads(Path(annotations).read_text()):
            for inst in record['instances']:signers[str(inst['video_id'])]=str(inst['signer_id'])
    rows={s:[] for s in ('train','val','test')}
    for sid,r in sorted(nslt.items()):
        pose=root/'keypoint'/f'{sid}.pkl';video=root/'WLASL2000'/f'{sid}.mp4'
        if not pose.exists() or not video.exists():raise ValueError(f'Missing official WLASL clip {sid}')
        label=int(r['action'][0]);split=r['subset']
        rows[split].append(dict(sample_id=sid,official_split=split,pose_relpath=str(pose.resolve()),video=str(video.resolve()),
                               width=256,height=256,fps=25.,class_id=label,gloss=names[label],signer_id=signers.get(sid,'unknown'),dataset='wlasl',
                               signer_dependent=True))
    return rows,dict(dataset='wlasl',vocab=vocab,source_sha256=fingerprint(path),signer_annotations=annotations,signer_overlap_expected=True)


def msasl(root):
    root=Path(root);path=root/'msasl1k.json';records=json.loads(path.read_text());rows={s:[] for s in ('train','val','test')};missing=[]
    import cv2
    for i,r in enumerate(records):
        sid=r.get('video_name',f'{i+1:06d}');pose=root/'keypoint'/f'{sid}.pkl';video=root/'cut_processed_video'/f'{sid}.mp4'
        if not pose.exists() or not video.exists():missing.append(sid);continue
        cap=cv2.VideoCapture(str(video));width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));fps=float(cap.get(cv2.CAP_PROP_FPS));cap.release()
        if min(width,height,fps)<=0:raise ValueError(f'Invalid video metadata {video}')
        split=r['subset'];rows[split].append(dict(sample_id=sid,official_split=split,pose_relpath=str(pose.resolve()),video=str(video.resolve()),
            width=width,height=height,fps=fps,class_id=int(r['label']),gloss=r['clean_text'],signer_id=str(r['signer_id']),dataset='msasl',
            source_youtube_id=r.get('ytb_id'),local_partial_dataset=True))
    return rows,dict(dataset='msasl',source_sha256=fingerprint(path),missing_ids=missing,local_partial_dataset=True,
                     fraction_available=sum(len(r) for r in rows.values())/len(records))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['wlasl','msasl'],required=True)
    ap.add_argument('--root',required=True);ap.add_argument('--out',required=True);ap.add_argument('--vocab',type=int,choices=[100,300,1000,2000],default=2000)
    ap.add_argument('--signer-annotations');a=ap.parse_args()
    rows,report=wlasl(a.root,a.vocab,a.signer_annotations) if a.dataset=='wlasl' else msasl(a.root)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    for split,rs in rows.items():(out/f'metadata_{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rs))
    atomic_json(out/'manifest.json',report|dict(counts={s:len(r) for s,r in rows.items()}))
if __name__=='__main__':main()
