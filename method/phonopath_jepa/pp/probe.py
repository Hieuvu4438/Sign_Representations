"""ASL-LEX probes, gloss-disjoint AND signer-disjoint, independent annotation labels."""
import argparse
import csv
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from pp.train import atomic_json

PARAMS={
 'SignType':('SignType.2.0',['OneHanded','SymmetricalOrAlternating','AsymmetricalDifferentHandshape','AsymmetricalSameHandshape']),
 'Movement':('Movement.2.0',['Straight','Curved','Circular','Z-shaped','BackAndForth']),
 'Repeated':('RepeatedMovement.2.0',['0','1']), 'MajorLoc':('MajorLocation.2.0',['Neutral','Head','Hand','Body','Arm']),
 'Contact':('Contact.2.0',['0','1']),'FlexChange':('FlexionChange.2.0',['0','1']),
 'WristTwist':('UlnarRotation.2.0',['0','1']),'SelFingers':('SelectedFingers.2.0',['imrp','i','im','t','m','p']),
 'Flexion':('Flexion.2.0',['FullyOpen','Curved','Flat','Bent','FullyClosed'])}


def bootstrap_balanced(yt,pred,gloss,boot=500,seed=0):
    rng=np.random.default_rng(seed);groups=np.unique(gloss);indices=[np.flatnonzero(gloss==g) for g in groups];values=[]
    for _ in range(boot):
        ii=np.concatenate([indices[i] for i in rng.integers(len(groups),size=len(groups))])
        if len(np.unique(yt[ii]))<2:continue
        values.append(100*balanced_accuracy_score(yt[ii],pred[ii]))
    return np.percentile(values,[2.5,97.5]).tolist() if values else [None,None]


def run(train,val,lex,code_map,readout='grp',pca=256,boot=500):
    tg,vg=train['gloss'],val['gloss']
    if set(train['signer'])&set(val['signer']):raise ValueError('ASL-LEX v3 requires signer-disjoint splits')
    gloss=np.unique(np.r_[tg,vg]);held=set(np.random.default_rng(0).choice(gloss,max(1,int(.3*len(gloss))),replace=False))
    mt=np.array([g not in held for g in tg]);mv=np.array([g in held for g in vg])
    if not mt.any() or not mv.any():raise ValueError('Insufficient gloss-disjoint probe data')
    rows_t=json.loads(str(train['rows']));rows_v=json.loads(str(val['rows']))
    sc=StandardScaler().fit(train[readout][mt]);G=sc.transform(train[readout][mt]);V=sc.transform(val[readout][mv])
    pc=PCA(min(pca,*G.shape),svd_solver='full').fit(G);G=pc.transform(G);V=pc.transform(V)
    def labels(rows,col,classes):
        values=[]
        for r in rows:
            value=lex.get(code_map.get(r.get('source_stem',''),''),{}).get(col,'').strip()
            if value.replace('.','',1).isdigit():value=str(int(float(value)))
            values.append(classes.index(value) if value in classes else -1)
        return np.array(values)
    result=dict(readout=readout,pca=G.shape[1],heldout_glosses=sorted(held),train_n=int(mt.sum()),val_n=int(mv.sum()),probes={})
    for name,(col,classes) in PARAMS.items():
        yt=labels(rows_t,col,classes)[mt];yv=labels(rows_v,col,classes)[mv];ot=yt>=0;ov=yv>=0
        if len(np.unique(yt[ot]))<2 or not ov.any():
            result['probes'][name]=dict(status='insufficient labels');continue
        model=LogisticRegression(max_iter=3000,C=.05,class_weight='balanced').fit(G[ot],yt[ot])
        pred=model.predict(V[ov]);true=yv[ov]
        result['probes'][name]=dict(balanced_accuracy=float(100*balanced_accuracy_score(true,pred)),
                                  ci95=bootstrap_balanced(true,pred,vg[mv][ov],boot),n=int(ov.sum()),exploratory=True)
    return result


def minimal_pairs(embedding,gloss,annotations,parameter):
    """Independent annotations: gloss -> parameter dictionary; one-difference pairs."""
    from pp.evaluate import normalize
    groups=np.unique(gloss);means=normalize(np.stack([embedding[gloss==g].mean(0) for g in groups]))
    pairs=[]
    for i,g in enumerate(groups):
        a=annotations.get(g)
        if not a:continue
        for j in range(i+1,len(groups)):
            b=annotations.get(groups[j])
            if b and set(a)==set(b) and [k for k in a if a[k]!=b[k]]==[parameter]:
                pairs.append(dict(gloss_a=str(g),gloss_b=str(groups[j]),cosine=float(means[i]@means[j])))
    return pairs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',required=True);ap.add_argument('--val',required=True)
    ap.add_argument('--asllex',required=True);ap.add_argument('--splits',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--readout',default='grp');ap.add_argument('--pca',type=int,default=256);ap.add_argument('--boot',type=int,default=500)
    a=ap.parse_args();from pathlib import Path
    with open(a.asllex,encoding='latin-1') as f:lex={r['Code']:r for r in csv.DictReader(f)}
    cmap={}
    for sp in ('train','val'):
        with open(Path(a.splits)/f'{sp}.csv') as f:
            for r in csv.DictReader(f):cmap[Path(r['Video file']).stem]=r['ASL-LEX Code']
    result=run(np.load(a.train),np.load(a.val),lex,cmap,a.readout,a.pca,a.boot)
    atomic_json(a.out,result)
if __name__=='__main__':main()
