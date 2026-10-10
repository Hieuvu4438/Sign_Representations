"""Retrospective CPU diagnostics from immutable ASLC002 cached features."""
import argparse
import json
from pathlib import Path
import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def rank_metrics(scores, labels, truth):
    classes = np.unique(labels)
    best = np.stack([scores[:, labels == c].max(axis=1) for c in classes], axis=1)
    # Float64 sensitivity diagnostics; exact rounded ties may differ from official float32.
    order = np.argsort(-best, axis=1, kind='stable')
    ranks = np.argmax(classes[order] == truth[:, None], axis=1) + 1
    return {'top1_percent': float((ranks <= 1).mean() * 100),
            'top5_percent': float((ranks <= 5).mean() * 100),
            'MRR': float((1/ranks).mean()), 'median_rank': float(np.median(ranks))}


def summary(values):
    x = np.asarray(values)
    return {'mean':float(x.mean()),'std':float(x.std()),
            'p05':float(np.quantile(x,.05)),'median':float(np.median(x)),
            'p95':float(np.quantile(x,.95)),'min':float(x.min()),'max':float(x.max())}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();folder=args.output.resolve()
    target=folder.parents[1]/'experiments'/folder.name/'diagnostics';target.mkdir(parents=True,exist_ok=True)
    report={'scope':'Retrospective float64 cache diagnostics; validation reuse is exploratory, not an independently evaluated new method.'}
    for phase in ['step0','step5000']:
        location=folder/'retrieval'/phase
        g=rows(location/'gallery_features.jsonl');q=rows(location/'queries_features.jsonl')
        labels=np.array([r['label'] for r in g]);truth=np.array([r['label'] for r in q])
        vectors=[];weighted=[];activity=[];weight_cv=[];window_norms=[]
        for r in g+q:
            with np.load(r['features'],allow_pickle=False) as file:
                f=file['features'].astype(np.float64);a=file['activity'].astype(np.float64)
            w=a.max(axis=1);vectors.append(f.mean(axis=0));weighted.append((f*w[:,None]).sum(axis=0)/w.sum())
            activity.append(a);weight_cv.append(float(w.std()/(abs(w.mean())+1e-12)));window_norms.append(np.linalg.norm(f,axis=1))
        raw=np.stack(vectors);features=normalize(raw);features_w=normalize(np.stack(weighted))
        np.savez_compressed(target/f'{phase}_pooled.npz',raw=raw,unit=features,weighted_unit=features_w)
        gallery=features[:len(g)];query=features[len(g):];scores=query@gallery.T
        same=truth[:,None]==labels[None,:];nearest=scores.argmax(axis=1)
        frequencies=np.bincount(nearest,minlength=len(g));true_scores=np.where(same,scores,-np.inf).max(axis=1)
        wrong_scores=np.where(~same,scores,-np.inf).max(axis=1)
        centered=gallery-gallery.mean(axis=0);cov=centered.T@centered/len(gallery)
        eig=np.linalg.eigvalsh(cov).clip(min=0);prob=eig/eig.sum();positive=prob[prob>0]
        within=np.zeros_like(gallery)
        for c in np.unique(labels):
            members=labels==c;within[members]=gallery[members]-gallery[members].mean(axis=0)
        between=float(np.sum(centered**2)-np.sum(within**2))
        variants={'official_protocol_float64_sensitivity':rank_metrics(scores,labels,truth)}
        # Use only gallery statistics; this is a pre-declared hypothesis probe,
        # not a new result selected/tuned on validation.
        center=gallery.mean(axis=0)
        variants['gallery_mean_centered']=rank_metrics(normalize(query-center)@normalize(gallery-center).T,labels,truth)
        train_scores=gallery@gallery.T;np.fill_diagonal(train_scores,-np.inf)
        train_nearest=train_scores.argmax(axis=1)
        signers=np.array([r['signer_id'] for r in g]);same_signer=signers[:,None]==signers[None,:]
        train_metrics=rank_metrics(train_scores,labels,labels)
        same_signer_nn=float((signers[train_nearest]==signers).mean())
        train_scores[same_signer]=-np.inf
        train_cross_signer=rank_metrics(train_scores,labels,labels)
        activity=np.concatenate(activity)
        result={'retrieval_variants':variants,'pooled_raw_norm':summary(np.linalg.norm(raw,axis=1)),
                'window_feature_norm':summary(np.concatenate(window_norms)),
                'same_class_query_gallery_cosine':summary(scores[same]),
                'different_class_query_gallery_cosine':summary(scores[~same]),
                'best_correct_minus_best_wrong_cosine':summary(true_scores-wrong_scores),
                'gallery_effective_rank':float(np.exp(-(positive*np.log(positive)).sum())),
                'gallery_top1_PC_variance_fraction':float(prob[-1]),
                'gallery_top10_PC_variance_fraction':float(prob[-10:].sum()),
                'gallery_between_class_variance_fraction':between/float(np.sum(centered**2)),
                'nearest_gallery_unique_videos':int((frequencies>0).sum()),
                'most_popular_gallery_video_query_count':int(frequencies.max()),
                'top10_gallery_videos_query_fraction':float(np.sort(frequencies)[-10:].sum()/len(q)),
                'train_leave_one_video_out':train_metrics,
                'train_nearest_neighbor_same_signer_fraction':same_signer_nn,
                'train_gallery_same_signer_fraction_by_chance':float((same_signer.sum(axis=1)-1).mean()/(len(g)-1)),
                'train_leave_one_video_out_excluding_same_signer':train_cross_signer,
                'activity_outputs':{'LH':summary(activity[:,0]),'RH':summary(activity[:,1]),
                    'outside_0_1_fraction':float(((activity<0)|(activity>1)).mean()),
                    'window_weight_CV_per_video':summary(weight_cv),
                    'cosine_between_mean_and_activity_pooled_video_vectors':summary((features*features_w).sum(axis=1))}}
        report[phase]=result
        (target/'cached_retrieval_trace.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'phase':phase,**result}),flush=True)


if __name__=='__main__':main()
