"""Read-only prior-mean reference and CPU component-gradient probes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from signrep_baseline.cache import load_standardized,clip_targets
from signrep_baseline.schema import PRIOR_SHAPES,TARGET_SCALES
from signrep_baseline.losses import reconstruction_loss,feature_regularization,adversarial_loss


def means(run,target):
    rows=[json.loads(line) for line in (run/'corpus/pretrain.jsonl').read_text().splitlines() if line]
    sums={k:np.zeros(int(np.prod(shape)),np.float64) for k,shape in PRIOR_SHAPES.items()}
    counts={k:np.zeros_like(v) for k,v in sums.items()}
    for i,row in enumerate(rows):
        d,_=load_standardized(row['pose_cache'],row['bone_statistics_sha256'],row['video_sha256'])
        starts=np.flatnonzero(np.convolve(d['body_detected'].astype(int),np.ones(16,int),mode='valid')==16)
        n=len(d['pose_3d']);delta=np.zeros(n+1,np.float64)
        np.add.at(delta,starts,1);np.add.at(delta,starts+16,-1)
        weight=(np.cumsum(delta[:-1])/len(starts))[:,None]
        t,m=clip_targets(d,np.arange(n))
        for key in PRIOR_SHAPES:
            observed=m[key].numpy()*weight
            sums[key]+=(t[key].numpy()*observed).sum(axis=0)
            counts[key]+=observed.sum(axis=0)
        if (i+1)%500==0:print('prior means videos',i+1,flush=True)
    result={k:np.divide(sums[k],counts[k],out=np.zeros_like(sums[k]),where=counts[k]>0).astype(np.float32)
            for k in sums}
    activity=json.loads((target.parent/'pose_priors_audit.json').read_text())['activity_by_side']
    assert all(c['inactive_windows']==0 and c['eligible_windows']>0 for c in activity.values())
    result['activity']=np.ones(2,np.float32)
    np.savez_compressed(target/'corpus_prior_means.npz',**result)
    (target/'corpus_prior_means_scope.json').write_text(json.dumps({
        'videos':len(rows),'scope':'Selected training corpus only; mean targets weighted by the actual uniform-video/uniform-window sampling distribution. Constant-time predictor; not asserted to minimize SmoothL1.',
        'activity':'All eligible audited training labels are active; constant prediction=1 for each hand.'},indent=2)+'\n')


def probe(run,target):
    from signrep_baseline.model import SignRep
    from signrep_baseline.data import PairedVideoDataset,collate_pairs
    from signrep_baseline.augmentation import VideoTransform
    from signrep_baseline.discriminator import StyleDiscriminator,gram_style,style_pairs
    model=SignRep();model.load_state_dict(torch.load(run/'final.model.pt',map_location='cpu',weights_only=True)['model'],strict=True)
    checkpoint=torch.load(run/'train/last.pt',map_location='cpu',weights_only=False)
    discriminator=StyleDiscriminator();discriminator.load_state_dict(checkpoint['discriminator'],strict=True);discriminator.eval()
    em=float(checkpoint['ema']['matched']);eu=float(checkpoint['ema']['unmatched']);del checkpoint
    discriminator_parameter_norms={name:float(value.detach().norm()) for name,value in discriminator.named_parameters()}
    dataset=PairedVideoDataset(run/'corpus/pretrain.jsonl',VideoTransform(True))
    with np.load(target/'corpus_prior_means.npz',allow_pickle=False) as file:
        constants={k:torch.from_numpy(file[k].copy())*TARGET_SCALES.get(k,1.) for k in file.files}
    reports=[]
    for seed in [20261008,20261009,20261010,20261011]:
        torch.manual_seed(seed);indices=torch.randperm(len(dataset))[:2].tolist()
        batch=collate_pairs([dataset[i] for i in indices]);model.train()
        with torch.no_grad():out=model.encode(batch['images'])
        features=out['features'].detach().requires_grad_(True)
        predicted={**model.cls_head(features),'activity':model.active_head(features)}
        recon,terms=reconstruction_loss(predicted,batch['targets'],batch['masks'])
        constant={k:v.expand_as(batch['targets'][k]) for k,v in constants.items()}
        const_loss,const_terms=reconstruction_loss(constant,batch['targets'],batch['masks'])
        var,cov=feature_regularization(features)
        style=gram_style(out['tokens'].detach()).requires_grad_(True)
        matched,unmatched=style_pairs(batch['video_ids'])
        pos,neg=discriminator.paired_logits(features,style,matched,unmatched)
        style_sensitivity=torch.autograd.grad(pos.mean(),style,retain_graph=True)[0]
        content_sensitivity=torch.autograd.grad(pos.mean(),features,retain_graph=True)[0]
        adv=adversarial_loss(pos.sigmoid(),neg.sigmoid(),em,eu)
        components={'reconstruction':recon,'variance':var,'weighted_covariance':.004*cov,'weighted_adversarial':2*adv}
        gradients={k:torch.autograd.grad(value,features,retain_graph=True)[0] for k,value in components.items()}
        total=sum(gradients.values());reg=gradients['variance']+gradients['weighted_covariance'];g=gradients['reconstruction']
        cosine=lambda a,b:float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-20))
        result={'seed':seed,'video_indices':indices,'video_ids':batch['video_ids'],
                'component_losses':{k:float(v.detach()) for k,v in components.items()},
                'component_feature_gradient_l2':{k:float(v.norm()) for k,v in gradients.items()},
                'regularizer_gradient_l2':float(reg.norm()),'total_gradient_l2':float(total.norm()),
                'regularizer_to_reconstruction_gradient_ratio':float(reg.norm()/g.norm().clamp_min(1e-20)),
                'reconstruction_vs_regularizer_gradient_cosine':cosine(g,reg),
                'reconstruction_vs_total_gradient_cosine':cosine(g,total),
                'model_prior_losses':{k:float(v.detach()) for k,v in terms.items()},
                'constant_train_mean_prior_losses':{k:float(v.detach()) for k,v in const_terms.items()},
                'constant_train_mean_reconstruction':float(const_loss),
                'style_vector_norms':style.norm(dim=1).tolist(),
                'matched_style_distance':(style-style[matched]).norm(dim=1).tolist(),
                'unmatched_style_distance':(style-style[unmatched]).norm(dim=1).tolist(),
                'discriminator_matched_probabilities':pos.detach().sigmoid().tolist(),
                'discriminator_unmatched_probabilities':neg.detach().sigmoid().tolist(),
                'discriminator_style_input_gradient_l2':float(style_sensitivity.norm()),
                'discriminator_content_input_gradient_l2':float(content_sensitivity.norm()),
                'activity_valid_elements':int(batch['masks']['activity'].sum())}
        np.savez_compressed(target/f'gradient_probe_seed{seed}.npz',
                            features=features.detach().numpy(),style=style.detach().numpy())
        reports.append(result);print(json.dumps(result),flush=True)
    record={'scope':'Four fixed-seed training-format batches, final checkpoint, CPU float32, 80% mask/drop-path/training augmentations. Gradients measured at features, not all backbone parameters; not a replay of original GPU BF16 minibatches.',
            'final_EMA_matched':em,'final_EMA_unmatched':eu,
            'discriminator_parameter_l2_norms':discriminator_parameter_norms,'probes':reports}
    (target/'gradient_signal_trace.json').write_text(json.dumps(record,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['means','probe']);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    run=args.output.resolve();target=ROOT/'experiments'/run.name/'diagnostics';target.mkdir(parents=True,exist_ok=True)
    {'means':means,'probe':probe}[args.stage](run,target)


if __name__=='__main__':main()
