"""Descriptive tables/curves/numerical cases from verified immutable scores."""
import argparse
import csv
import io
import json
from math import fsum
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from bounded_artifacts import verify_store


def read(path):return json.loads(Path(path).read_text())


def csv_bytes(rows):
    text=io.StringIO(newline='');w=csv.DictWriter(text,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return text.getvalue().encode()


def main():
    p=argparse.ArgumentParser()
    for k in ('config','run-id','output-dir'):p.add_argument('--'+k,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with BoundedCPUContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        c=run.config;root=Path(c['measured_runtime']);verified=Path(c['verified_runtime'])
        verify_store(root);verify_store(verified);proof=read(verified/'rejection_verification.json');assert proof['passed']
        report=read(root/'rejection_eval.json');meta=read(root/'rejection_metadata.json');erratum=read(verified/'dcg_precision_erratum.json')
        corrections={(r['part'],r['seed'],r['size'],r['method'],r['policy']):r for r in erratum['rows']}
        rows=[]
        for r in report['metric_rows']:
            correction=corrections[(r['part'],r['seed'],r['size'],r['method'],r['policy'])]
            rows.append(dict(part=r['part'],seed=r['seed'],size=r['size'],score=r['method'],policy=r['policy'],
                n_known=r['point']['n_known'],n_unknown=r['point']['n_unknown'],threshold=r['point']['threshold'],
                FAR=r['point']['FAR'],CCR=r['point']['CCR'],known_acceptance=r['point']['known_acceptance'],
                closed_known_Top1=r['known_retrieval']['Top1'],known_Top5=r['known_retrieval']['Top5'],
                known_DCG_FP64_corrected=correction['corrected_FP64_DCG'],known_DCG_FP32_original=correction['reported_FP32_DCG'],
                known_MRR=r['known_retrieval']['MRR'],adapted_joint_AUC=r['adapted_grouped_trapezoid_AUC'],
                common_known_CCR=r['common_known_CCR'],common_known_acceptance=r['common_known_acceptance'],
                persistent_unknown_FAR=r['persistent_unknown_FAR']))
        means=[]
        for part in ('base_dev','MS_dev'):
            for size in (10,25,50):
                for method in c['score_names']:
                    for policy in ('FROZEN50','MATCHED_SOURCE_SIZE'):
                        selected=[r for r in rows if (r['part'],r['size'],r['score'],r['policy'])==(part,size,method,policy)];assert len(selected)==3
                        means.append(dict(part=part,size=size,score=method,policy=policy,aggregation='equal_fixed_three_seed_mean',
                            **{k:fsum(r[k] for r in selected)/3 for k in list(rows[0])[5:]}))
        assert len(rows)==216 and len(means)==48
        run.store.write_bytes('per_seed.csv',csv_bytes(rows));run.store.write_bytes('mean.csv',csv_bytes(means))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        curves=np.load(root/'count_curves.npy',allow_pickle=False);fig,axes=plt.subplots(1,2,figsize=(11,4.7),sharex=True,sharey=True)
        colors=['#31688e','#b44635','#7d5c9d','#318a5b']
        for ax,part,title in zip(axes,('base_dev','MS_dev'),('ASL Citizen dev','MS-ASL validation')):
            for method,color in zip(c['score_names'],colors):
                segments=[s for s in meta['curve_segments'] if (s['part'],s['size'],s['method'])==(part,50,method)]
                assert len(segments)==3
                for j,s in enumerate(segments):
                    curve=curves[s['start']:s['stop']];ax.plot(curve[:,3]/s['n_unknown'],curve[:,2]/s['n_known'],color=color,alpha=.45,lw=1,label=method if j==0 else None)
                op=next(r for r in means if (r['part'],r['size'],r['score'],r['policy'])==(part,50,method,'FROZEN50'))
                ax.scatter([op['FAR']],[op['CCR']],color=color,s=42,edgecolor='white',linewidth=.6,zorder=5)
            ax.set(title=title,xlabel='Unknown false acceptance rate (FAR)',xlim=(0,1),ylim=(0,1));ax.grid(alpha=.2)
        axes[0].set_ylabel('Correct known-class acceptance rate (CCR)');axes[1].legend(frameon=False,fontsize=9,loc='upper left')
        fig.suptitle('50-class one-shot dictionaries: three fixed galleries per score',fontsize=12)
        fig.text(.5,.02,'Thin curves: individual galleries. Dots: mean operating point using source-known 95% acceptance calibration.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.07,1,.95))
        for extension in ('png','pdf'):
            buffer=io.BytesIO();fig.savefig(buffer,format=extension,dpi=180);run.store.write_bytes('joint_curves.'+extension,buffer.getvalue())
        plt.close(fig)
        scores=np.load(root/'scores.npy',allow_pickle=False);pred=np.load(root/'predictions.npy',allow_pickle=False);rank=np.load(root/'known_rank.npy',allow_pickle=False)
        rng=np.random.default_rng(a.seed);cases=[];bucket_order=['known_correct_accepted','known_correct_rejected','known_wrong_accepted','unknown_accepted','unknown_rejected']
        for part in sorted([p for p in meta['parts'] if p['name'] in ('base_dev','MS_dev')],key=lambda p:p['name']):
            for si,seed in enumerate(meta['support_seeds']):
                tau=report['calibration'][f'{seed}:50:maxcosine']['threshold'];buckets={name:[] for name in bucket_order}
                for i in range(part['start'],part['stop']):
                    known=rank[si,2,i]>0;correct=pred[si,2,i]==meta['labels'][i];accepted=scores[si,2,0,i]>=tau
                    if known and correct:key='known_correct_accepted' if accepted else 'known_correct_rejected'
                    elif known and accepted:key='known_wrong_accepted'
                    elif not known:key='unknown_accepted' if accepted else 'unknown_rejected'
                    else:continue
                    buckets[key].append(i)
                for bucket,pool in buckets.items():
                    chosen=sorted(rng.choice(pool,size=min(2,len(pool)),replace=False).tolist()) if pool else []
                    selected=[]
                    for i in chosen:
                        selected.append(dict(sample_id=meta['query_ids'][i],class_id=meta['labels'][i],official_signer=meta['signers'][i],predicted_class=int(pred[si,2,i]),known_rank=int(rank[si,2,i]),
                            scores={method:float(scores[si,2,mi,i]) for mi,method in enumerate(c['score_names'])},
                            accepted={method:bool(scores[si,2,mi,i]>=report['calibration'][f'{seed}:50:{method}']['threshold']) for mi,method in enumerate(c['score_names'])}))
                    cases.append(dict(part=part['name'],seed=seed,bucket=bucket,eligible_count=len(pool),sampled=selected))
        run.save('numerical_cases.json',dict(posthoc=True,seed=a.seed,selection='maxcos/FROZEN50/gallery50, sortedpart/ascendingseed/fixedbucketorder, uniform up to2',
            cases=cases,scope='Metadata/ranks/acceptance only; no human linguistic/visual interpretation.'))
        run.save('report_summary.json',dict(primary=report['primary'],per_seed_table_rows=216,mean_table_rows=48,figure_curve_count=24,
            operating_point_count=8,DCG_precision_version=erratum['version'],matplotlib_version=matplotlib.__version__,
            posthoc_case_buckets=len(cases),data_source=str(root),verification_source=str(verified),GPU_seconds=0,new_encoder_calls=0,
            scope='Derived descriptive artifacts, all four existing scores, no new significance/hypothesis/test or method promotion.'))


if __name__=='__main__':main()
