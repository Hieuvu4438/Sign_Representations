"""Post-outcome descriptive tables/figures/cases; no new fit or method selection."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from bounded_cpu_context import BoundedCPUContext
from bounded_artifacts import verify_store


def main():
    p=argparse.ArgumentParser()
    for key in ('config','run-id','output-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with BoundedCPUContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        c=run.config;parent=Path(c['measured_runtime']);verified=Path(c['verified_runtime'])
        assert json.loads((parent/'status.json').read_text())['status']=='completed'
        assert json.loads((verified/'geometry_eval_verification.json').read_text())['passed']
        stores=dict(measured=verify_store(parent),verified=verify_store(verified))
        report=json.loads((parent/'geometry_eval.json').read_text());meta=json.loads((parent/'shared_metadata.json').read_text())['ranking']
        modes=meta['modes'];seeds=meta['support_seeds'];assert modes==['RR','TR','RT','TT','raw','centered'] and seeds==[1701,1702,1703]
        rows=[]
        for mode in modes:
            for gallery in [str(s) for s in seeds]+['mean_fixed_three']:
                metrics=report['mean_over_fixed_galleries'][mode] if gallery=='mean_fixed_three' else report['metrics'][mode][gallery]
                rows.append(dict(mode=mode,gallery=gallery,query_count=748,class_count=100,
                    **{kind+'_'+key:metrics[kind][key] for kind in ('micro','macro') for key in ('dcg','mrr','top1','top5')}))
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        run.store.write_bytes('metrics.csv',stream.getvalue().encode())
        # Fixed-ID descriptive examples, selected after results; never scientific controls or new tuning.
        ranks=np.load(parent/'ranks.npy',allow_pickle=False).reshape(6,3,748,100);labels=np.array(meta['labels'])
        rr=ranks[0,0,:,0]==labels;raw=ranks[4,0,:,0]==labels
        buckets={'shared_success':np.flatnonzero(rr & raw),'shared_failure':np.flatnonzero(~rr & ~raw),
                 'RR_only_correct':np.flatnonzero(rr & ~raw),'raw_only_correct':np.flatnonzero(~rr & raw)}
        rng=np.random.default_rng(a.seed);cases={}
        for name,indices in buckets.items():
            chosen=sorted(rng.choice(indices,size=min(3,len(indices)),replace=False).tolist());entries=[]
            for qi in chosen:
                entry=dict(sample_id=meta['query_ids'][qi],class_id=int(labels[qi]),official_signer_id=meta['participants'][qi],
                    ranks={mode:{str(seed):int(np.flatnonzero(ranks[mi,si,qi]==labels[qi])[0])+1 for si,seed in enumerate(seeds)} for mi,mode in enumerate(modes)})
                entries.append(entry)
            cases[name]=dict(gallery=1701,eligible_bucket_count=len(indices),sampled=entries)
        run.save('descriptive_cases.json',dict(cases=cases,seed=a.seed,
            selection='Post-outcome seeded sample up to3 sorted IDs per RR/raw Top1 bucket on fixed gallery1701; all modes/seeds shown',
            limitation='Metadata/ranking cases only; no video or linguistic/human error interpretation and no selection-adjusted inference'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4.2))
        means=[report['mean_over_fixed_galleries'][mode]['micro']['dcg'] for mode in modes]
        colors=['#3377aa']*4+['#777777','#333333']
        axes[0].bar(modes,means,color=colors);axes[0].set_ylim(0,.75);axes[0].set_ylabel('Micro binary DCG (0–1)')
        axes[0].set_title('Mean of three fixed one-shot galleries')
        for i,value in enumerate(means):axes[0].text(i,value+.01,f'{value:.4f}',ha='center',fontsize=9)
        for mode,color in zip(('RR','TR','raw','centered'),('#3377aa','#aabbd0','#777777','#333333')):
            axes[1].plot(seeds,[report['metrics'][mode][str(s)]['micro']['dcg'] for s in seeds],marker='o',label=mode,color=color)
        axes[1].set_xticks(seeds);axes[1].set_ylim(0,.75);axes[1].set_ylabel('Micro binary DCG (0–1)');axes[1].legend(fontsize=8)
        axes[1].set_title('Every gallery: RR beats TR, trails raw/centered')
        fig.suptitle('Adapted MS-ASL validation: 748 queries, 100 classes; frozen SignRep')
        fig.text(.5,.01,'Primary RR−TR +0.0393; conditional 19-signer interval [0.0237,0.0627]. No method superiority or official benchmark claim.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.06,1,.93))
        for extension in ('png','pdf'):
            buf=io.BytesIO();fig.savefig(buf,format=extension,dpi=150)
            run.store.write_bytes('geometry_comparison.'+extension,buf.getvalue())
        plt.close(fig)
        rrmean=means[0];delta_raw=rrmean-means[4];delta_centered=rrmean-means[5]
        run.save('report_summary.json',dict(passed_descriptive_export=True,table_rows=len(rows),primary=report['primary'],
            RR_minus_raw_DCG=delta_raw,RR_minus_centered_DCG=delta_centered,
            RR_trails_raw_every_gallery=all(report['metrics']['RR'][str(s)]['micro']['dcg']<report['metrics']['raw'][str(s)]['micro']['dcg'] for s in seeds),
            RR_trails_centered_every_gallery=all(report['metrics']['RR'][str(s)]['micro']['dcg']<report['metrics']['centered'][str(s)]['micro']['dcg'] for s in seeds),
            decision='Retain fixed orientation-transfer analysis; do not scale or promote residual whitening as an improved downstream method',
            secondary='Descriptive differences only; no extra intervals, significance, baseline tuning or winner adoption',
            stores=stores,GPU_hours=0,new_encoder_calls=0))


if __name__=='__main__':main()
