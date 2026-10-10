"""Export linguistic-quality/signer-leakage views of the frozen pilot."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'signrel/config.py').exists())
sys.path.insert(0,str(ROOT))
from signrel.data.manifest import sha256
try:
    from signrel.eval.pilot_summary import metrics_from_report
except ImportError:
    spec = importlib.util.spec_from_file_location('pending_summary',ROOT/'pending_source/signrel/eval/pilot_summary.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    metrics_from_report = module.metrics_from_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--summary',required=True)
    p.add_argument('--baseline',action='append',nargs=2,metavar=('LABEL','REPORT'),default=[])
    p.add_argument('--out',required=True)
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = json.loads(Path(args.summary).read_text())
    fig,axes = plt.subplots(1,2,figsize=(11,4.5))
    colors = {'A_generic':'#4c78a8','B_articulator':'#f58518','C_relation':'#54a24b','D_combined':'#b279a2'}
    for name,values in summary['summary'].items():
        signer = values['signer_accuracy']
        if signer['mean'] is None:
            continue
        for ax,metric in zip(axes,('Recall@1','phonology_macro_f1')):
            value = values[metric]
            ax.errorbar(value['mean'],signer['mean'],xerr=value['std'],yerr=signer['std'],marker='o',
                        color=colors.get(name),capsize=3,label=f'{name} (n={value["seeds"]})')
    sources = []
    for label,path in args.baseline:
        report = json.loads(Path(path).read_text())
        values = metrics_from_report(report)
        sources.append({'label':label,'report':path,'sha256':sha256(path),'comparison':'external frozen reference; different pretraining/protocol may confound causal comparison'})
        for ax,metric in zip(axes,('Recall@1','phonology_macro_f1')):
            if values['signer_accuracy'] is not None:
                ax.scatter(values[metric],values['signer_accuracy'],marker='*',s=100,label=label)
    for ax,xlabel in zip(axes,('Frozen retrieval Recall@1','Signer-independent phonology macro-F1')):
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Known-signer probe accuracy')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1.03)
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=7,loc='lower right')
    fig.suptitle(f'ASL Citizen 40-gloss pilot ({summary["split"]}); mean ± training-seed SD',fontsize=11)
    fig.text(.5,.015,'Lower signer accuracy is useful only when linguistic performance is maintained. External stars are reference baselines.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.95))
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out.with_suffix('.png'),dpi=180)
    fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)
    out.with_suffix('.json').write_text(json.dumps({'summary_sha256':sha256(args.summary),'script_sha256':sha256(__file__),
                    'baselines':sources,'scope':summary['scope'],'decision':summary['decision']['status']},indent=2))


if __name__ == '__main__':
    main()
