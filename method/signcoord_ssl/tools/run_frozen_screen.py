"""Real A0 frozen screening in an immutable source capsule, no SSL/test tuning."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from signcoord.provenance import atomic_json,execution_stamp,object_hash,sha256_file


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--gallery-manifest',required=True);parser.add_argument('--query-manifest',required=True)
    parser.add_argument('--device',default='cpu');args=parser.parse_args()
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    capsule=root/'code/src';capsule.mkdir(parents=True)
    shutil.copytree(PROJECT/'src/signcoord',capsule/'signcoord',ignore=shutil.ignore_patterns('__pycache__'))
    stamp=execution_stamp()
    copied={str(p.relative_to(capsule/'signcoord')):sha256_file(p) for p in sorted((capsule/'signcoord').rglob('*.py'))}
    if object_hash(copied)!=stamp['implementation_hash']:raise ValueError('source changed while creating run capsule')
    cfg={'arm':'A0','model':{'backbone':'hiera_base_video_adapter','initialization_mode':'generic_hiera_mae',
         'initialization_checkpoint':str(PROJECT/'runs/assets/mae_hiera_base_16x224.pth'),
         'generic_vendor_root':str(PROJECT/'vendor/hiera'),'input_frames':32,'attention_kernel':'sdpa_4d_zero_cls'},
         'train':{'device':args.device,'precision':'fp32'}}
    import yaml
    config=root/'reference.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    gallery,query=Path(args.gallery_manifest).resolve(),Path(args.query_manifest).resolve()
    labels={}
    for path in (gallery,query):
        current=json.loads(path.with_suffix('.labels.json').read_text())
        if set(labels)&set(current):raise ValueError('gallery/query sample IDs overlap')
        labels.update(current)
    atomic_json(root/'labels.json',labels)
    atomic_json(root/'run_inputs.json',{**stamp,'gallery_manifest':str(gallery),'query_manifest':str(query),
        'gallery_sha256':sha256_file(gallery),'query_sha256':sha256_file(query),
        'source_capsule':str(capsule),'full_corpus_benchmark':False,'purpose':'fixed_50_class_downstream_dev_screen',
        'test_evaluation':False,'sign_ssl_updates':0})
    env=dict(os.environ,PYTHONPATH=str(capsule),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    def run(name,*arguments):
        command=[sys.executable,'-m','signcoord.cli',*arguments,'--config',str(config)]
        atomic_json(root/'live_step.json',{'step':name,'command':command,'worker_parent_pid':os.getpid()})
        with (root/(name+'.log')).open('w') as f:
            result=subprocess.run(command,env=env,cwd=PROJECT.parents[1],stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{name} failed; see log')
        print(name+' passed',flush=True)
    run('initialize','init-reference','--output',str(root/'reference.pt'))
    run('gallery_features','extract-features','--checkpoint',str(root/'reference.pt'),'--manifest',str(gallery),'--output',str(root/'gallery.npz'))
    run('query_features','extract-features','--checkpoint',str(root/'reference.pt'),'--manifest',str(query),'--output',str(root/'queries.npz'))
    run('retrieval','eval-retrieval','--gallery',str(root/'gallery.npz'),'--queries',str(root/'queries.npz'),
        '--labels',str(root/'labels.json'),'--output',str(root/'retrieval.json'))
    result=json.loads((root/'retrieval.json').read_text())
    atomic_json(root/'screen_report.json',{**stamp,'status':'passed','arm':'A0','sign_ssl_updates':0,
        'full_corpus_benchmark':False,'scope':'fixed_metadata_selected_train_gallery_and_dev_queries',
        'metrics':result['metrics'],'retrieval_report':str(root/'retrieval.json'),
        'no_candidate_objective_comparison_yet':True,'test_evaluation':False})
    print(json.dumps(result['metrics']),flush=True)


if __name__=='__main__':main()
