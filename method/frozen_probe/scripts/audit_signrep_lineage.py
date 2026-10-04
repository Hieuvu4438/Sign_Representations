"""Bounded release/checkpoint/source-identifier audit; no feature/task scoring."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0,str(ROOT))
from cache import file_sha
from run_context import RunContext


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--run-id',required=True)
    p.add_argument('--seed',required=True,type=int)
    a=p.parse_args();c=json.loads(a.config.read_text())
    with RunContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        for name,digest in c['input_hashes'].items():
            if file_sha(REPO/name)!=digest:raise ValueError('Changed input '+name)
        import torch
        path=REPO/c['checkpoint']
        if file_sha(path)!=c['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        checkpoint=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        if not isinstance(checkpoint,dict):raise ValueError('Unexpected checkpoint root')
        summary={}
        for key,value in checkpoint.items():
            item={'type':type(value).__name__}
            if isinstance(value,dict):item.update(key_count=len(value),keys=list(map(str,value)) if key!='model' else None)
            elif isinstance(value,(int,float,str,bool,type(None))):item['value']=value if not isinstance(value,str) or len(value)<1000 else 'long string omitted'
            elif isinstance(value,(list,tuple)):item['length']=len(value)
            summary[str(key)]=item
        model=checkpoint.get('model',{})
        tensor_count=sum(isinstance(v,torch.Tensor) for v in model.values())
        if torch.cuda.is_initialized():raise ValueError('Unexpected CUDA')
        upstream=REPO/'third_party/SignRep'
        tracked=subprocess.check_output(['git','-C',str(upstream),'ls-tree','-r','--name-only',c['upstream_commit']],text=True).splitlines()
        cohort_samples=[]
        for name in c['cohorts']:
            cohort_samples.extend(json.loads((REPO/name).read_text())['samples'])
        if any(s['split'] not in ('train','val') for s in cohort_samples):raise ValueError('Unexpected cohort split')
        all_one=json.loads(Path(c['asl_metadata']).read_text())
        joined=[all_one[k['sample_id'].removesuffix('.mp4')] for k in cohort_samples if k['sample_id'].removesuffix('.mp4') in all_one]
        fields=sorted({k for row in joined for k in row})
        local_ids=Path(c['local_youtube_asl_ids']).read_text().splitlines()
        valid_ids=[x.strip() for x in local_ids if re.fullmatch(r'[A-Za-z0-9_-]{11}',x.strip())]
        evidence={}
        responses={}
        for name,url in c['public_metadata_urls'].items():
            record={'url':url,'max_response_bytes':c['max_response_bytes']}
            try:
                with urlopen(Request(url,headers={'User-Agent':'SignRep-research-provenance-audit','Accept':'application/json'}),timeout=15) as response:
                    data=response.read(c['max_response_bytes']+1)
                    record.update(http_status=response.status,headers={k:response.headers.get(k) for k in ('ETag','Last-Modified','Content-Type')})
                if len(data)>c['max_response_bytes']:raise ValueError('Bounded metadata response exceeded cap')
                record['bytes']=len(data)
                (run.output/(name+'.json')).write_bytes(data)
                record['sha256']=file_sha(run.output/(name+'.json'))
                responses[name]=json.loads(data)
            except (HTTPError,URLError,TimeoutError,ValueError) as error:
                record['error_type']=type(error).__name__;record['error']=str(error)
                if isinstance(error,HTTPError):record['http_status']=error.code
            evidence[name]=record
        release=responses.get('release',{})
        assets=[{'name':x['name'],'size':x['size'],'digest':x.get('digest')} for x in release.get('assets',[])]
        asset_matches=[x for x in assets if x['name']=='ckpt.pt' and x['digest']=='sha256:'+c['checkpoint_sha256'] and x['size']==path.stat().st_size]
        gcs=responses.get('youtube_sl25_objects',{})
        objects=[{'name':x['name'],'size':x.get('size'),'generation':x.get('generation'),'md5Hash':x.get('md5Hash')} for x in gcs.get('items',[])]
        report=dict(checkpoint_sha256=c['checkpoint_sha256'],checkpoint_bytes=path.stat().st_size,
                    checkpoint_top_level_metadata=summary,model_tensor_count=tensor_count,
                    upstream_commit=c['upstream_commit'],upstream_tracked_paths=tracked,
                    public_metadata=evidence,release_tag=release.get('tag_name'),release_assets=assets,
                    released_asset_match_verified=len(asset_matches)==1,
                    youtube_sl25_object_listing=objects,listing_complete='nextPageToken' not in gcs if gcs else False,
                    local_youtube_asl_id_rows=len(local_ids),local_youtube_asl_unique_valid_ids=len(set(valid_ids)),
                    selected_ASL_train_val_rows=len(cohort_samples),local_aux_metadata_joined_rows=len(joined),
                    selected_aux_metadata_fields=fields,
                    checkpoint_specific_training_membership='not verified from asset identity/parameter tensors',
                    ASL_Citizen_pretraining_byte_source_signer_class_disjointness='unknown',
                    accepted_ASL_to_YouTube_source_join=False,
                    scope='public source-ID inventory and checkpoint metadata only; no media download, task score, source or class disjointness inferred from filename mismatch',
                    GPU_hours=0,new_encoder_calls=0,test_feature_or_quality_access='none')
        run.save('lineage_audit.json',report)
        run.status['output_hashes']={x.name:file_sha(x) for x in run.output.iterdir() if x.is_file() and x.name not in ('status.json','environment.json','config.json')}


if __name__=='__main__':main()
