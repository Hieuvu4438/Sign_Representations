"""Bounded author-space metadata and local SHA equality; no model download."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'method/frozen_probe'))
sys.path.insert(0,str(ROOT))
from cache import file_sha
from run_context import RunContext


def main():
    parser=argparse.ArgumentParser()
    for name in ('config','output-dir'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--seed',required=True,type=int)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    with RunContext(args.output_dir,args.run_id,args.config,args.seed) as run:
        if 'author_source' in config:
            source=REPO/config['author_source']['local_path']
            if file_sha(source)!=config['author_source']['sha256']:raise ValueError('Author asset-reference source changed')
            (run.output/'author_source_snapshot.py').write_bytes(source.read_bytes())
        request=urllib.request.Request(config['metadata_url'],headers={'Accept':'application/json','Accept-Encoding':'identity'})
        with urllib.request.urlopen(request,timeout=30) as response:
            data=response.read(config['max_response_bytes']+1)
            if len(data)>config['max_response_bytes']:raise ValueError('Bounded release metadata cap exceeded')
            headers=dict(response.headers);status=response.status
        members=json.loads(data)
        if not isinstance(members,list):raise ValueError('Expected author-space file tree')
        run.save('remote_tree.json',members)
        results=[]
        for asset in config['assets']:
            matches=[m for m in members if m['path']==asset['remote_path']]
            if len(matches)!=1 or 'lfs' not in matches[0]:raise ValueError('Missing/ambiguous published LFS metadata')
            remote=matches[0];lfs=remote['lfs'];digest=lfs['oid'].removeprefix('sha256:')
            path=REPO/asset['local_path'];local=file_sha(path)
            if local!=asset['expected_local_sha256']:raise ValueError('Local asset changed from recorded inventory')
            results.append(dict(local_path=asset['local_path'],remote_path=asset['remote_path'],
                 local_bytes=path.stat().st_size,remote_bytes=lfs['size'],local_sha256=local,remote_sha256=digest,
                 equal=local==digest and path.stat().st_size==lfs['size']))
        run.save('asset_equality.json',dict(run_id=args.run_id,accessed_utc=datetime.now(timezone.utc).isoformat(),
             metadata_url=config['metadata_url'],http_status=status,http_headers=headers,response_bytes=len(data),
             response_sha256=hashlib.sha256(data).hexdigest(),assets=results,all_equal=all(a['equal'] for a in results),
             source='author project linked demo code points to author model repository; not GoogleDrive byte comparison',
             author_source=config.get('author_source'),
             scope='published author asset equality only; no model load/parity/exclusion membership or task score'))
        run.status['output_hashes']={name:file_sha(run.output/name) for name in ('remote_tree.json','asset_equality.json')}


if __name__=='__main__':main()
