"""Fetch the pinned WLASL annotation/trim provenance into owned data storage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

PROJECT=Path(__file__).resolve().parents[1]
REVISION='ac00e6be631c1a2a486621b65f202219f3964d6b'
ASSETS={
    'WLASL_v0.3.json':('WLASL_v0.3_ac00e6be.json','31ba5a5c421bdbff87494ee22ca6d4e276a8dfac44f43cc9a58b76ed1c4cd684'),
    'preprocess.py':('WLASL_preprocess_ac00e6be.py','41237014ac51117fdd3ad125b6a9136046046caab7e21015d2ed37ad0f6bd737'),
}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(PROJECT/'data/source_metadata'))
    args=parser.parse_args();root=Path(args.output).resolve()
    if PROJECT not in root.parents:raise ValueError('metadata output must remain inside this method')
    root.mkdir(parents=True,exist_ok=True);reports=[]
    for remote,(local,expected) in ASSETS.items():
        path=root/local;url=f'https://raw.githubusercontent.com/dxli94/WLASL/{REVISION}/start_kit/{remote}'
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError(f'existing metadata differs from pinned content: {path}')
            action='verified_existing'
        else:
            with urllib.request.urlopen(url,timeout=60) as response:content=response.read()
            if hashlib.sha256(content).hexdigest()!=expected:raise ValueError('upstream content hash differs')
            fd,temporary=tempfile.mkstemp(dir=root,prefix=local+'.')
            try:
                with os.fdopen(fd,'wb') as file:file.write(content);file.flush();os.fsync(file.fileno())
                os.replace(temporary,path)
            finally:
                if os.path.exists(temporary):os.unlink(temporary)
            action='downloaded_and_verified'
        reports.append({'url':url,'revision':REVISION,'path':str(path),'sha256':expected,'action':action})
    print(json.dumps({'status':'passed','assets':reports}))


if __name__=='__main__':main()
