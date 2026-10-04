"""Pinned public source-list coverage, not actual checkpoint training membership."""
import argparse
import base64
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from urllib.request import Request,urlopen
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
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
        with urlopen(Request(c['public_url'],headers={'User-Agent':'SignRep-research-provenance-audit'}),timeout=20) as response:
            data=response.read(c['max_bytes']+1)
            headers={k:response.headers.get(k) for k in ('ETag','Last-Modified','Content-Length','x-goog-generation','x-goog-hash')}
            status=response.status
        if (len(data)!=c['expected_bytes'] or len(data)>c['max_bytes'] or
                base64.b64encode(hashlib.md5(data).digest()).decode()!=c['expected_md5_base64'] or
                headers['x-goog-generation']!=c['expected_generation']):
            raise ValueError('Public manifest bytes/generation mismatch')
        path=run.output/'youtube-sl-25-metadata.csv'
        path.write_bytes(data)
        reader=csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
        fields=reader.fieldnames
        rows=list(reader)
        candidates=[f for f in fields if re.sub(r'[^a-z0-9]','',f.lower()) in ('videoid','youtubeid','youtubevideoid')]
        if len(candidates)!=1:raise ValueError('Unrecognized source-ID field '+str(fields))
        id_field=candidates[0]
        ids=[r[id_field].strip() for r in rows]
        if any(not re.fullmatch(r'[A-Za-z0-9_-]{11}',k) for k in ids):
            raise ValueError('Invalid public YouTube source ID')
        local=[s.strip() for s in Path(c['local_youtube_asl_ids']).read_text().splitlines()]
        if any(not re.fullmatch(r'[A-Za-z0-9_-]{11}',k) for k in local):
            raise ValueError('Invalid local YouTube-ASL source ID')
        # Literal field values only; do not turn language tags/captions into gloss labels.
        other={f:dict(Counter(r[f] for r in rows)) for f in fields if f!=id_field}
        if any(len(counts)>1000 for counts in other.values()):
            other={f:dict(value_count=len(counts)) for f,counts in other.items()}
        report=dict(public_url=c['public_url'],http_status=status,headers=headers,
                    source_bytes=len(data),source_sha256=file_sha(path),source_generation=c['expected_generation'],
                    source_fields=fields,source_id_field=id_field,row_count=len(rows),unique_source_ids=len(set(ids)),
                    duplicate_id_rows=len(ids)-len(set(ids)),other_field_literal_counts=other,
                    local_YouTube_ASL_unique_ids=len(set(local)),local_ids_present_in_public_pool=len(set(local)&set(ids)),
                    local_ids_absent_from_public_pool=len(set(local)-set(ids)),
                    public_ids_absent_from_local_ASL_list=len(set(ids)-set(local)),
                    checkpoint_specific_filtered_training_list_available=False,
                    ASL_Citizen_source_join_available=False,ASL_Citizen_pretraining_disjointness='unknown',
                    interpretation='public corpus source pool coverage only; local list coverage is not checkpoint training membership, media-byte equality, signer or isolated-gloss class exclusion',
                    GPU_hours=0,new_encoder_calls=0,test_feature_or_quality_access='none')
        run.save('manifest_audit.json',report)
        run.status['output_hashes']={name:file_sha(run.output/name) for name in ('youtube-sl-25-metadata.csv','manifest_audit.json')}


if __name__=='__main__':main()
