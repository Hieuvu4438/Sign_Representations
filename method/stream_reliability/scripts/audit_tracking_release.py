"""Bounded official tracking annotation schema and local train/dev ID feasibility."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'method/frozen_probe'));sys.path.insert(0,str(ROOT))
from cache import file_sha
from resources import scratch_preflight
from run_context import RunContext


def main():
    p=argparse.ArgumentParser()
    for n in ('config','output-dir'):p.add_argument('--'+n,required=True,type=Path)
    p.add_argument('--seed',required=True,type=int);p.add_argument('--run-id',required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    with RunContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        for path,sha in cfg['input_hashes'].items():
            if file_sha(REPO/path)!=sha:raise ValueError('Registered metadata/own code changed')
        scratch_preflight('/dev/shm',cfg['max_expanded_bytes']+cfg['max_download_bytes'])
        work=Path('/dev/shm')/('shubert_tracking_release_'+a.run_id);work.mkdir(exist_ok=False)
        request=urllib.request.Request(cfg['url'],headers={'User-Agent':'SignRepresentations-research/1.0'})
        with urllib.request.urlopen(request,timeout=40) as response:
            data=response.read(cfg['max_download_bytes']+1)
            retrieval=dict(final_url=response.url,status=response.status,ETag=response.headers.get('ETag'),Last_Modified=response.headers.get('Last-Modified'),Content_Type=response.headers.get('Content-Type'))
        if len(data)>cfg['max_download_bytes']:raise ValueError('Annotation download cap exceeded')
        target=work/'official_tracking.xml.gz';target.write_bytes(data)
        with gzip.open(target,'rb') as f:expanded=f.read(cfg['max_expanded_bytes']+1)
        if len(expanded)>cfg['max_expanded_bytes']:raise ValueError('Expanded annotation cap exceeded')
        if b'<!DOCTYPE' in expanded or b'<!ENTITY' in expanded:raise ValueError('Unsupported XML entity/doctype')
        xml=ET.fromstring(expanded)
        tag_counts=Counter(e.tag for e in xml.iter())
        attribute_keys={tag:sorted(set(k for e in xml.iter(tag) for k in e.attrib)) for tag in tag_counts}
        # Inspect identifiers/schema only. Values of manual point coordinates are
        # not exposed or used to select a method/cohort or compute a metric.
        ids={v for e in xml.iter() for k,v in e.attrib.items() if k.lower() in ('id','name','sentence','sentence-id','sentence_id','video','recording')}
        metadata={}
        for split,path in cfg['local_metadata'].items():
            rows=list(csv.DictReader(open(path),delimiter='|'))
            names={row['name'] for row in rows}
            exact=sorted(names&ids)
            metadata[split]=dict(path=path,sha256=file_sha(path),rows=len(rows),exact_identifier_matches=len(exact),
                matched_ids=exact,frame_directory_count=sum((Path(cfg['local_frames'])/split/key/'1').is_dir() for key in exact))
        report=dict(run_id=a.run_id,GPU_hours=0,release_url=cfg['url'],retrieval=retrieval,
            payload_path=str(target),payload_bytes=len(data),payload_sha256=file_sha(target),expanded_bytes=len(expanded),
            expanded_sha256=hashlib.sha256(expanded).hexdigest(),root_tag=xml.tag,tag_counts=dict(tag_counts),attribute_keys=attribute_keys,
            root_children=[dict(tag=e.tag,attribute_keys=sorted(e.attrib)) for e in list(xml)[:4]],
            identifier_count=len(ids),local_metadata=metadata,
            ground_truth_definition='Official webpage: left/right palm centers and nose tip; actual schema and coordinate/image mapping still must be audited',
            known_language='DGS; does not supply ASL_Citizen natural-failure labels',
            metrics_computed=False,coordinate_values_used=False,new_test_quality_access=False,
            scientific_pilot_authorized=False,scope='Metadata/schema/join feasibility only, no detector run or recognition/tracking metric')
        run.save('tracking_release_audit.json',report)
        run.status['output_hashes']={'tracking_release_audit.json':file_sha(run.output/'tracking_release_audit.json')}


if __name__=='__main__':main()
