"""Candidate filename normalization is not a join; bounded official pixel check."""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import zlib

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
            if file_sha(REPO/path)!=sha:raise ValueError('Bound annotation/metadata/code changed')
        source=json.loads((REPO/cfg['source_report']).read_text())
        if file_sha(source['payload_path'])!=source['payload_sha256']:raise ValueError('Official XML payload changed')
        xml=ET.fromstring(gzip.open(source['payload_path'],'rb').read())
        local={};rows=[]
        for split,path in cfg['local_metadata'].items():
            for row in csv.DictReader(open(path),delimiter='|'):
                if row['name'] in local:raise ValueError('Local split duplicate ID')
                local[row['name']]=split
        for video in xml.findall('video'):
            original=video.attrib['name'];candidate=original.replace('_default-', '-',1)
            frames=[int(row.attrib['number']) for row in video.findall('frame')]
            split=local.get(candidate);folder=Path(cfg['local_frames'])/str(split)/candidate/'1'
            files=sorted(folder.glob('*.png')) if split else []
            rows.append(dict(official_id=original,candidate_id=candidate,local_split=split,
                annotation_frames=len(frames),number_min=min(frames),number_max=max(frames),local_frames=len(files),
                candidate_frame_count_equal=bool(files and len(files)==len(frames)),
                candidate_only=True,identity_accepted=False))
        scratch_preflight('/dev/shm',cfg['range_bytes']+cfg['max_expanded_bytes'])
        store=Path('/dev/shm')/('shubert_tracking_join_'+a.run_id);store.mkdir(exist_ok=False)
        request=urllib.request.Request(cfg['archive_url'],headers={'Range':f"bytes=0-{cfg['range_bytes']-1}",'If-Range':cfg['archive_etag'],'User-Agent':'SignRepresentations-research/1.0'})
        with urllib.request.urlopen(request,timeout=40) as response:
            expected=f"bytes 0-{cfg['range_bytes']-1}/{cfg['archive_size']}"
            if response.status!=206 or response.headers.get('Content-Range')!=expected or response.headers.get('ETag')!=cfg['archive_etag']:
                raise ValueError('Exact bounded206required; refuse full archive')
            compressed=response.read(cfg['range_bytes']+1)
            retrieval=dict(status=response.status,Content_Range=response.headers.get('Content-Range'),ETag=response.headers.get('ETag'),final_url=response.url)
        if len(compressed)!=cfg['range_bytes']:raise ValueError('Range byte length differs')
        prefix=store/'official_prefix.gz';prefix.write_bytes(compressed)
        decoder=zlib.decompressobj(16+zlib.MAX_WBITS)
        expanded=decoder.decompress(compressed,cfg['max_expanded_bytes'])
        examined=[];complete_images=[]
        # The archive itself is deliberately incomplete. Parse only members whose
        # complete payload is present, never extract paths or guess frame offsets.
        try:
            with tarfile.open(fileobj=io.BytesIO(expanded),mode='r|') as archive:
                for member in archive:
                    if len(examined)>=cfg['max_members']:break
                    examined.append(dict(path=member.name,size=member.size,is_file=member.isfile()))
                    if not member.isfile() or PurePosixPath(member.name).suffix.lower() not in ('.png','.jpg','.jpeg'):continue
                    if member.size>cfg['max_image_bytes']:raise ValueError('Archive image cap exceeded')
                    if member.offset_data+member.size>len(expanded):break
                    pixels=archive.extractfile(member).read()
                    import hashlib
                    image_sha=hashlib.sha256(pixels).hexdigest();parts=PurePosixPath(member.name).parts
                    known=next((row for row in rows if row['official_id'] in parts or row['candidate_id'] in parts),None)
                    comparison=dict(path=member.name,sha256=image_sha,bytes=len(pixels),candidate_local_path=None,byte_equal=None)
                    if known and known['local_split']:
                        path=Path(cfg['local_frames'])/known['local_split']/known['candidate_id']/'1'/PurePosixPath(member.name).name
                        comparison['candidate_local_path']=str(path)
                        if path.is_file():
                            comparison.update(local_sha256=file_sha(path),byte_equal=file_sha(path)==image_sha)
                            import cv2,numpy as np
                            official=cv2.imdecode(np.frombuffer(pixels,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
                            native=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
                            comparison.update(official_shape=list(official.shape),local_shape=list(native.shape),
                                decoded_pixel_equal=bool(official.shape==native.shape and np.array_equal(official,native)))
                    complete_images.append(comparison)
                    if len(complete_images)>=cfg['max_images']:break
        except tarfile.ReadError:
            # Expected only for deliberately incomplete archive prefix, preserved
            # in report; no metric or identity is accepted from truncated payload.
            if not examined:raise
        report=dict(run_id=a.run_id,GPU_hours=0,candidates=rows,
            normalization='_default- to - exactly once, filename screening hypothesis only; no official mapping assumed',
            candidate_counts={s:sum(row['local_split']==s for row in rows) for s in ('train','dev')},
            equal_frame_count_candidates=sum(row['candidate_frame_count_equal'] for row in rows),
            archive_url=cfg['archive_url'],retrieval=retrieval,prefix_path=str(prefix),prefix_sha256=file_sha(prefix),
            prefix_bytes=len(compressed),expanded_prefix_bytes=len(expanded),gzip_end_reached=decoder.eof,
            archive_prefix_only=True,examined_members=examined,complete_images=complete_images,
            any_identity_accepted=False,metrics_computed=False,coordinate_values_used=False,scientific_pilot_authorized=False,
            scope='Train/dev metadata candidates and bounded source pixel identity probe; no inferred offset, frame alignment or ground-truth join accepted')
        run.save('tracking_join_audit.json',report);run.status['output_hashes']={'tracking_join_audit.json':file_sha(run.output/'tracking_join_audit.json')}


if __name__=='__main__':main()
