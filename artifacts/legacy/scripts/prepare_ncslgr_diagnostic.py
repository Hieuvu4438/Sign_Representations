"""Prepare lossless cue-preserving NCSLGR inputs; lock an internal protocol.

Class labels are positively recorded functional tiers, not text-derived
semantics or inferred affirmative/absent labels. Burned-in metadata is cropped
using the observed fixed historical 324x312 template (top324x240 image).
"""
import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from _common import ROOT
from signrepr.io import read_jsonl, sha256, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/external/ncslgr/diagnostic_v1')
    args = parser.parse_args()
    cohort = ROOT / 'data/external/ncslgr/readiness_decoded.jsonl'
    rows = [r for _, r in read_jsonl(cohort)]
    corrections = {r['sample_id']: r for r in json.loads(
        (ROOT / 'evidence/ncslgr/lana_frontal_heads.json').read_text())}
    registry = json.loads((ROOT / 'provenance/assets.json').read_text())['assets']
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'protocol.lock.json').exists():
        raise ValueError('Diagnostic already locked; do not overwrite')
    manifest, audited, failures = [], [], []
    impl_hash = sha256(Path(__file__))
    for row in rows:
        try:
            if row['media_audit_status'] != 'PASS_METADATA_TIMING':
                raise ValueError('Original views have unresolved media timing')
            specs = []
            for v in row['decoded_views']:
                if v['view_index'] == '0' and row['sample_id'] in corrections:
                    c = corrections[row['sample_id']]
                    specs.append({'kind': 'body', 'camera_id': '1',
                                  'asset_id': c['asset_id'],
                                  'sequence_id': c['index_row']['Video sequence id']})
                else:
                    index_row = next(r for r in row['views'] if r['Perspective/Camera id'] == v['view_index'])
                    specs.append({'kind': 'body' if v['view_index'] == '0' else 'face',
                                  'camera_id': v['view_index'], 'asset_id': v['asset_id'],
                                  'sequence_id': index_row['Video sequence id']})
            records = []
            for spec in specs:
                asset = registry[spec['asset_id']]
                source = ROOT / asset['local_path']
                provenance = json.loads((ROOT / 'provenance' / (spec['asset_id'] + '.json')).read_text())
                source_hash = sha256(source)
                if source_hash != provenance['sha256']:
                    raise ValueError('Source acquisition hash mismatch')
                clock = json.loads(subprocess.check_output([
                    'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                    'packet=pts_time', '-of', 'json', str(source)]))
                pts = np.asarray([float(p['pts_time']) for p in clock['packets']])
                if len(pts) == 0 or np.max(np.abs(pts - np.arange(len(pts))/30)) > 1e-6:
                    raise ValueError('Source frame PTS differ from validated30FPS clock')
                sample_id = row['sample_id'] + ':' + spec['kind']
                key = hashlib.sha256((sample_id + source_hash + impl_hash).encode()).hexdigest()
                target = args.output / (key + '.avi')
                if target.exists():
                    raise ValueError('Prepared target exists; resume not implemented, refusing replacement')
                cap = cv2.VideoCapture(str(source))
                writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'FFV1'), 30, (324,240))
                if not writer.isOpened():
                    raise ValueError('FFV1 writer unavailable')
                source_roi_hash = hashlib.sha256()
                n = 0
                try:
                    while True:
                        valid, bgr = cap.read()
                        if not valid:
                            break
                        if bgr.shape != (312,324,3):
                            raise ValueError('Historical input geometry differs')
                        roi = np.ascontiguousarray(bgr[:240])
                        source_roi_hash.update(roi.tobytes())
                        writer.write(roi)
                        n += 1
                finally:
                    cap.release(); writer.release()
                out = cv2.VideoCapture(str(target))
                output_roi_hash = hashlib.sha256()
                output_frames = 0
                output_fps = float(out.get(cv2.CAP_PROP_FPS))
                try:
                    while True:
                        valid, bgr = out.read()
                        if not valid:
                            break
                        output_roi_hash.update(bgr.tobytes())
                        output_frames += 1
                finally:
                    out.release()
                if (n != len(pts) or n != output_frames or abs(output_fps-30)>1e-6
                        or source_roi_hash.digest() != output_roi_hash.digest()):
                    raise ValueError('Lossless ROI/frame/FPS verification failed')
                meta = {'sample_id': sample_id, 'source_asset': spec['asset_id'],
                        'source_sha256': source_hash, 'prepared_sha256': sha256(target),
                        'annotation_sha256': row['annotation_sha256'],
                        'annotation_source': row['annotation_source'],
                        'roi_xywh': [0,0,324,240], 'roi_lossless_verified': True,
                        'frames': n, 'fps':30.0, 'source_camera_id':spec['camera_id'],
                        'view_role':spec['kind'], 'implementation_sha256':impl_hash,
                        'clock_policy':'All source frame-start PTS verified index/30. Output has same frames/start clock; final packet duration normalized to1/30.'}
                metadata_path = target.with_suffix('.metadata.json')
                write_json(metadata_path,meta)
                record = {'sample_id':sample_id,'utterance_sample_id':row['sample_id'],
                          'source_root':str(args.output.resolve()),'relative_path':target.name,
                          'source_exists':True,'metadata_source':str(metadata_path.resolve()),
                          'input_kind':'video','start_sec':None,'end_sec':None,
                          'label':row['label'],'split':row['split'],'signer':row['signer'],
                          'view_role':spec['kind'],'source_camera_id':spec['camera_id'],
                          'recording_group':'ncslgr:sequence:'+spec['sequence_id'],
                          'conservative_xml_group':row['recording_group'],
                          'utterance_start_ms':row['start_ms'],'utterance_end_ms':row['end_ms'],
                          'functional_core_intervals':row['functional_core_intervals']}
                records.append(record)
            if len({r['recording_group'] for r in records}) != 1:
                raise ValueError('Paired views have different official source sequence IDs')
            if len({json.loads(Path(r['metadata_source']).read_text())['frames'] for r in records}) != 1:
                raise ValueError('Corrected paired views have different frame counts')
            manifest.extend(records)
            audited.append({'sample_id':row['sample_id'],'status':'PASS'})
        except Exception as error:
            failures.append({'sample_id':row['sample_id'],'status':'FAIL','error':type(error).__name__+': '+str(error)})
        print(json.dumps({'sample_id':row['sample_id'],'success_utterances':len(audited),'failed_utterances':len(failures)}),flush=True)
    counts = defaultdict(Counter)
    groups = defaultdict(set)
    signers = defaultdict(set)
    for r in manifest:
        if r['view_role']=='body':counts[r['split']][r['label']]+=1
        groups[r['recording_group']].add(r['split'])
        signers[r['signer']].add(r['split'])
    if any(len(v)>1 for v in groups.values()) or any(len(v)>1 for v in signers.values()):
        raise ValueError('Source sequence or signer overlap; do not lock protocol')
    path=args.output/'manifest.jsonl';write_jsonl(path,manifest)
    protocol = {'status':'LOCKED_FOR_INTERNAL_DIAGNOSTIC_WITH_LIMITS' if not failures else 'INCOMPLETE',
                'cohort_sha256':sha256(cohort),'manifest_sha256':sha256(path),
                'input_preparation_sha256':impl_hash,'samples':len(rows),
                'usable_utterances':len(audited),'counts':{k:dict(v) for k,v in counts.items()},
                'primary_metric':'macro_recall','classes':['NEG','WH','YN'],
                'target_semantics':'One positively recorded functional target tier among NEG/WH/YN; exclude target cooccurrences. No inferred semantic affirmative or fully absent class.',
                'split_policy':'Source XML participant NAME held out; two train signers, one validation signer, one test signer. Official sequence ID groups views; no source sequence or signer crosses splits.',
                'signer_assignment':{s:next(iter(ss)) for s,ss in signers.items()},
                'readouts':['prior','body_mean_linear','face_mean_linear','paired_normalized_mean_linear'],
                'feature_policy':'Frozen SignRep, public16-frame native features; mean pool only window centers inside utterance interval. Same768-dimensional linear readout for body/face/paired mean.',
                'lr_grid':[0.01,0.03,0.1],'training_seeds':[0,1,2],'epochs':100,
                'optimizer':'AdamW','weight_decay':0.0001,
                'head_selection':'Validation macro recall only, deterministic first-candidate tie break. No best-seed headline.',
                'external_MTP':'Not used for training/tuning; access/canonical exclusions remain pending.',
                'limits':['Historical research mirror annotation release; not certified as current2011 XML bytes.',
                          'View resolution, color and background differ; this comparison does not isolate facial information causally.',
                          'Validation has only2YN and4NEG examples; head selection unstable.',
                          'One test signer; no signer-population inference.',
                          'Four shared XML archive groups remain as conservative grouping sensitivity; no claim of clean pretraining/external transfer.',
                          'No gold scope or candidate translation used as model inputs; no method novelty claimed.'],
                'failures':failures}
    write_json(args.output/'protocol.lock.json',protocol)
    write_json(ROOT/'reports/ncslgr_input_preparation.json',protocol)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
