"""Saved handedness category scores are not exported localization reliability."""
import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'method/frozen_probe'));sys.path.insert(0,str(ROOT))
from cache import file_sha
from run_context import RunContext


def main():
    p=argparse.ArgumentParser()
    for n in ('config','output-dir'):p.add_argument('--'+n,required=True,type=Path)
    p.add_argument('--seed',required=True,type=int);p.add_argument('--run-id',required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    with RunContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        for path,sha in cfg['input_hashes'].items():
            if file_sha(REPO/path)!=sha:raise ValueError('Bound raw observation/source changed')
        landmarks=json.loads((REPO/cfg['landmark_report']).read_text());inputs=json.loads((REPO/cfg['input_report']).read_text())
        samples=[];all_duplicate_scores=[]
        for source,aligned in zip(landmarks['samples'],inputs['samples']):
            if source['sample_id']!=aligned['sample_id']:raise ValueError('Fixed source order changed')
            if file_sha(source['raw_path'])!=source['raw_sha256'] or file_sha(aligned['metadata_path'])!=aligned['metadata_sha256']:
                raise ValueError('Raw coordinates/mask metadata changed')
            rows=json.load(gzip.open(source['raw_path'],'rt'))['frames'];metadata=json.loads(Path(aligned['metadata_path']).read_text())
            if len(rows)!=source['frames'] or len(metadata)!=len(rows):raise ValueError('Native alignment differs')
            scores=[];duplicated=[];counts={label:0 for label in ('Left','Right')}
            for row,meta in zip(rows,metadata):
                hands=row['landmarks']['hand_landmarks'] or []
                if len(hands)!=len(row['handedness']):raise ValueError('Hand-category alignment differs')
                categories=[]
                for values in row['handedness']:
                    if len(values)!=1:raise ValueError('Expected one top-handedness category per detection')
                    c=values[0]
                    if c['label'] not in counts or not 0<=c['score']<=1:raise ValueError('Handedness category contract differs')
                    categories.append(c);scores.append(c['score']);counts[c['label']]+=1
                if meta['same_detection']:
                    index=meta['left_detection_index']
                    if index!=meta['right_detection_index']:raise ValueError('Shared detection metadata differs')
                    c=categories[index];duplicated.append(dict(frame_index=row['frame_index'],detector_index=index,handedness_label=c['label'],handedness_score=c['score']))
            all_duplicate_scores.extend(x['handedness_score'] for x in duplicated)
            samples.append(dict(sample_id=source['sample_id'],frames=len(rows),detected_hand_categories=len(scores),labels=counts,
                handedness_score_min=min(scores),handedness_score_max=max(scores),duplicate_assignment_categories=duplicated,
                recorded_landmark_fields=list(rows[0]['landmarks']),landmark_vector_width=3,
                exported_localization_confidence=False,exported_pose_visibility=False))
        report=dict(run_id=a.run_id,GPU_hours=0,samples=samples,total_frames=sum(x['frames'] for x in samples),
            duplicate_assignment_count=len(all_duplicate_scores),duplicate_handedness_min=min(all_duplicate_scores),duplicate_handedness_max=max(all_duplicate_scores),
            semantic_binding='Installed HandLandmarkerResult.handedness is handedness classification; its Category.score is not a localization/visibility/tracking quality probability',
            observation_binding='Saved raw landmark vectors contain x/y/z only; upstream drops pose visibility and exports no hand detection/presence/tracking confidence score',
            distinction='Configuration confidence thresholds are detector controls; returned handedness score cannot be substituted for them or for manual crop correctness',
            localization_error_metric_computed=False,quality_calibration_verified=False,scientific_pilot_authorized=False,
            scope='Four fixed raw observation/source semantics audit only; duplicate assignment is structural source reuse, not annotated true tracking failure')
        run.save('confidence_semantics.json',report);run.status['output_hashes']={'confidence_semantics.json':file_sha(run.output/'confidence_semantics.json')}


if __name__=='__main__':main()
