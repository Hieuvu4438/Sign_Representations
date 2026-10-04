"""All-frame native encoded-crop/body parity for the explicit observation adapter."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'method/frozen_probe'));sys.path.insert(0,str(ROOT))
from cache import file_sha
from resources import scratch_preflight
from run_context import RunContext


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def main():
    parser=argparse.ArgumentParser()
    for name in ('config','output-dir'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--seed',required=True,type=int);parser.add_argument('--run-id',required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    with RunContext(args.output_dir,args.run_id,args.config,args.seed) as run:
        for name,digest in cfg['input_hashes'].items():
            if file_sha(REPO/name)!=digest:raise ValueError('Registered contract/input/source changed')
        resource=scratch_preflight('/dev/shm',cfg['max_output_bytes'])
        store=Path('/dev/shm')/('shubert_inputs_'+args.run_id);store.mkdir(exist_ok=False)
        run.status['artifact_root']=str(store);run.save('status.json',run.status)
        import numpy as np
        import cv2
        import decord
        from input_contract import AlignedNativeCrops
        cv2.setNumThreads(1)
        face=load('native_input_face',REPO/cfg['face_source'])
        hands=load('native_input_hands',REPO/cfg['hands_source'])
        body=load('native_input_body',REPO/cfg['body_source'])
        cohort=json.loads((REPO/cfg['cohort_manifest']).read_text());sources={r['sample_id']:r for r in cohort['samples']}
        landmark=json.loads((REPO/cfg['landmark_report']).read_text())
        geometry=json.loads((REPO/cfg['geometry_report']).read_text());geo={(r['sample_id'],r['frame_index']):r for r in geometry}
        report=dict(run_id=args.run_id,protocol=cfg['protocol'],GPU_hours=0,resources=resource,
                    artifact_root=str(store),samples=[],scope='CPU encoded-crop/body/mask contract only; no model/task inference',
                    native_bad_branches_parity=False,adaptations='initial None frame retained/14D body sentinel; invalid geometry fails before state commit')
        run.save('input_parity.json',report)
        for index,sample in enumerate(landmark['samples']):
            source=sources[sample['sample_id']]
            if source['split'] not in ('train','val') or file_sha(source['input_path'])!=sample['source_sha256']:raise ValueError('Fixed train/dev source changed')
            if file_sha(sample['raw_path'])!=sample['raw_sha256']:raise ValueError('Raw observations changed')
            with gzip.open(sample['raw_path'],'rt') as stream:raw=json.load(stream)['frames']
            directory=store/str(index);directory.mkdir()
            pose_dir=directory/'poses';pose_dir.mkdir()
            stem=Path(source['input_path']).stem
            pose_file=pose_dir/(stem+'_pose.json');pose_file.write_text(json.dumps({str(i):r['landmarks'] for i,r in enumerate(raw)}))
            reference=directory/'reference';own=directory/'adapted';reference.mkdir();own.mkdir()
            # All native functions receive explicit fresh owned output directories.
            face.video_holistic(source['input_path'],reference,directory/'problems.txt',pose_dir)
            hands.video_holistic(source['input_path'],reference,directory/'problems.txt',pose_dir)
            body.keypoints_to_numpy(pose_file,reference)
            reference_paths=dict(face=reference/(stem+'_face.mp4'),left_hand=reference/(stem+'_hand1.mp4'),
                                 right_hand=reference/(stem+'_hand2.mp4'),body_posture=reference/(stem+'_pose.npy'))
            adapted_paths={name:own/path.name for name,path in reference_paths.items()}
            reader=decord.VideoReader(source['input_path'],ctx=decord.cpu(0),num_threads=1)
            if len(reader)!=len(raw):raise ValueError('Raw/source native count differs')
            fps=float(reader.get_avg_fps());timestamps=reader.get_frame_timestamp(range(len(reader)))
            writers={name:cv2.VideoWriter(str(adapted_paths[name]),cv2.VideoWriter_fourcc(*'mp4v'),fps,(224,224))
                     for name in ('face','left_hand','right_hand')}
            if not all(w.isOpened() for w in writers.values()):raise ValueError('Registered native mp4v writer unavailable')
            adapter=AlignedNativeCrops(face,hands,body);metadata=[];body_values=[]
            try:
                for frame_index,row in enumerate(raw):
                    if row['frame_index']!=frame_index or abs(row['pts_start_s']-float(timestamps[frame_index,0]))>cfg['pts_tolerance_seconds']:
                        raise ValueError('Raw/frame timestamp mapping differs')
                    streams,meta=adapter.step(reader[frame_index].asnumpy(),row['landmarks'],frame_index)
                    if meta['same_detection']!=geo[(sample['sample_id'],frame_index)]['same_detection']:
                        raise ValueError('Native hand ambiguity flag differs from independent geometry audit')
                    for name,writer in writers.items():writer.write(streams[name][:,:,::-1].copy())
                    body_values.append(streams['body_posture'])
                    meta.update(pts_start_s=row['pts_start_s'],pts_end_s=row['pts_end_s'])
                    metadata.append(meta)
            finally:
                for writer in writers.values():writer.release()
            own_body=np.stack(body_values);np.save(adapted_paths['body_posture'],own_body)
            native_body=np.load(reference_paths['body_posture'],allow_pickle=False)
            if own_body.shape!=(len(reader),14) or not np.array_equal(own_body,native_body):raise ValueError('Native body array parity failed')
            media=[]
            for name in ('face','left_hand','right_hand'):
                ref_reader=decord.VideoReader(str(reference_paths[name]),ctx=decord.cpu(0),num_threads=1)
                own_reader=decord.VideoReader(str(adapted_paths[name]),ctx=decord.cpu(0),num_threads=1)
                if len(ref_reader)!=len(reader) or len(own_reader)!=len(reader):raise ValueError('Encoded crop lost/added frames; no truncation')
                maximum=0
                for i in range(len(reader)):
                    reference_pixels=ref_reader[i].asnumpy();own_pixels=own_reader[i].asnumpy()
                    if reference_pixels.shape!=(224,224,3) or not np.array_equal(reference_pixels,own_pixels):
                        raise ValueError('Exact encoded native crop pixel parity failed')
                    maximum=max(maximum,int(np.max(np.abs(reference_pixels.astype(int)-own_pixels.astype(int)))))
                media.append(dict(stream=name,frames=len(reader),shape=[len(reader),224,224,3],
                                  max_pixel_error=maximum,reference_path=str(reference_paths[name]),
                                  reference_sha256=file_sha(reference_paths[name]),adapted_path=str(adapted_paths[name]),
                                  adapted_sha256=file_sha(adapted_paths[name])))
            metadata_path=own/'observation_metadata.json';metadata_path.write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
            report['samples'].append(dict(sample_id=sample['sample_id'],source_id=sample['sample_id'],split=source['split'],
                signer_group=source['participant_id'],language='ASL',input_path=source['input_path'],source_sha256=sample['source_sha256'],
                frames=len(reader),native_avg_fps=fps,sampling_rule='all native source frames; original PTS retained in metadata',
                codec='author mp4v/OpenCV BGR writer→decord RGB at avgFPS; metadata preserves source-native timeline',
                media=media,body_shape=list(own_body.shape),body_dtype=str(own_body.dtype),
                body_reference_path=str(reference_paths['body_posture']),body_reference_sha256=file_sha(reference_paths['body_posture']),
                body_adapted_path=str(adapted_paths['body_posture']),body_adapted_sha256=file_sha(adapted_paths['body_posture']),
                metadata_path=str(metadata_path),metadata_sha256=file_sha(metadata_path),
                native_duplicates_flagged=sum(r['same_detection'] for r in metadata),
                observed_counts={name:sum(r['streams'][name]['observed'] for r in metadata) for name in adapter.names},
                fill_counts={name:{kind:sum(r['streams'][name]['fill_kind']==kind for r in metadata)
                                  for kind in ('observed','repeat_previous','initial_blank','initial_body_sentinel')} for name in adapter.names},
                all_exact_parity=True,correct_anatomy_verified=False,natural_failure_labels=False))
            current_bytes=sum(p.stat().st_size for p in store.rglob('*') if p.is_file())
            if current_bytes>cfg['max_output_bytes']:raise ValueError('Owned RAM payload cap exceeded')
            run.save('input_parity.json',report)
        report.update(completed=True,total_frames=sum(s['frames'] for s in report['samples']),
                      all_exact_parity=True,artifact_bytes=sum(p.stat().st_size for p in store.rglob('*') if p.is_file()),
                      ready_for_registered_real_model_smoke=True,scientific_pilot_authorized=False)
        run.save('input_parity.json',report)
        run.status['output_hashes']={'input_parity.json':file_sha(run.output/'input_parity.json')}


if __name__=='__main__':main()
