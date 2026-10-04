"""Native helper counterexamples and fixed train/dev crop/body/color contracts."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'method/frozen_probe'))
sys.path.insert(0, str(ROOT))
from cache import file_sha
from resources import scratch_preflight
from run_context import RunContext


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def observe(name, function):
    try:
        return dict(name=name, result=function(), error=None)
    except Exception as error:
        return dict(name=name, result=None, error=dict(type=type(error).__name__, message=str(error)))


def main():
    parser = argparse.ArgumentParser()
    for field in ('config', 'output-dir'):
        parser.add_argument('--' + field, required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        for name, digest in cfg['input_hashes'].items():
            if file_sha(REPO / name) != digest:
                raise ValueError('Registered source/metadata changed')
        resources = scratch_preflight('/dev/shm', cfg['max_fixture_bytes'])
        work = Path('/dev/shm') / ('shubert_crop_audit_' + args.run_id)
        work.mkdir(exist_ok=False)
        run.status['fixture_root'] = str(work)
        run.save('status.json', run.status)
        import cv2
        import decord
        import numpy as np
        cv2.setNumThreads(1)
        hands = module('native_hands', REPO / cfg['hands_source'])
        face = module('native_face', REPO / cfg['face_source'])
        body = module('native_body', REPO / cfg['body_source'])
        snapshot = run.output / 'upstream_source_snapshot'
        for name in (cfg['hands_source'], cfg['face_source'], cfg['body_source']):
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, target)
        pose = [[.5, .5, 0.] for _ in range(33)]
        for i, xy in {0:(.5,.3), 2:(.45,.25), 11:(.3,.5), 12:(.7,.5),
                      13:(.2,.6), 14:(.8,.6), 15:(.15,.7), 16:(.85,.7)}.items():
            pose[i][:2] = xy
        indices = [0, 11, 12, 13, 14, 15, 16]
        # Independent closed-form oracle, manually specified canonical geometry.
        expected = np.asarray([[0,-11/28],[-1/6,-1/4],[1/6,-1/4],[-1/4,-5/28],
                               [1/4,-5/28],[-7/24,-3/28],[7/24,-3/28]]).reshape(14)
        normalized = np.asarray(body.normalize_pose_keypoints(pose[:25]))[indices].reshape(14)
        if not np.allclose(normalized, expected, atol=1e-12, rtol=0):
            raise ValueError('Independent canonical body oracle differs')
        valid = dict(pose_landmarks=[pose], hand_landmarks=None, face_landmarks=None)

        def body_rows(name, rows):
            source = work / (name + '.json')
            source.write_text(json.dumps({str(i): row for i, row in enumerate(rows)}))
            dest = work / 'body_outputs'
            body.keypoints_to_numpy(source, dest)
            array = np.load(dest / (name + '.npy'), allow_pickle=False)
            return dict(shape=list(array.shape), finite=bool(np.isfinite(array).all()),
                        rows_equal=bool(len(array)>1 and np.array_equal(array[0],array[1])))

        fixtures = [dict(name='canonical_body_independent_oracle', result=dict(pass_oracle=True,
                        max_abs_error=float(np.max(np.abs(normalized-expected)))), error=None)]
        fixtures += [observe('body_observed_then_missing', lambda: body_rows('after', [valid, None])),
                     observe('body_initial_whole_row_None', lambda: body_rows('initial_row', [None, valid])),
                     observe('body_initial_pose_field_None', lambda: body_rows('initial_field', [dict(pose_landmarks=None), valid]))]
        collapsed = [p[:] for p in pose]
        collapsed[12] = collapsed[11][:]
        with np.errstate(divide='ignore', invalid='ignore'):
            fixtures.append(observe('collapsed_shoulders', lambda: dict(
                finite=bool(np.isfinite(np.asarray(body.normalize_pose_keypoints(collapsed))).all()))))

        def hand(x,y):
            return [[x,y,0.] for _ in range(21)]
        two = [hand(.15,.7), hand(.85,.7)]
        def select(p, candidates):
            left, right = hands.select_hands(p, candidates, (48,64,3))
            return dict(left_index=next((i for i,h in enumerate(candidates or []) if h is left),None),
                        right_index=next((i for i,h in enumerate(candidates or []) if h is right),None),
                        same_detection=left is not None and left is right)
        fixtures.append(observe('two_separated_hands', lambda: select(pose,two)))
        close = [p[:] for p in pose]
        close[15] = [.5,.5,0.]; close[16] = [.54,.5,0.]
        fixtures.append(observe('one_shared_detection_both_wrists', lambda: select(close,[hand(.52,.5)])))
        fixtures.append(observe('empty_hand_list', lambda: select(pose,[])))
        threshold_pose = [p[:] for p in pose]
        threshold_pose[15] = [0.,0.,0.]; threshold_pose[16] = [.9,.9,0.]
        fixtures.append(observe('exact_0p1_wrist_threshold', lambda: select(threshold_pose,[hand(.1,0.)])))
        pixels = np.zeros((48,64,3),dtype=np.uint8)
        pixels[:,:32] = [240,20,5]; pixels[:,32:] = [5,20,240]
        box = hands.adjust_bounding_box((-10,-10,200,200), pixels.shape)
        crop = hands.crop_frame(pixels,box)
        fixtures.append(dict(name='oversized_hand_box', result=dict(box=list(box),
                        pixel_shape=list(crop.shape), declared_extent_matches=tuple(crop.shape[:2])==(box[3],box[2])),error=None))
        lm = [[.5,.5,0.] for _ in range(478)]
        for i,xy in {69:(.35,.3),168:(.5,.4),156:(.38,.4),118:(.4,.45),54:(.45,.35),
                     299:(.55,.3),347:(.6,.4),336:(.65,.45),301:(.6,.35),
                     164:(.4,.6),212:(.45,.65),432:(.6,.65),18:(.5,.7)}.items():
            lm[i][:2] = xy
        rgb = face.cues_on_grey_background(pixels,lm)
        bgr = face.cues_on_grey_background(pixels[:,:,::-1].copy(),lm)
        if not np.array_equal(rgb,bgr[:,:,::-1]):
            raise ValueError('RGB/BGR crop helper permutation oracle failed')
        fixtures.append(dict(name='valid_face_color_permutation',result=dict(shape=list(rgb.shape),equal=True),error=None))
        edge = [p[:] for p in lm]
        edge[69][0] = -.2; edge[168][0] = 1.1
        fixtures.append(observe('face_cues_outside_image', lambda: dict(
            shape=list(face.cues_on_grey_background(pixels,edge).shape))))

        # Execute native crop functions with isolated in-memory reader/writers.
        # They see only fresh fixture paths; no upstream main/default path invoked.
        class Reader:
            def __len__(self): return 2
            def get_avg_fps(self): return 1.
            def __getitem__(self,index): return SimpleNamespace(asnumpy=lambda: pixels.copy())
            def seek(self,index): pass
        for name, author, output_count in [('face',face,1),('hands',hands,2)]:
            observed_writers = []
            class Writer:
                def __init__(self,*args): self.frames=[]; observed_writers.append(self)
                def write(self,image): self.frames.append(np.asarray(image).copy())
                def release(self): pass
            class CvProxy:
                VideoWriter = Writer
                def __getattr__(self,name): return getattr(cv2,name)
            old_cv,old_decord = author.cv2,author.decord
            author.cv2 = CvProxy()
            author.decord = SimpleNamespace(VideoReader=lambda path: Reader())
            fake = work / ('initial_none_' + name + '.mp4')
            pose_file = work / (fake.stem + '_pose.json')
            pose_file.write_text(json.dumps({'0':None,'1':dict(pose_landmarks=None)}))
            try:
                result = observe('initial_None_native_'+name+'_writer', lambda: (
                    author.video_holistic(str(fake),work/'fake_outputs',work/'problems.txt',work),
                    dict(expected_native_frames=2, written_frames=[len(w.frames) for w in observed_writers]))[1])
                fixtures.append(result)
            finally:
                author.cv2,author.decord = old_cv,old_decord

        fixture_video = work / 'color_lossless.mkv'
        frames = np.stack([pixels,pixels[:,::-1].copy()])
        command = ['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
                   '-s','64x48','-r','1','-i','pipe:0','-c:v','ffv1','-pix_fmt','bgr0','-threads','1',str(fixture_video)]
        subprocess.run(command,input=frames.tobytes(),capture_output=True,check=True,timeout=30)
        reader = decord.VideoReader(str(fixture_video),ctx=decord.cpu(0),num_threads=1)
        decoded = reader.get_batch(range(len(reader))).asnumpy()
        capture = cv2.VideoCapture(str(fixture_video))
        captured = []
        while True:
            ok,image = capture.read()
            if not ok: break
            captured.append(image[:,:,::-1])
        capture.release()
        if not np.array_equal(decoded,frames) or not np.array_equal(np.stack(captured),frames):
            raise ValueError('Independent lossless RGB/BGR oracle failed')
        fixtures.append(dict(name='lossless_known_color_oracle',result=dict(exact_decord_RGB=True,
                        exact_OpenCV_BGR_after_permutation=True,source_shape=list(frames.shape),command=command),error=None))
        roundtrip = work/'author_color_roundtrip.mp4'
        writer = cv2.VideoWriter(str(roundtrip),cv2.VideoWriter_fourcc(*'mp4v'),1.,(64,48))
        if not writer.isOpened(): raise ValueError('Registered mp4v CPU writer unavailable')
        for image in frames: writer.write(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        writer.release()
        coded = decord.VideoReader(str(roundtrip),ctx=decord.cpu(0),num_threads=1).get_batch([0,1]).asnumpy()
        fixtures.append(dict(name='author_swap_then_Bgr_writer_roundtrip',result=dict(
            max_abs_pixel_error=int(np.max(np.abs(coded.astype(int)-frames.astype(int)))),
            mean_abs_pixel_error=float(np.abs(coded.astype(int)-frames.astype(int)).mean()),
            left_R_exceeds_B=bool(coded[0,24,16,0]>coded[0,24,16,2]),
            lossy_codec=True,exact_pixel_parity_claim=False),error=None))
        report = dict(run_id=args.run_id,GPU_hours=0,fixtures=fixtures,samples=[],
                      resources=resources,fixture_root=str(work),upstream_sources=cfg['input_hashes'],
                      scope='CPU input geometry/observability/color audit only, no DINO/model/task inference',
                      natural_failure_ground_truth=False)
        run.save('crop_contract.json',report)
        raw_report = json.loads((REPO/cfg['landmark_report']).read_text())
        cohort = json.loads((REPO/cfg['cohort_manifest']).read_text())
        sources = {row['sample_id']:row for row in cohort['samples']}
        frame_rows = []
        for sample in raw_report['samples']:
            if file_sha(sample['raw_path']) != sample['raw_sha256']: raise ValueError('Raw observations changed')
            source = sources[sample['sample_id']]
            if source['split'] not in ('train','val') or file_sha(source['input_path']) != sample['source_sha256']:
                raise ValueError('Fixed train/dev video changed')
            with gzip.open(sample['raw_path'],'rt') as stream: data = json.load(stream)
            reader = decord.VideoReader(source['input_path'],ctx=decord.cpu(0),num_threads=1)
            capture = cv2.VideoCapture(source['input_path'])
            if len(reader)!=sample['frames'] or len(data['frames'])!=len(reader): raise ValueError('Native length differs')
            timestamps=reader.get_frame_timestamp(range(len(reader)))
            counts=dict(left_selected=0,right_selected=0,same_detection=0,face_crop_errors=0,
                        hand_crop_errors=0,body_nonfinite=0,real_decord_vs_CvRGB_max_error=0)
            for i,row in enumerate(data['frames']):
                if row['frame_index']!=i or abs(float(timestamps[i,0])-row['pts_start_s'])>cfg['pts_tolerance_seconds']:
                    raise ValueError('Frame index/timestamp identity differs')
                image=reader[i].asnumpy();ok,bgr=capture.read()
                if not ok or bgr.shape!=image.shape: raise ValueError('Independent native decode/count differs')
                counts['real_decord_vs_CvRGB_max_error']=max(counts['real_decord_vs_CvRGB_max_error'],
                    int(np.max(np.abs(image.astype(int)-bgr[:,:,::-1].astype(int)))))
                raw=row['landmarks'];p=raw['pose_landmarks'][0] if raw['pose_landmarks'] else None
                record=dict(sample_id=sample['sample_id'],frame_index=i,pts_start_s=row['pts_start_s'])
                if p is None: raise ValueError('Current raw cohort has unexpected missing body; no substitution')
                selected=select(p,raw['hand_landmarks'])
                record.update(selected)
                counts['same_detection']+=int(selected['same_detection'])
                for side in ('left','right'):
                    idx=selected[side+'_index'];counts[side+'_selected']+=int(idx is not None)
                    if idx is not None:
                        box=hands.adjust_bounding_box(hands.get_bounding_box(raw['hand_landmarks'][idx],image.shape,scale_factor=1.5),image.shape)
                        native_crop=hands.crop_frame(image,box)
                        resized=hands.resize_frame(native_crop,(224,224))
                        good=resized is not None and resized.shape==(224,224,3)
                        record[side+'_crop']=dict(box=[int(v) for v in box],valid=good,shape=list(native_crop.shape),
                            declared_extent_matches=tuple(native_crop.shape[:2])==(box[3],box[2]))
                        counts['hand_crop_errors']+=int(not good)
                face_lm=face.select_face(p,raw['face_landmarks']) if raw['face_landmarks'] else None
                result=observe('face_crop',lambda: dict(shape=list(face.resize_frame(face.cues_on_grey_background(image,face_lm),(224,224)).shape)))
                record['face_crop']=result;counts['face_crop_errors']+=int(result['error'] is not None)
                norm=np.asarray(body.normalize_pose_keypoints(p[:25]))[indices].reshape(14)
                record['body_finite']=bool(np.isfinite(norm).all());counts['body_nonfinite']+=int(not record['body_finite'])
                frame_rows.append(record)
            if capture.read()[0]: raise ValueError('Independent decoder has extra frames')
            capture.release()
            report['samples'].append(dict(sample_id=sample['sample_id'],split=source['split'],frames=len(reader),counts=counts))
            run.save('crop_contract.json',report)
        total=sum(p.stat().st_size for p in work.rglob('*') if p.is_file())
        if total>cfg['max_fixture_bytes']: raise ValueError('Fixture byte cap exceeded')
        run.save('frame_geometry.json',frame_rows)
        report.update(completed=True,total_frames=len(frame_rows),fixture_bytes=total,
                      read_only_upstream=True,real_four_stream_gate='hold for registered missingness/assignment/codec adapter')
        run.save('crop_contract.json',report)
        run.status['output_hashes']={name:file_sha(run.output/name) for name in ('crop_contract.json','frame_geometry.json')}


if __name__=='__main__':
    main()
