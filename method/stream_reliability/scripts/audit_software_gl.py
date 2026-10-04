"""Isolated Mesa/software detector replay; conservatively charge the probe budget."""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'method/frozen_probe'))
sys.path.insert(0,str(ROOT))
from cache import file_sha
from gpu_accounting import inspect_effective_budget
from resources import gpu_preflight
from run_context import RunContext


def worker(config,output):
    import numpy as np
    import decord
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    spec=importlib.util.spec_from_file_location('author_landmark_cpu_reference',REPO/config['author_source'])
    author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
    cohort=json.loads((REPO/config['cohort_manifest']).read_text())
    sources={row['sample_id']:row for row in cohort['samples']}
    raw=json.loads((REPO/config['landmark_report']).read_text())
    samples=[]
    for sample in raw['samples']:
        if file_sha(sample['raw_path'])!=sample['raw_sha256']:raise ValueError('Raw original changed')
        with gzip.open(sample['raw_path'],'rt') as stream:reference=json.load(stream)['frames']
        reader=decord.VideoReader(sources[sample['sample_id']]['input_path'],ctx=decord.cpu(0),num_threads=1)
        face_options=vision.FaceLandmarkerOptions(base_options=python.BaseOptions(
            model_asset_path=str(REPO/config['face_task']),delegate=python.BaseOptions.Delegate.CPU),
            output_face_blendshapes=True,output_facial_transformation_matrixes=True,num_faces=6)
        hand_options=vision.HandLandmarkerOptions(base_options=python.BaseOptions(
            model_asset_path=str(REPO/config['hand_task']),delegate=python.BaseOptions.Delegate.CPU),
            num_hands=6,min_hand_detection_confidence=.05)
        errors={name:0. for name in ('face_landmarks','hand_landmarks','pose_landmarks')}
        observed_hand_frames=0
        with vision.FaceLandmarker.create_from_options(face_options) as face, \
                vision.HandLandmarker.create_from_options(hand_options) as hand, \
                mp.solutions.holistic.Holistic(min_detection_confidence=.1) as holistic:
            author.face_detector=face;author.hand_detector=hand;author.mp_holistic=holistic
            for index in range(min(config['prefix_frames'],len(reader))):
                counts,landmarks=author.detect_holistic(reader[index].asnumpy())
                if counts!=reference[index]['counts']:raise ValueError('Detector availability/count parity differs')
                observed_hand_frames+=int(counts['#hands']>0)
                for name,actual in landmarks.items():
                    expected=reference[index]['landmarks'][name]
                    if (actual is None)!=(expected is None):raise ValueError('Raw missingness parity differs')
                    if actual is not None:
                        actual=np.asarray(actual);expected=np.asarray(expected)
                        if actual.shape!=expected.shape or not np.isfinite(actual).all():raise ValueError('Raw coordinate shape/finite differs')
                        error=float(np.max(np.abs(actual-expected)));errors[name]=max(errors[name],error)
                        if not np.allclose(actual,expected,atol=config['atol'],rtol=0):raise ValueError('Predeclared landmark coordinate tolerance differs')
        samples.append(dict(sample_id=sample['sample_id'],frames=min(config['prefix_frames'],len(reader)),
                            observed_hand_frames=observed_hand_frames,max_absolute_errors=errors,parity=True))
    maps=Path('/proc/self/maps').read_text()
    forbidden=[name for name in ('libEGL_nvidia','libGLESv2_nvidia','libnvidia-glcore','libcuda.so') if name in maps]
    device_fds=[]
    for fd in Path('/proc/self/fd').iterdir():
        try:target=os.readlink(fd)
        except OSError:continue
        if target.startswith('/dev/nvidia'):device_fds.append(target)
    if forbidden or device_fds:raise ValueError('Software-only worker loaded/opened NVIDIA graphics/CUDA resources')
    result=dict(samples=samples,all_pass=True,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                selected_environment={k:os.environ.get(k) for k in config['child_environment']},
                forbidden_NVIDIA_libraries=forbidden,NVIDIA_device_fds=device_fds,
                library_mappings=[line for line in maps.splitlines() if any(n in line for n in ('libEGL_mesa','swrast_dri'))])
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser()
    for field in ('config','output-dir'):parser.add_argument('--'+field,required=True,type=Path)
    parser.add_argument('--seed',required=True,type=int);parser.add_argument('--run-id',required=True)
    parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if args.worker:
        worker(config,args.output_dir/'worker_result.json');return
    with RunContext(args.output_dir,args.run_id,args.config,args.seed) as run:
        for name,digest in config['input_hashes'].items():
            if file_sha(REPO/name)!=digest:raise ValueError('Pinned software/resource/input changed')
        vendor=Path(config['child_environment']['__EGL_VENDOR_LIBRARY_FILENAMES'])
        if file_sha(vendor)!=config['mesa_vendor_sha256']:raise ValueError('Existing Mesa vendor configuration changed')
        budget=inspect_effective_budget(config);observation=gpu_preflight()
        if observation['uuid']!=config['granted_uuid']:raise ValueError('Granted GPU identity changed')
        run.start_gpu(budget)  # Conservative whole probe allowance, even if software succeeds.
        env=dict(os.environ);env.update(config['child_environment'])
        command=[sys.executable,str(Path(__file__).resolve()),'--config',str(args.config.resolve()),
                 '--seed',str(args.seed),'--run-id',args.run_id,'--output-dir',str(run.output.resolve()),'--worker']
        try:
            child=subprocess.run(command,env=env,capture_output=True,timeout=config['child_timeout_seconds'])
        except subprocess.TimeoutExpired as error:
            (run.output/'child_stdout.txt').write_bytes(error.stdout or b'')
            (run.output/'child_stderr.txt').write_bytes(error.stderr or b'')
            raise
        (run.output/'child_stdout.txt').write_bytes(child.stdout)
        (run.output/'child_stderr.txt').write_bytes(child.stderr)
        if child.returncode:raise RuntimeError(f'Software-only replay child failed{child.returncode}; stderr retained')
        stderr=child.stderr.decode(errors='replace')
        renderer=[line for line in stderr.splitlines() if 'renderer:' in line]
        if not renderer or any('llvmpipe' not in line.lower() for line in renderer):raise ValueError('Observed software renderer required for every context')
        result=json.loads((run.output/'worker_result.json').read_text())
        if result['CUDA_VISIBLE_DEVICES']!=os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('Scheduler visibility changed')
        result.update(run_id=args.run_id,renderer_lines=renderer,child_command=command,
            scope='128 fixed prefix frames/software resource parity only, not natural-failure or benchmark validation',
            accounting='Entire parent child-probe allocation conservatively charged; no retroactive refund of LANDMARK001')
        run.save('software_gl_audit.json',result)
        run.status['output_hashes']={name:file_sha(run.output/name) for name in
            ('software_gl_audit.json','worker_result.json','child_stdout.txt','child_stderr.txt')}


if __name__=='__main__':main()
