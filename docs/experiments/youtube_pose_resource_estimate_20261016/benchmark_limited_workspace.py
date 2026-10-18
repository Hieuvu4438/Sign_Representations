import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/home/haipd/Sign_Representations')
OUT=ROOT/'docs/experiments/youtube_pose_resource_estimate_20261016/limited_workspace'


def worker(folder,index):
    import torch
    import cv2
    import numpy as np
    from rtmlib import Wholebody
    torch.set_num_threads(1);cv2.setNumThreads(1)
    from rtmlib.tools.base import RTMLIB_SETTINGS
    RTMLIB_SETTINGS['onnxruntime']['cuda']=('CUDAExecutionProvider',{
        'gpu_mem_limit':str(4*1024**3),'arena_extend_strategy':'kSameAsRequested',
        'cudnn_conv_algo_search':'HEURISTIC','cudnn_conv_use_max_workspace':'0'})
    models=ROOT/'method/asm_jepa/data/models/rtmw133'
    detector=Wholebody(det=str(models/'yolox_m_8xb8-300e_humanart-c2c7a14a.onnx'),det_input_size=(640,640),
                      pose=str(models/'rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx'),pose_input_size=(288,384),
                      mode='performance',to_openpose=False,backend='onnxruntime',device='cuda')
    providers={'det':detector.det_model.session.get_providers(),'pose':detector.pose_model.session.get_providers()}
    assert all(p[0]=='CUDAExecutionProvider' for p in providers.values()),providers
    rows=list(csv.DictReader(open('/home/dongvk/datasets/ASL_Citizen/splits/train.csv')))
    frames=[]
    for row in rows[:3]:
        capture=cv2.VideoCapture('/home/dongvk/datasets/ASL_Citizen/videos/'+row['Video file'])
        for _ in range(24):
            ok,frame=capture.read()
            if not ok:break
            frames.append(frame)
        capture.release()
    assert frames
    for frame in frames[:5]:detector(frame)
    (folder/f'{index}.ready.json').write_text(json.dumps({'pid':os.getpid(),'providers':providers}))
    deadline=time.monotonic()+90
    while not (folder/'start').exists():
        if time.monotonic()>deadline:raise TimeoutError('Benchmark start barrier timeout')
        time.sleep(.05)
    started=time.time();count=people=0
    while time.time()-started<15:
        keypoints,scores=detector(frames[count%len(frames)])
        assert keypoints.shape[1:]==(133,2)
        people+=len(keypoints);count+=1
    finished=time.time()
    result={'pid':os.getpid(),'frames':count,'started':started,'finished':finished,'fps':count/(finished-started),
            'average_detected_people':people/count,'frame_shapes':sorted({str(frame.shape) for frame in frames}),
            'providers':providers}
    (folder/f'{index}.result.json').write_text(json.dumps(result))


def free_mib():
    return int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())


def memory():
    output=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    result={}
    for line in output.splitlines():
        try:pid,value=line.split(',');result[int(pid)]=int(value)
        except ValueError:continue
    return result


def parent():
    OUT.mkdir(parents=True,exist_ok=True);results=[]
    for n in [1,2,4]:
        if results and free_mib()<n*max(results[-1]['peak_per_process_MiB'].values())*1.3+2048:
            print(json.dumps({'skipped_workers':n,'reason':'VRAM headroom'}),flush=True);break
        folder=OUT/f'workers_{n}';folder.mkdir(exist_ok=False)
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        processes=[];logs=[];peaks={}
        try:
            for index in range(n):
                log=(folder/f'{index}.log').open('w');logs.append(log)
                proc=subprocess.Popen([str(ROOT/'.venv-unisign/bin/python'),__file__,'--worker',str(index),'--folder',str(folder)],env=env,stdout=log,stderr=subprocess.STDOUT)
                processes.append(proc);peaks[proc.pid]=0
            deadline=time.monotonic()+90
            while not all((folder/f'{i}.ready.json').exists() for i in range(n)):
                usage=memory()
                for proc in processes:peaks[proc.pid]=max(peaks[proc.pid],usage.get(proc.pid,0))
                if any(proc.poll() is not None for proc in processes):raise RuntimeError('Worker failed; inspect benchmark logs')
                if time.monotonic()>deadline:raise TimeoutError('Detector startup timeout')
                time.sleep(.2)
            print(json.dumps({'workers':n,'stage':'warmup_complete','process_VRAM_MiB':peaks}),flush=True)
            (folder/'start').touch()
            while any(proc.poll() is None for proc in processes):
                usage=memory()
                for proc in processes:peaks[proc.pid]=max(peaks[proc.pid],usage.get(proc.pid,0))
                time.sleep(.2)
            assert all(proc.returncode==0 for proc in processes)
            rows=[json.loads((folder/f'{i}.result.json').read_text()) for i in range(n)]
            elapsed=max(r['finished'] for r in rows)-min(r['started'] for r in rows)
            result={'workers':n,'aggregate_fps':sum(r['frames'] for r in rows)/elapsed,
                    'per_worker_fps':[r['fps'] for r in rows],'peak_per_process_MiB':peaks,
                    'sum_process_peak_MiB':sum(peaks.values()),'average_detected_people':[r['average_detected_people'] for r in rows]}
            result['speedup_vs_one_worker']=result['aggregate_fps']/results[0]['aggregate_fps'] if results else 1.
            results.append(result);print(json.dumps(result),flush=True)
        finally:
            for proc in processes:
                if proc.poll() is None:proc.terminate()
            for proc in processes:
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            for log in logs:log.close()
    (OUT/'benchmark.json').write_text(json.dumps({'configuration':'YOLOX-m640 + RTMW-x384x288 /133keypoints /ONNX CUDA /batch1 /HEURISTIC /max_workspace0 /4GiB_arena_per_session',
        'scope':'15s steady-state predecoded ASL-Citizen frames, includes detector+pose+CPU pre/postprocessing; excludes video decode/tracking/cache writes; GPU shared with existing jobs',
        'VRAM_measurement':'nvidia-smi per-PID sampled ~0.2s; sampled peaks, not guaranteed allocator maxima',
        'results':results},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);parser.add_argument('--folder')
    args=parser.parse_args()
    if args.worker is None:parent()
    else:worker(Path(args.folder),args.worker)
