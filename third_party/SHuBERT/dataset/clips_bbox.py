import os
import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import webvtt
from datetime import datetime, timedelta
import sys
import argparse

import os
import pickle
import gzip
import webvtt
import string
import cv2
import argparse
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

import time



def load_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

def is_string_in_file(file_path, target_string):
    try:
        with Path(file_path).open("r") as f:
            for line in f:
                if target_string in line:
                    return True
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def convert_to_frame_number(time, fps):
    return int(time.total_seconds() * fps)


def short_side_rescale(target_length, width, height):
    ratio = width / height
    if ratio > 1.0:
        new_height = target_length
        new_width = target_length * ratio
    else:
        new_width = target_length
        new_height = target_length / ratio
    return int(new_width), int(new_height)


def long_side_rescale(target_length, width, height):
    ratio = width / height
    if ratio < 1.0:
        new_height = target_length
        new_width = target_length * ratio
    else:
        new_width = target_length
        new_height = target_length / ratio
    return int(new_width), int(new_height)


def get_optical_flow(images):
    prv_gray = None
    motion_mags = []
    for frame in images:
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_size = (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5))
        cur_gray = cv2.resize(cur_gray, gray_size)
        if prv_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prv_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag = (255.0*(mag-mag.min())/max(float(mag.max()-mag.min()), 1)).astype(np.uint8)
            mag = cv2.resize(mag, (frame.shape[1], frame.shape[0]))
        else:
            mag = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        prv_gray = cur_gray
        motion_mags.append(mag)
    return motion_mags


def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def find_target_bbox(bbox_arr, opts, iou_thr=0.5, len_ratio_thr=0.5):
    tubes = []
    num_rest = sum([len(x) for x in bbox_arr])
    while num_rest > 0:
        for i, bboxes in enumerate(bbox_arr):
            if len(bboxes) > 0:
                anchor = [i, bbox_arr[i].pop()]
                break
        tube = [anchor]
        for i in range(len(bbox_arr)):
            bboxes = bbox_arr[i]
            if anchor[0] == i or len(bboxes) == 0:
                continue
            ious = np.array([get_iou(anchor[1], bbox) for bbox in bboxes])
            j = ious.argmax()
            if ious[j] > iou_thr:
                target_bbox = bboxes.pop(j)
                tube.append([i, target_bbox])
        tubes.append(tube)
        num_rest = sum([len(x) for x in bbox_arr])
        
    max_val, max_tube = -1, None
    for itube, tube in enumerate(tubes):
        mean_val = 0
        for iframe, bbox in tube:
            x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            mean_val += opts[iframe][max(y0, 0): y1, max(x0, 0): x1].mean()
        mean_val /= len(tube)
        
        if len(tube)/len(opts) > len_ratio_thr:
            if mean_val > max_val:
                max_val, max_tube = mean_val, tube
                
    if max_tube is not None:
        target_bbox = np.array([bbox[1] for bbox in max_tube]).mean(axis=0).tolist()
    else:
        target_bbox = None
    return target_bbox, tubes


def crop_clip(video_path, problem_file_path, crop_clip_path, yolo_model_path):
    try:
        # Given a video file which is a single clip, crop the clip to the bouding box of the signer    
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        #new_width, new_height = long_side_rescale(640, width, height)
        new_width, new_height = width, height
        
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        
        up_exp, down_exp, left_exp, right_exp = 0.01, 0.01, 0.3, 0.3
        
        
        model = YOLO(yolo_model_path)    
        
        bboxes = []
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run YOLOv8 inference
            results = model(frame_rgb)
            
            # Filter for person class (class 0 in COCO dataset)
            person_bboxes = []
            for r in results:
                for box in r.boxes:
                    if box.cls == 0:  # person class
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        person_bboxes.append([x1, y1, x2, y2])
            
            bboxes.append(person_bboxes)
            frames.append(frame)
        
        if len(frames) == 0:
            return
                
            
        for i in range(len(bboxes)):
            for j in range(len(bboxes[i])):
                x0, y0, x1, y1 = bboxes[i][j]
                w, h = x1-x0+1, y1-y0+1
                x0, y0, x1, y1 = x0-w*left_exp, y0-h*up_exp, x1+w*right_exp, y1+h*down_exp
                bboxes[i][j] = [x0, y0, x1, y1]
        
        # try:
        if max([len(x) for x in bboxes]) == 1:
            bboxes = list(filter(lambda x: len(x) == 1, bboxes))
            bbox = np.array(bboxes).mean(axis=0)[0].tolist()
            tubes = []
        else:
            opts = get_optical_flow(frames)
            bbox, tubes = find_target_bbox(bboxes, opts)
        # except:
        #     bbox = None
        #     tubes = []
            
        if bbox is None and len(tubes) > 0:
            if max([len(x) for x in tubes]) > 0:
                total_sizes = []
                for tube in tubes:
                    total_size = sum([(bbox[3]-bbox[0])*(bbox[2]-bbox[1]) for _, bbox in tube])
                    total_sizes.append(total_size)
                idx = np.array(total_sizes).argmax()
                bbox = np.array([x for _, x in tubes[idx]]).mean(axis=0).tolist()
        
        if bbox is not None:
            x_0 = int(max(bbox[0], 0))
            y_0 = int(max(bbox[1], 0))
            x_1 = int(min(bbox[2], new_width))
            y_1 = int(min(bbox[3], new_height))
            
            for i in range(len(frames)):
                frames[i] = frames[i][y_0:y_1, x_0:x_1]
        # max_size = max(frames[0].shape[0], frames[0].shape[1])
        # clip_width, clip_height = short_side_rescale(max_size, frames[0].shape[1], frames[0].shape[0])
        
        clip_width, clip_height = frames[0].shape[1], frames[0].shape[0]
        resized_clip = cv2.VideoWriter(
            crop_clip_path,
            fourcc, fps, (clip_width, clip_height),
        )
        for frame in frames:
            # frame = cv2.resize(frame, (clip_width, clip_height))
            resized_clip.write(frame)
        resized_clip.release()


        
    except Exception as e:
        with open(problem_file_path, "a") as f:
            f.write(f"{video_path}\n")
        # pass
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True,
                        help='index of the sub_list to work with')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch size')
    # parser.add_argument('--time_limit', type=int, required=False,
    #                     help='time limit in seconds')
    parser.add_argument('--files_list', type=str, required=True,
                        help='path to the files list')
    parser.add_argument('--output_clips_directory', type=str, required=True,
                        help='path to the output clips directory')
    parser.add_argument('--problem_file_path', type=str, required=True,
                        help='path to the problem file path')
    parser.add_argument('--yolo_model_path', type=str, required=True,
                        help='path to the yolo model')
    
    args = parser.parse_args()
    
    index = args.index
    batch_size = args.batch_size
    # time_limit = args.time_limit
    files_list = args.files_list
    output_clips_directory = args.output_clips_directory
    problem_file_path = args.problem_file_path
    yolo_model_path = args.yolo_model_path
    fixed_list = load_file(files_list)
    output_path = f"{output_clips_directory}/"
 
    
    start_time = time.time()
    
    video_batches = [fixed_list[i:i + batch_size] for i in range(0, len(fixed_list), batch_size)]
    print(f"length of video_batches: {len(video_batches[0])}")
    for video_file in video_batches[index]:
        crop_clip_path = f"{output_path}{video_file.split('/')[-1]}"
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        current_time = time.time()
        # if current_time - start_time > time_limit:
        #     print("Time limit of 7 hours and 40 minutes reached. Stopping execution.")
        #     break    


        if os.path.exists(crop_clip_path):
            print(f"Skipping {video_file} - output already exists")
            continue
        elif is_string_in_file(problem_file_path, video_file):
            print(f"Skipping {video_file} - problem file already exists")
            continue
        else:
            print(f"Processing {video_file}")        
            crop_clip(video_file, problem_file_path, crop_clip_path, yolo_model_path)
