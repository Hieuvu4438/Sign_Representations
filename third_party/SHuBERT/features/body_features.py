import cv2
import numpy as np
import os
import pickle
import gzip
from datetime import datetime
from pathlib import Path
import decord
import argparse
import json
import glob

import time


def get_mp4_files(directory):
    # Ensure the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f'Directory not found: {directory}')

    # Use glob to find all .mp4 files
    mp4_files = glob.glob(os.path.join(directory, '*.mp4'))

    # Convert to absolute paths
    absolute_paths = [os.path.abspath(file) for file in mp4_files]

    return absolute_paths

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

def normalize_pose_keypoints(pose_landmarks):
    # Extract relevant landmarks
    left_shoulder = np.array(pose_landmarks[11][:2])
    right_shoulder = np.array(pose_landmarks[12][:2])
    left_eye = np.array(pose_landmarks[2][:2])
    nose = np.array(pose_landmarks[0][:2])

    # Calculate head unit in normalized space
    head_unit = np.linalg.norm(right_shoulder - left_shoulder) / 2

    # Define signing space dimensions in normalized space
    signing_space_width = 6 * head_unit
    signing_space_height = 7 * head_unit

    # Calculate signing space bounding box in normalized space
    signing_space_top = left_eye[1] - 0.5 * head_unit
    signing_space_bottom = signing_space_top + signing_space_height
    signing_space_left = nose[0] - signing_space_width / 2
    signing_space_right = signing_space_left + signing_space_width

    # Create transformation matrix
    translation_matrix = np.array([[1, 0, -signing_space_left],
                                   [0, 1, -signing_space_top],
                                   [0, 0, 1]])
    scale_matrix = np.array([[1 / signing_space_width, 0, 0],
                             [0, 1 / signing_space_height, 0],
                             [0, 0, 1]])
    shift_matrix = np.array([[1, 0, -0.5],
                             [0, 1, -0.5],
                             [0, 0, 1]])
    transformation_matrix = shift_matrix @ scale_matrix @ translation_matrix

    # Apply transformation to pose keypoints
    normalized_keypoints = []
    for landmark in pose_landmarks:
        keypoint = np.array([landmark[0], landmark[1], 1])
        normalized_keypoint = transformation_matrix @ keypoint
        normalized_keypoints.append(normalized_keypoint[:2])

    return normalized_keypoints



def keypoints_to_numpy(pose_file, pose_emb_path):
    try:
        landmark_json_path = Path(pose_file)
        with open(landmark_json_path, 'r') as rd:
            result_dict = json.load(rd)
    except Exception as e:
        return

    prev_pose = None
    video_pose_landmarks = []
    #extract pose landmarks from the json file and save them into a directory as a single .npy file for each video
    for i in range(len(result_dict)):
        if result_dict[str(i)] is None:
            if prev_pose is not None:
                frame_pose_landmarks = prev_pose
            else:
                frame_pose_landmarks = np.full((7,2), -9999)
        elif result_dict[str(i)]['pose_landmarks'] is not None:
            frame_pose_landmarks = result_dict[str(i)]['pose_landmarks'][0]
            #select only the points at some specifice indices 
            # 0 is the nose, 11 is the left shoulder, 12 is the right shoulder, 
            # 13 is the left elbow, 14 is the right elbow, 15 is the left wrist, 
            # 16 is the right wrist
            indices = [0,11,12,13,14,15,16] 
            #pose_landmarks = [pose_landmarks[j] for j in indices]
            frame_pose_landmarks = normalize_pose_keypoints(frame_pose_landmarks[0:25])
            frame_pose_landmarks = [frame_pose_landmarks[j] for j in indices]
            #flatten the above vector 
            frame_pose_landmarks = np.array(frame_pose_landmarks).flatten()
            prev_pose = frame_pose_landmarks
        elif prev_pose is not None:
            frame_pose_landmarks = prev_pose
        else:
            #create a tensor and fill it with -9999
            frame_pose_landmarks = np.full((7,2), -9999).flatten()
        video_pose_landmarks.append(frame_pose_landmarks)
    
    video_pose_landmarks = np.array(video_pose_landmarks)
    
  
    np_path = Path(pose_emb_path) / Path(f"{str(pose_file).split('/')[-1].rsplit('.', 1)[0]}.npy")
    np_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(np_path, video_pose_landmarks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True,
                        help='index of the sub_list to work with')
    parser.add_argument('--files_list', type=str, required=True,
                        help='path to the pose file')
    parser.add_argument('--pose_features_path', type=str, required=True,
                        help='path to the pose features file')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch size')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='time limit')
    args = parser.parse_args()

    start_time = time.time()
    
    
    
    index = args.index

    fixed_list = load_file(args.files_list)
    files_list = args.files_list
    pose_emb_path = args.pose_features_path
    batch_size = args.batch_size    
    time_limit = args.time_limit

    video_batches = [fixed_list[i:i + batch_size] for i in range(0, len(fixed_list), batch_size)]
    for pose_file in video_batches[index]:
        pose_file = Path(pose_file)# / Path(pose_file.split('/')[-1].rsplit('.', 1)[0] + "_pose.json")
        np_path = Path(pose_emb_path) / Path(f"{str(pose_file).split('/')[-1].rsplit('.', 1)[0]}.npy")
        if os.path.exists(np_path):
            continue
        current_time = time.time()
        if current_time - start_time > time_limit:
            print("Time limit reached. Stopping execution.")
            break
        # print(pose_file)
        keypoints_to_numpy(pose_file, pose_emb_path)
