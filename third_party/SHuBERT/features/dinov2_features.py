import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import decord
from decord import VideoReader
from decord import cpu, gpu
import numpy as np
import os
import pickle
import gzip
from pathlib import Path
import argparse
import json
import csv
import glob
import time 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


def get_dino_finetuned_downloaded(dino_path):
    # Load the original DINOv2 model with the correct architecture and parameters.
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg', pretrained=False)  # Removed map_location
    # Load finetuned weights
    pretrained = torch.load(dino_path, map_location=device)
    # Make correct state dict for loading
    new_state_dict = {}
    for key, value in pretrained['teacher'].items():
        if 'dino_head' in key:
            print('not used')
        else:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value
    # Change shape of pos_embed
    pos_embed = nn.Parameter(torch.zeros(1, 257, 384))
    model.pos_embed = pos_embed
    # Load state dict
    model.load_state_dict(new_state_dict, strict=True)
    # Move model to GPU
    model.to(device)
    return model


def preprocess_image(image):
    #Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size expected by the model
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])
    # image is a PIL Image
    return transform(image).unsqueeze(0)  # Add batch dimension



def preprocess_frame(frame):
    """Preprocess a single frame"""
    transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.fromarray(frame)
    return transform(image)[:3]  # Ensure only RGB channels are considered

def video_to_embeddings(video_path, output_folder, dino_path, batch_size=128):
    #"""Extract frames from a video and compute embeddings in batches"""
    try:
        vr = VideoReader(video_path, width=224, height=224)
    except:
        print(f'path doesnt exist: {video_path}')
        return
    total_frames = len(vr)
    all_embeddings = []

    for idx in range(0, total_frames, batch_size):
        batch_frames = vr.get_batch(range(idx, min(idx + batch_size, total_frames))).asnumpy()
        
        # Preprocess and stack frames to form a batch
        batch_tensors = torch.stack([preprocess_frame(frame) for frame in batch_frames]).cuda()

        with torch.no_grad():
            # Process the entire batch through the model
            batch_embeddings = model(batch_tensors.to('cuda')).cpu().numpy()
        
        all_embeddings.append(batch_embeddings)

    embeddings = np.concatenate(all_embeddings, axis=0)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    np_path = f"{output_folder}/{video_name}.npy"
    np.save(np_path, embeddings)

        
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True,
                        help='index of the sub_list to work with')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='time limit in seconds')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='number of videos to process in this batch')
    parser.add_argument('--files_list', type=str, required=True,
                        help='path to the files list file')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='path to the output folder')
    parser.add_argument('--dino_path', type=str, required=True,
                        help='path to the dino model')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    index = args.index
    time_limit = args.time_limit
    batch_size_in = args.batch_size
    files_list = args.files_list
    output_folder = args.output_folder
    fixed_list = load_file(files_list)    
    global dino_path
    dino_path = args.dino_path
    
    
    video_batches = [fixed_list[i:i + batch_size_in] for i in range(0, len(fixed_list), batch_size_in)]
    print(f"Total number of video batches: {len(video_batches)}")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    model = get_dino_finetuned_downloaded(dino_path)
    
    for video_path in video_batches[index]:
        clip_video_id = video_path.split('/')[-1].split('.')[0] 
        
        current_time = time.time()
        if current_time - start_time > time_limit:
            print("Time limit reached. Stopping execution.")
            break
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        np_path = f"{output_folder}/{video_name}.npy"   
           
        if os.path.exists(np_path):
            continue
        else:
            video_to_embeddings(video_path, output_folder, dino_path, batch_size=512)
