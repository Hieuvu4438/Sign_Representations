import os
import cv2
import argparse
from datetime import datetime
from pathlib import Path

def time_to_seconds(time_str):
    time_object = datetime.strptime(time_str, "%H:%M:%S.%f")
    return (time_object.hour * 3600 + 
            time_object.minute * 60 +
            time_object.second +
            time_object.microsecond / 1_000_000)

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

def make_clips_from_video(video_path, start_time, end_time, output_path, done_file_path, problem_file_path):
    try:
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        video_id = Path(video_path).stem
        clip_name = f"{video_id}_{start_frame:06d}-{end_frame:06d}.mp4"
        clip_path = output_path / clip_name
        
        if clip_path.exists():
            return
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
        
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(start_frame, end_frame + 1):
            ret, frame = video.read()
            if ret:
                out.write(frame)
            else:
                break
                
        video.release()
        out.release()
        
        with open(done_file_path, "a") as f:
            f.write(f"{video_path}\n")
            
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        with open(problem_file_path, "a") as f:
            f.write(f"{video_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--manifest_path', type=str, required=True)
    parser.add_argument('--videos_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--done_file_path', type=str, required=True)
    parser.add_argument('--problem_file_path', type=str, required=True)
    args = parser.parse_args()
    
    # Convert string paths to Path objects
    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read manifest file
    clips_list = []
    with open(args.manifest_path) as f:
        for line in f:
            video_name, start_time, end_time, _ = line.strip().split("\t")
            clips_list.append((video_name, float(start_time), float(end_time)))
    
    # Group into batches
    batch_size = 50
    clip_batches = [clips_list[i:i + batch_size] for i in range(0, len(clips_list), batch_size)]
    
    # Process assigned batch
    for video_name, start_time, end_time in clip_batches[args.index]:
        video_path = videos_dir / f"{video_name}.mp4"
        
        if not video_path.exists() or is_string_in_file(args.done_file_path, str(video_path)):
            continue
            
        make_clips_from_video(
            str(video_path),
            start_time,
            end_time,
            output_dir,
            args.done_file_path,
            args.problem_file_path
        )