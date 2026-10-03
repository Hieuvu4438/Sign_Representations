import argparse
from yt_dlp import YoutubeDL
from pathlib import Path
import os

def load_txt(path):    
    with open(path, "r") as f:
        vids = f.readlines()
    return vids

 
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
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True,
                        help='index of the sub_list to work with')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='output folder')
    parser.add_argument('--problem_path', type=str, required=True,
                        help='problem path')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='number of videos to download per device')
    parser.add_argument('--yasl_txt_file_path', type=str, required=True, 
                        help='path to the yasl txt file')
    args = parser.parse_args()
    index = args.index
    output_folder = args.output_folder
    problem_path = args.problem_path
    batch_size = args.batch_size
    yasl_txt_file_path = args.yasl_txt_file_path
    
    fixed_list = load_txt(yasl_txt_file_path)
    
    

    video_batches = [fixed_list[i:i + batch_size] for i in range(0, len(fixed_list), batch_size)]
    ydl_opts = {
        "format": "mp4",
        "merge_output_format": "mp4",
        "outtmpl": f"{output_folder}/%(id)s.%(ext)s"    
        }

    for vid in video_batches[index]:        
        expected_path = f"{output_folder}/{vid}.mp4"
        if Path(expected_path).exists():
            continue
        else:
            Path(f"{output_folder}").mkdir(parents=True, exist_ok=True)            
            ydl_opts["outtmpl"] = f"{output_folder}/%(id)s.%(ext)s"
            
            try:
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download(vid)
                
            except Exception as e:
                print(f"Error processing {vid}: {e}")
                with open(problem_path, "a") as f:
                    f.write(f"{vid}\n")
                continue
                