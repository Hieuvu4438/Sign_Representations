import gzip
import pickle
import os


input_files_list = os.listdir("path/to/your/input/directory")

output_path = "path/to/your/output/directory"
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(output_path, "clips_bbox.list")
with gzip.GzipFile(output_file, 'wb') as file_b:
    file_b.write(pickle.dumps(input_files_list, protocol=0))