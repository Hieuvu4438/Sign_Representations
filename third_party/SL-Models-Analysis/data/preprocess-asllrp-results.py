import numpy as np
import os
import pathlib
import utils

fields = ["video_id", "utterance_id", "phenomenon", "good_sentence", "bad_sentence", "hypothesis", "good_nll", "bad_nll", "nll_diff", "bleurt"]

def extract_fields(dataset, fields):
    selected = []
    for item in dataset:
        selected.append([item[field] for field in fields])

    return selected

input_folder = "data/results/ASLLRP_WeightedSum/"
output_folder = "data/results/ASLLRP_Processed-refs/"


pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        data = utils.read_jsonl(os.path.join(input_folder, filename))
        selected = extract_fields(data, fields)
        output_path = os.path.join(output_folder, filename.replace(".json", ".csv"))
        output_path_string = str(output_path)
        utils.write_csv(selected,output_path_string, header=fields)