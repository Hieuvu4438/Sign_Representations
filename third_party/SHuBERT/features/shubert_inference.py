import torch
import numpy as np
import csv
import os
from tqdm import tqdm
import torch
import argparse
from examples.shubert.models.shubert import SHubertModel, SHubertConfig

# Function to load the model
def load_model(checkpoint_path):
    cfg = SHubertConfig() 
    
    # Initialize the model
    model = SHubertModel(cfg)
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # If the checkpoint is saved with a 'model' key
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load the state dictionary into the model
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    model.cuda()  # Move to GPU if available
    return model


# Function to process a single sample
def process_sample(model, face_path, left_hand_path, right_hand_path, body_posture_path):
    # Load numpy arrays
    face_np = np.load(face_path)
    left_hand_np = np.load(left_hand_path)
    right_hand_np = np.load(right_hand_path)
    body_posture_np = np.load(body_posture_path)
         

    face = torch.from_numpy(face_np).float().cuda()
    left_hand = torch.from_numpy(left_hand_np).float().cuda()
    right_hand = torch.from_numpy(right_hand_np).float().cuda()
    body_posture = torch.from_numpy(body_posture_np).float().cuda()

    length = face.shape[0]
    # Prepare input
    source = [{
        "face": face,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "body_posture": body_posture,
        # Add dummy labels to match the expected input format
        "label_face": torch.zeros((length, 1)).cuda(),
        "label_left_hand": torch.zeros((length, 1)).cuda(),
        "label_right_hand": torch.zeros((length, 1)).cuda(),
        "label_body_posture": torch.zeros((length, 1)).cuda()
    }]
    
    # Extract features
    with torch.no_grad():
        result = model.extract_features(source, padding_mask=None, kmeans_labels=None, mask=False)
    
    
    # Extract layer outputs
    layer_outputs = []
    for layer in result['layer_results']:
        # layer_output has shape [T, B, D]
        # Since batch size B is 1, we can squeeze it
        layer_output = layer[-1]
        layer_output = layer_output.squeeze(1)  # Shape: [T, D]
        # layer_output = layer_output.squeeze(0)  # Shape: [T, D]
        layer_outputs.append(layer_output.cpu().numpy())  # Convert to NumPy array

    # Stack the outputs from all layers to get an array of shape [L, T, D]
    features = np.stack(layer_outputs, axis=0)  # Shape: [L, T, D]
    return features

# Main script
def main(csv_list, checkpoint_path, output_dir, index):
    model = load_model(checkpoint_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
        
    for row in csv_list:

        cues_list = row[0].split('\t')
        face_path, left_hand_path, right_hand_path, body_posture_path = cues_list[0], cues_list[1], cues_list[2], cues_list[3]

        output_filename = f"{os.path.basename(face_path).rsplit('.',1)[0].rsplit('_',1)[0]}.npy"
        output_path = os.path.join(output_dir, output_filename)
        
        # check if the output file already exists
        if os.path.exists(output_path):
            print(f"Skipping {output_path} as it already exists")
            continue
        
        # Process the sample
        features = process_sample(model, face_path, left_hand_path, right_hand_path, body_posture_path)

        np.save(output_path, features)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True,
                        help='index of the sub_list to work with')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='path to the CSV file')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='path to the checkpoint file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='directory to save output files')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch size for processing')
    args = parser.parse_args()
    index = args.index
    csv_path = args.csv_path
    checkpoint_path = args.checkpoint_path
    output_dir = args.output_dir
    batch_size = int(args.batch_size)

    # make output dir
    os.makedirs(output_dir, exist_ok=True)
    
    fixed_list = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            fixed_list.append(row)
    
    
    video_batches = [fixed_list[i:i + batch_size] for i in range(0, len(fixed_list), batch_size)]
    
    csv_list = video_batches[index]
    main(csv_list, checkpoint_path, output_dir, index)
