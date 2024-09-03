import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import time
import os
import matplotlib.pyplot as plt
import json

from models.Bisenet import BiSeNet
import argparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(
                    # prog='FastSCNN Semantic Segmentation',
                    description='Takes video and creates semantic segmentation',
                    epilog='Text at the bottom of help')
parser.add_argument('-f', '--filename')           # positional argument
parser.add_argument('-m', '--model_name')           # positional argument
parser.add_argument('-d', '--directory')
args = parser.parse_args()

folder_name = args.directory#"scnn_crossEntropyLoss_Weights"
model_name = args.model_name#"scnn_crossEntropyLoss_Weights_1e-5lr_60epochs_after_1e-3lr_30epochs"

assert model_name, "Model name is required"
assert folder_name, "Directory is required"
assert args.filename, "Filename is required"

if model_name.endswith('.pth'):
    model_name = model_name[:-4]  # Remove the file extension
    
base_path = "."
video_path = f"{base_path}/videos/{args.filename}"
model_path = f"{base_path}/files/{folder_name}"
weights_path = f"{model_path}/{model_name}.pth"

config_path = f"{base_path}/config_v2.0.json"
results_path = f"{base_path}/results"
output_path = f"{results_path}/{args.filename}_{model_name}.mp4"

assert os.path.exists(video_path), f"Video file not found at {video_path}"
assert os.path.exists(model_path), f"Model folder not found at {model_path}"
assert os.path.exists(weights_path), f"Model weights not found at {weights_path}"


#Config map
config = json.load(open(config_path))
config_map = {label: config['labels'][label]['color'] 
              for label in range(len(config['labels']))}  

# Define transformation (resize, normalize, etc.) if necessary
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((1024, 2048)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Add other necessary transforms like normalization here
])

# Load pre-trained model 
model = BiSeNet(num_classes=124, context_path='resnet18')
model = model.to(device)
model.load_state_dict(torch.load(weights_path, weights_only=True))
model.eval()

# Load the video
# video_path = 'input_video.mp4'
cap = cv2.VideoCapture(video_path)

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                        (2048, 1024))
                    #   (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

frames = 0
overall_reading_time = 0
overall_transform_time = 0
overall_model_time = 0
overall_writing_time = 0

torch.cuda.empty_cache()
#Apply config map
colormap = torch.tensor([config_map[i] for i in range(len(config_map))], dtype=torch.uint8).to(device)

while cap.isOpened():
    # print(i, end=" ")
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    frames += 1
    reading_time = time.time()
    # Convert the frame to PIL image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Apply the necessary transformations
    input_tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension
    
    # Move the input tensor to the appropriate device (GPU/CPU)
    input_tensor = input_tensor.to(device)
    transform_time = time.time()
    
    # Perform segmentation
    with torch.no_grad():
        output = model(input_tensor)
    model_time = time.time()
    
    # Process the output to get the segmentation mask
    # Assuming the output is a logit tensor, apply softmax or argmax as needed
    segmentation = output.argmax(dim=1).squeeze().detach()

    # Optionally, map class indices to colors
    segmentation = colormap[segmentation].to('cpu').numpy()  # Define this function as needed

    # Convert segmentation mask to a format suitable for video saving
    # segmentation = np.uint8(segmentation * 255)  # Example: binary mask
    # segmentation = cv2.cvtColor(segmentation, cv2.COLOR_GRAY2BGR)
    
    # Write the frame to the output video
    out.write(segmentation)
    writing_time = time.time()

    overall_reading_time += reading_time - start
    overall_transform_time += transform_time - reading_time
    overall_model_time += model_time - transform_time
    overall_writing_time += writing_time - model_time 
# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows() 

#Printout
print(f"Number of frames: {frames}")
print(f"Reading time: {overall_reading_time}")
print(f"Transform time: {overall_transform_time}")
print(f"Model time: {overall_model_time}")
print(f"Writing time: {overall_writing_time}")
print(f"Total time: {overall_reading_time + overall_transform_time + overall_model_time + overall_writing_time}")