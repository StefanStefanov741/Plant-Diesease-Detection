import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image

def get_folder_name_at_index(index):
    folder_path = "C:/Users/tetij/Desktop/IVP/Plant Dataset 4GB/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
    # Get a list of all folders in the specified path
    folder_list = [folder for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]

    # Sort the list of folders to ensure a consistent order
    folder_list.sort()

    # Check if the index is within the valid range
    if 0 <= index < len(folder_list):
        # Return the folder name at the specified index
        return folder_list[index]
    else:
        # If the index is out of range, return None or handle the error as needed
        return None

# Function to load the saved model
def load_model(model_path):
    # Load the entire model
    return torch.load(model_path)

# for moving data to device (CPU or GPU)
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Path to the saved model file
saved_model_path = 'plant-disease-model-complete.pth'

# Load the model
loaded_model = load_model(saved_model_path)

test_dir = "C:/Users/tetij/Desktop/IVP/Plant Dataset 4GB/test"
test_images = sorted(os.listdir(test_dir))

device = get_default_device()

# Use a list to store the paths of individual test images
test_image_paths = [os.path.join(test_dir, img) for img in test_images]

def predict_image(image_path, model):
    # Open and transform the image
    img = transforms.ToTensor()(Image.open(image_path))
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    # Return the predicted label index
    return preds[0].item()

# Predicting each image in the test directory
for image_path in test_image_paths:
    print('Image:', os.path.basename(image_path), ', Predicted Label:', get_folder_name_at_index(predict_image(image_path, loaded_model)))
