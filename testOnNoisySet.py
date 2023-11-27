import os
import torch
import torchvision.transforms as transforms
from PIL import Image

#Function to load the saved model
def load_model(model_path,device):
    #Load the entire model
    return torch.load(model_path, map_location=device)

#for moving data to device (CPU or GPU)
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

#Path to the saved model file
saved_model_path = 'plant-disease-model-original.pth'

data_dir = "C:/Users/tetij/Desktop/IVP/Plant Dataset 4GB/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
#test_dir = "C:/Users/tetij/Desktop/IVP/Plant Dataset 4GB/noiseValid"
test_dir = "C:/Users/tetij/Desktop/IVP/Plant Dataset 4GB/valid"
#test_dir = "C:/Users/tetij/Desktop/IVP/Plant Dataset 4GB/test"
test_images = sorted(os.listdir(test_dir))
device = get_default_device()

#Load the model
loaded_model = load_model(saved_model_path,device)

def predict_single_image(image_path, model):
    #Open and transform the image
    img = transforms.ToTensor()(Image.open(image_path).convert('RGB'))  #Ensure the image is RGB
    #Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    #Get predictions from model
    yb = model(xb)
    #Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    #Return the predicted label index
    return preds[0].item()

def get_folder_name_at_index(index):
    folder_path = data_dir+"/train"
    #Get a list of all folders in the specified path
    folder_list = [folder for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]

    #Sort the list of folders to ensure a consistent order
    folder_list.sort()

    #Check if the index is within the valid range
    if 0 <= index < len(folder_list):
        #Return the folder name at the specified index
        return folder_list[index]
    else:
        #If the index is out of range, return None or handle the error as needed
        return None
    
def predict_dataset(dataset_path, model):
    #Get a list of all folders in the specified path
    folder_list = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

    #Sort the list of folders to ensure a consistent order
    folder_list.sort()

    #Initialize counters
    correct_predictions = 0
    incorrect_predictions = 0

    #Iterate through each folder
    for folder in folder_list:
        folder_path = os.path.join(dataset_path, folder)

        #Get a list of image files in the folder
        image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG'))]

        #Iterate through each image in the folder
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)

            #Get the predicted label for the image
            predicted_label = predict_single_image(image_path, model)

            #Get the actual label from the folder name
            actual_label = get_folder_name_at_index(predicted_label)

            #Check if the predicted label matches the actual label
            if actual_label == folder:
                correct_predictions += 1
            else:
                incorrect_predictions += 1

    return [correct_predictions, incorrect_predictions]

performance = predict_dataset("C:\\Users\\tetij\\Desktop\\IVP\\Plant Dataset 4GB\\valid",loaded_model)
print(performance)
print(f"Accuracy: {(performance[0]/(performance[0]+performance[1]))}")

#test_image_paths = [os.path.join(test_dir, image_name) for image_name in test_images]

#for image_path in test_image_paths:
#    print('Image:', os.path.basename(image_path), ', Predicted Label:', get_folder_name_at_index(predict_single_image(image_path, loaded_model)))
