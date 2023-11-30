import os
import numpy as np
from PIL import Image

def add_noise_and_save(original_path, output_path):
    #Load the original image
    image = np.array(Image.open(original_path))

    #Generate noise with the same shape as the image
    noise = np.random.normal(loc=0, scale=150 , size=image.shape)

    #Add noise to the image
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)

    #Convert the NumPy array to a PIL Image
    noisy_image_pil = Image.fromarray(noisy_image)

    #Convert RGBA to RGB if the image has an alpha channel
    if noisy_image_pil.mode == 'RGBA':
        noisy_image_pil = noisy_image_pil.convert('RGB')

    #Save the noisy image
    noisy_image_pil.save(output_path)

def process_folder(folder_path):
    #Get the list of files in the folder
    files = os.listdir(folder_path)

    #Process each file in the folder
    for file in files:
        #Check if the file is an image (you can add more image extensions if needed)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
            #Generate the paths for the original and noisy images
            original_image_path = os.path.join(folder_path, file)
            noisy_image_path = os.path.join(folder_path, file.replace('.', '2.')) #Use this to add a second noisy image to dataset
            #noisy_image_path = original_image_path #Use this to replace the original image with a noisy version

            #Add noise and save the noisy image
            add_noise_and_save(original_image_path, noisy_image_path)

def process_folders(base_path):
    #Get the list of folders in the base path
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    #Process each folder
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        process_folder(folder_path)

if __name__ == '__main__':
    #Specify the base path where your folders are located
    base_path = 'C:\\Users\\tetij\\Desktop\\IVP\\Plant Dataset 4GB\\trainWithNoise\\'

    #Process folders and add noise to images
    process_folders(base_path)
