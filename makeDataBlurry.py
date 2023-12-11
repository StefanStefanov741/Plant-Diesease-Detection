import os
import numpy as np
from PIL import Image, ImageFilter

def add_blur_and_save(original_path, output_path):
    #Load the original image
    image = Image.open(original_path)

    #Apply Gaussian blur to the image
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=3))

    #Convert RGBA to RGB if the image has an alpha channel
    if blurred_image.mode == 'RGBA':
        blurred_image = blurred_image.convert('RGB')

    #Save the blurred image
    blurred_image.save(output_path)

def process_folder(folder_path):
    #Get the list of files in the folder
    files = os.listdir(folder_path)

    #Process each file in the folder
    for file in files:
        #Check if the file is an image
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
            #Generate the paths for the original and blurred images
            original_image_path = os.path.join(folder_path, file)
            #blurred_image_path = os.path.join(folder_path, file.replace('.', 'blurred.'))  #Use this to add a blurred image to the dataset
            blurred_image_path = original_image_path  #Use this to replace the original image with a blurred version

            #Add blur and save the blurred image
            add_blur_and_save(original_image_path, blurred_image_path)

def process_folders(base_path):
    #Get the list of folders in the base path
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    #Process each folder
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        process_folder(folder_path)

if __name__ == '__main__':
    #Specify the base path where the folders are located
    base_path = 'C:\\Users\\tetij\\Desktop\\IVP\\Plant Dataset 4GB\\blurValid\\'

    #Process folders and add blur to images
    process_folders(base_path)
