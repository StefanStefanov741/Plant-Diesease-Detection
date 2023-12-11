import os
import random
from PIL import Image

def apply_random_rotation_and_save(original_path, output_path):
    #Load the original image
    image = Image.open(original_path)

    #Generate a random rotation angle between -180 and 180 degrees
    angle = random.uniform(-180, 180)

    #Rotate the image by the random angle
    rotated_image = image.rotate(angle)

    #Save the rotated image
    rotated_image.save(output_path)

def process_folder(folder_path):
    #Get the list of files in the folder
    files = os.listdir(folder_path)

    #Process each file in the folder
    for file in files:
        #Check if the file is an image
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
            #Generate the paths for the original and rotated images
            original_image_path = os.path.join(folder_path, file)
            rotated_image_path = original_image_path  #Use this to replace the original image with a rotated version

            #Apply random rotation and save the rotated image
            apply_random_rotation_and_save(original_image_path, rotated_image_path)

def process_folders(base_path):
    #Get the list of folders in the base path
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    #Process each folder
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        process_folder(folder_path)

if __name__ == '__main__':
    #Specify the base path where the folders are located
    base_path = 'C:\\Users\\tetij\\Desktop\\IVP\\Plant Dataset 4GB\\rotatedValid\\'

    #Process folders and apply random rotations to images
    process_folders(base_path)
