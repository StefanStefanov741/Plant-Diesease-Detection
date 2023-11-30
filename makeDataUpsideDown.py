import os
from PIL import Image, ImageOps

def flip_upside_down_and_save(original_path, output_path):
    #Load the original image
    image = Image.open(original_path)

    #Flip the image upside down
    flipped_image = ImageOps.flip(image)

    #Save the flipped image
    flipped_image.save(output_path)

def process_folder(folder_path):
    #Get the list of files in the folder
    files = os.listdir(folder_path)

    #Process each file in the folder
    for file in files:
        #Check if the file is an image (you can add more image extensions if needed)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
            #Generate the paths for the original and flipped images
            original_image_path = os.path.join(folder_path, file)
            flipped_image_path = original_image_path  #Use this to replace the original image with a flipped version

            #Flip upside down and save the flipped image
            flip_upside_down_and_save(original_image_path, flipped_image_path)

def process_folders(base_path):
    #Get the list of folders in the base path
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    #Process each folder
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        process_folder(folder_path)

if __name__ == '__main__':
    #Specify the base path where your folders are located
    base_path = 'C:\\Users\\tetij\\Desktop\\IVP\\Plant Dataset 4GB\\flippedValid\\'

    #Process folders and flip images upside down
    process_folders(base_path)
