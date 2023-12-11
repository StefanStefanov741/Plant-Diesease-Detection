import os
from PIL import Image, ImageEnhance

def adjust_brightness_contrast_and_save(original_path, output_path, brightness_factor=1.5, contrast_factor=1.5):
    #Load the original image
    image = Image.open(original_path)

    #Adjust brightness and contrast
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    #Convert RGBA to RGB if the image has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    #Save the modified image
    image.save(output_path)

def process_folder_with_brightness_contrast(folder_path):
    #Get the list of files in the folder
    files = os.listdir(folder_path)

    #Process each file in the folder
    for file in files:
        #Check if the file is an image
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
            #Generate the paths for the original and modified images
            original_image_path = os.path.join(folder_path, file)
            #output_image_path = os.path.join(folder_path, file.replace('.', 'adjusted.'))  #Use this to add an adjusted image to the dataset
            output_image_path = original_image_path  #Use this to replace the original image with an adjusted version

            #Adjust brightness and contrast and save the modified image
            adjust_brightness_contrast_and_save(original_image_path, output_image_path)

def process_folders_with_brightness_contrast(base_path):
    #Get the list of folders in the base path
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    #Process each folder
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        process_folder_with_brightness_contrast(folder_path)

if __name__ == '__main__':
    #Specify the base path where the folders are located
    base_path = 'C:\\Users\\tetij\\Desktop\\IVP\\Plant Dataset 4GB\\exposureValid\\'

    #Process folders and add brightness and contrast to images
    process_folders_with_brightness_contrast(base_path)
