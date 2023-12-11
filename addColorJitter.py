import numpy as np
from PIL import Image, ImageEnhance

def adjust_brightness_contrast(image_path, output_path, brightness_factor=1.5, contrast_factor=1.5):
    #Load image
    image = np.array(Image.open(image_path))

    #Convert NumPy array to a PIL Image
    pil_image = Image.fromarray(image)

    #Adjust brightness and contrast
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(brightness_factor)

    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(contrast_factor)

    #Save the modified image
    pil_image.save(output_path)

if __name__ == '__main__':
    #Specify input and output paths
    image_path = 'C:/Users/tetij/Desktop/IVP/Plant Dataset 4GB/test/blightPotatoNoisy.JPG'
    output_path = 'C:/Users/tetij/Desktop/IVP/Plant Dataset 4GB/test/blightPotatoAdjusted.JPG'

    #Adjust brightness and contrast (you can adjust the factors as needed)
    adjust_brightness_contrast(image_path, output_path, brightness_factor=1.5, contrast_factor=1.5)
