import numpy as np
from PIL import Image, ImageFilter

#Load image
image_path = 'C:/Users/tetij/Desktop/IVP/Plant Dataset 4GB/test/blightPotatoNoisy.JPG'
image = np.array(Image.open(image_path))

#Apply Gaussian blur to the image
blurred_image = Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=1))

#Convert RGBA to RGB if the image has an alpha channel
if blurred_image.mode == 'RGBA':
    blurred_image = blurred_image.convert('RGB')

#Save the blurred image
blurred_image.save('C:/Users/tetij/Desktop/IVP/Plant Dataset 4GB/test/blightPotatoBlurred.JPG')
