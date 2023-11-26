import numpy as np
from PIL import Image

# Load your image
image = np.array(Image.open('C:/Users/tetij/Desktop/IVP/Plant Dataset 4GB/test/blightPotatoNoisy.JPG'))

# Generate noise with the same shape as the image, 7 times stronger
noise = np.random.normal(loc=0, scale=25 * 2, size=image.shape)

# Add noise to the image
noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)

# Convert the NumPy array to a PIL Image
noisy_image_pil = Image.fromarray(noisy_image)

# Convert RGBA to RGB if the image has an alpha channel
if noisy_image_pil.mode == 'RGBA':
    noisy_image_pil = noisy_image_pil.convert('RGB')

# Save the noisy image
noisy_image_pil.save('C:/Users/tetij/Desktop/IVP/Plant Dataset 4GB/test/blightPotatoNoisy2.JPG')
