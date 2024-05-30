import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import io, exposure, filters, measure, color, morphology
from scipy import ndimage as ndi

# Parameters
threshold_value = 0.9
border_width = 20
radius = 2.0
amount = 1.0
image_range = range(1, 1001)  # Adjust this range for your specific images

def unsharp_mask(image, radius, amount):
    blurred = filters.gaussian(image, sigma=radius)
    sharpened = image + amount * (image - blurred)
    return sharpened.clip(0, 1)

# Process each image in the folder
for i in image_range:
    image_path = f'Stone- ({i}).jpg'  
    try:
        original_image = io.imread(image_path)
        original_image = color.rgb2gray(original_image)
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        continue

    # Preprocessing
    original_image_eq = exposure.equalize_adapthist(original_image)
    original_image_rescaled = original_image_eq.astype('float32')
    sharpened_image = unsharp_mask(original_image_rescaled, radius, amount) 


    # Artifact and Middle-Third Removal (with Trapezoidal Mask)
    mask = np.ones_like(sharpened_image, dtype=bool)
    mask[:border_width, :] = False
    mask[-border_width:, :] = False
    mask[:, :border_width] = False
    mask[:, -border_width:] = False

    height, width = mask.shape
    top_width = width // 6 
    bottom_width = width // 6 

    top_center = width // 2
    bottom_center = width // 2

    for y in range(height):
        left_x = int(top_center - (top_width / 2) + (y / height) * ((bottom_width / 2) - (top_width / 2)))
        right_x = int(top_center + (top_width / 2) + (y / height) * ((bottom_width / 2) - (top_width / 2)))
        mask[y, left_x:right_x] = False

    # Thresholding and Morphology
    binary_image = sharpened_image > threshold_value
    binary_image = np.logical_and(binary_image, mask)  
    binary_image = morphology.remove_small_objects(binary_image, min_size=32)
    binary_image = ndi.binary_fill_holes(binary_image)

    # Save the mask
    mask_path = f'Stone- ({i})_mask.jpg'  # Name the mask accordingly
    io.imsave(mask_path, binary_image.astype(np.uint8) * 255)  # Save as grayscale image
