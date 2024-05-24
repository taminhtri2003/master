import streamlit as st
import cv2
from skimage import io, exposure, filters, morphology, color, measure
from scipy import ndimage as ndi  # Import ndi correctly
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# Load the trained kidney stone detection model
model = load_model('kidney_stone_classifier.h5')

# --- Image Processing Functions ---
def unsharp_mask(image, radius, amount):
    blurred = filters.gaussian(image, sigma=radius)
    sharpened = image + amount * (image - blurred)
    return sharpened.clip(0, 1)


# --- Streamlit App ---
st.title("Kidney Stone Detection with Image Masking")

uploaded_file = st.file_uploader("Choose an image (JPG, PNG, etc.)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Step 1: Image Import and Preprocessing
    original_image = io.imread(uploaded_file)

    # Check if image is grayscale and convert if necessary
    if len(original_image.shape) == 3 and original_image.shape[2] == 3:  # Check for RGB image
        original_image_grey = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    else:
        original_image_grey = original_image.copy()

    original_image_eq = exposure.equalize_adapthist(original_image_grey)  # Use grayscale image
    original_image_rescaled = original_image_eq.astype('float32')
    sharpened_image = unsharp_mask(original_image_rescaled, radius=2.0, amount=1.0)

    # Step 2: Mask Extraction Parameters
    threshold_value = 0.85
    border_width = 20

    # Step 2: Mask Extraction 
    mask = np.ones_like(sharpened_image, dtype=bool)
    mask[:border_width, :] = False
    mask[-border_width:, :] = False
    mask[:, :border_width] = False
    mask[:, -border_width:] = False

    # Trapezoidal Mask Creation
    height, width = mask.shape
    top_width = width // 6
    bottom_width = width // 6
    top_center = width // 2
    bottom_center = width // 2

    for y in range(height):
        left_x = int(top_center - (top_width / 2) + (y / height) * ((bottom_width / 2) - (top_width / 2)))
        right_x = int(top_center + (top_width / 2) + (y / height) * ((bottom_width / 2) - (top_width / 2)))
        mask[y, left_x:right_x] = False
    
    binary_image = sharpened_image > threshold_value
    binary_image = np.logical_and(binary_image, mask) 
    binary_image = morphology.remove_small_objects(binary_image, min_size=64)
    binary_image = ndi.binary_fill_holes(binary_image)
    
    binary_image = binary_image.astype(np.uint8) * 255

    if len(binary_image.shape) == 3 and binary_image.shape[2] == 3:  # Check for RGB image
        binary_image_check = cv2.cvtColor(binary_image, cv2.COLOR_RGB2GRAY)
    else:
        binary_image_check = binary_image.copy()

    binary_image_check = cv2.resize(binary_image_check, (128, 128))
    binary_image_check = binary_image_check / 255.0
    binary_image_check = binary_image_check.reshape(1, 128, 128, 1) 

    # Step 3: Apply Mask and Use for Prediction
    prediction = model.predict(binary_image_check)
    class_label = np.argmax(prediction)

    # Display results
    st.image(original_image, caption="Uploaded Image", use_column_width=True)
    st.subheader('Mask Extraction and Detection')
    st.image(binary_image, caption="Binary Mask", use_column_width=True)

    st.subheader('Prediction:')
    if class_label == 0:
        st.write("Normal Kidney (No Stones)")
    else:
        st.write("Kidney Stone Detected")

    # Display prediction probabilities
    prob_normal_kidney = prediction[0][0]
    prob_kidney_stone = prediction[0][1]

    st.subheader("Prediction Probabilities:")
    st.write("Probability of being normal kidney:  ", prob_normal_kidney)
    st.write("Probability of having kidney stone: ", prob_kidney_stone)

    # Plot probabilities
    fig, ax = plt.subplots()
    ax.bar(["Normal Kidney", "Kidney Stone"], [prob_normal_kidney, prob_kidney_stone])
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")
    st.pyplot(fig)
