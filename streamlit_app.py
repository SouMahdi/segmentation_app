import streamlit as st
import numpy as np
import cv2
import os
import gdown
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ---------------------------- CONFIG ----------------------------
MODEL_FILENAME = "unet_mobilenetv2_finetuned_model.h5"
DRIVE_FILE_ID = "1jz5EMrf-DILqHg33MWXstPHewvrMCvgu"  
INPUT_SIZE = (128, 128)
THRESHOLD = 0.5  # For converting probabilities to binary mask

# -------------------------- STYLING -----------------------------
st.set_page_config(page_title="🧠 Tumor Segmentation", layout="wide")

st.markdown("""
    <style>
        .main {
            background: linear-gradient(90deg, #f2f4f8, #e9ecf3);
            font-size: 18px;
        }
        h1, h2, h3 {
            color: #004477;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Brain Tumor Segmentation")
st.markdown("Upload an MRI scan to visualize predicted tumor regions using a fine-tuned U-Net model.")

# ------------------------ LOAD MODEL -----------------------------
@st.cache_resource
def load_segmentation_model():
    if not os.path.exists(MODEL_FILENAME):
        gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_FILENAME, quiet=False)
    model = load_model(MODEL_FILENAME)
    return model

model = load_segmentation_model()

# --------------------- IMAGE PREPROCESSING -----------------------
def preprocess_image(image: Image.Image):
    image = image.resize(INPUT_SIZE)
    array = img_to_array(image) / 255.0
    array = np.expand_dims(array, axis=0)
    return array

def postprocess_mask(mask):
    mask = (mask > THRESHOLD).astype(np.uint8)[0, :, :, 0]  # shape: (224,224)
    return mask

def overlay_mask(image, mask, alpha=0.5):
    image = np.array(image.resize(INPUT_SIZE)).astype(np.uint8)
    red_mask = np.zeros_like(image)
    red_mask[:, :, 0] = 255  # red channel
    overlay = np.where(mask[..., None] == 1, red_mask, image)
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return blended

# ---------------------------- SIDEBAR -----------------------------
with st.sidebar:
    st.header("📁 Upload MRI")
    uploaded_file = st.file_uploader("Choose a brain MRI image", type=["jpg", "jpeg", "png"])

# ------------------------ MAIN LOGIC -----------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Running segmentation...")
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    mask = postprocess_mask(prediction)

    # Overlay mask
    result = overlay_mask(image, mask)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.markdown("### Tumor Segmentation Overlay")
        st.image(result, use_column_width=True)

    with st.expander("View raw binary mask"):
        st.image(mask * 255, caption="Binary Mask", clamp=True, use_column_width=True)

else:
    st.info("Please upload an image using the sidebar to begin segmentation.")
