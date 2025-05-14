import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageEnhance
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import os

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")

# ---- CUSTOM CSS ----
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #f3e5f5, #e1f5fe);
    font-size: 18px;
}
.main h1 {
    font-size: 48px !important;
}
</style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown("""
    <h1 style='text-align: center; color: #1a237e;'>🧠 Brain Tumor Segmentation</h1>
    <p style='text-align: center;'>Upload an MRI image and generate a tumor mask using your trained segmentation model.</p>
""", unsafe_allow_html=True)

# ---- SIDEBAR ----
with st.sidebar:
    st.header("⚙️ Options")
    st.write("Model: U-Net (Keras)")
    st.markdown("---")
    st.write("📞 Contact: brainai@appsupport.com")
    st.write("🔗 [GitHub](https://github.com/SouMahdi/segmentation_app)")

# ---- LOAD MODEL ----
st.markdown("### 🧠 Loading Model...")
model = load_model("segmentation_model.h5")
st.success("Segmentation model loaded successfully!")

# ---- IMAGE UPLOAD ----
st.markdown("### 📤 Upload an MRI Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    resized = image.resize((256, 256))
    array_img = img_to_array(resized) / 255.0
    input_img = np.expand_dims(array_img, axis=0)

    # Predict mask
    if st.button("🚀 Segment Tumor"):
        prediction = model.predict(input_img)[0]
        mask = (prediction > 0.5).astype(np.uint8) * 255
        mask = np.squeeze(mask)

        # Resize back to original image size
        mask_img = Image.fromarray(mask).resize(image.size)

        # Overlay mask on original
        overlay = ImageEnhance.Brightness(mask_img.convert("RGB")).enhance(1.5)
        blended = Image.blend(image, overlay, alpha=0.5)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.image(image, caption="🧠 Original Image", use_column_width=True)
        with col2:
            st.image(mask_img, caption="🔍 Predicted Mask", use_column_width=True)
        with col3:
            st.image(blended, caption="🎯 Overlay", use_column_width=True)

else:
    st.info("Please upload an image to get started.")
