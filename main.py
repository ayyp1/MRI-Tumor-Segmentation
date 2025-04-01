
import streamlit as st
import cv2
import numpy as np
from inference import evaluate
from PIL import Image

st.title("Brain Tumor MRI Segmentation")
brain = cv2.imread("brain.jpg")
st.image(brain, channels='RGB')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif"])
if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension == 'tif':
        pil_image = Image.open(uploaded_file)
        pil_image = pil_image.convert("RGB")
        original_image = np.array(pil_image)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    predicted_image = evaluate(original_image)
    
    if predicted_image is None:
        st.error("Error: The prediction function returned None. Please check the 'evaluate' function in inference.py.")
    else:
        if predicted_image.max() <= 1:
            predicted_image = (predicted_image * 255).astype(np.uint8)
        
        if len(predicted_image.shape) == 2:
            _, mask = cv2.threshold(predicted_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif len(predicted_image.shape) == 3:
            gray_mask = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2GRAY) if predicted_image.shape[2] == 3 else cv2.cvtColor(predicted_image, cv2.COLOR_RGBA2GRAY)
            _, mask = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            st.error("Unexpected mask format")
            st.stop()
        
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        mask_resized = cv2.resize(mask, (original_rgb.shape[1], original_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = np.zeros_like(original_rgb, dtype=np.uint8)
        overlay[mask_resized == 255] = [0, 255, 255]
        
        alpha, beta = 0.4, 1.0
        masked_image = original_rgb.copy()
        tumor_area = mask_resized > 0
        
        if np.count_nonzero(tumor_area) > 50:
            masked_image[tumor_area] = cv2.addWeighted(original_rgb[tumor_area], beta, overlay[tumor_area], alpha, 0.0)
        else:
            st.warning("No significant tumor area detected.")
        
        predicted_rgb = cv2.cvtColor(predicted_image, cv2.COLOR_GRAY2RGB) if len(predicted_image.shape) == 2 else cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
        
        st.subheader("Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(original_image, channels="BGR", caption="Original", use_container_width = True )
        with col2:
            st.image(predicted_rgb, channels="RGB", caption="Predicted Mask" , use_container_width = True)
        with col3:
            st.image(masked_image, channels="RGB", caption="Overlay",use_container_width = True )
else:
    st.write("Please upload an image to see the prediction.")
