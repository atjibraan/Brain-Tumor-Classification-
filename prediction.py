#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import io

# Constants
IMAGE_SIZE = (150, 150)  # Change this to your model's expected input size
CLASSES = ['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']

# Load model once
@st.cache_resource
def load_trained_model():
    model = load_model('bt.h5')
    return model

# Preprocess uploaded image
def preprocess_image(uploaded_file, image_size):
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize(image_size)
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, image

# Predict class
def predict_image(model, image_array):
    prediction = model.predict(image_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))
    predicted_class = CLASSES[class_idx]
    return predicted_class, confidence, prediction

# Streamlit App UI
st.set_page_config(page_title="Brain MRI Tumor Classifier", layout="centered")
st.title("🧠 Brain MRI Tumor Detection")
st.markdown("Upload a brain MRI scan and get the predicted tumor type.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with st.spinner("Processing image and predicting..."):
        image_array, display_image = preprocess_image(uploaded_file, IMAGE_SIZE)
        model = load_trained_model()
        predicted_class, confidence, all_probs = predict_image(model, image_array)

        st.image(display_image, caption="Uploaded MRI Image", use_column_width=True)
        st.markdown(f"### ✅ Predicted Class: `{predicted_class}`")
        st.markdown(f"### 📊 Confidence: `{confidence*100:.2f}%`")

        # Show probabilities for all classes
        st.markdown("#### 🔍 Class Probabilities:")
        for i, class_name in enumerate(CLASSES):
            st.progress(float(all_probs[0][i]))

else:
    st.info("Please upload an MRI image to proceed.")

