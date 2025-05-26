# 🧠 Brain Tumor Detection using VGG16 (Transfer Learning)

![Brain MRI](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Meningioma_MRI.jpg/640px-Meningioma_MRI.jpg)

This project uses a fine-tuned **VGG16** model to automatically detect and classify brain tumors from MRI images. It leverages the power of **transfer learning** to deliver high-accuracy classification with limited training data — a critical approach in medical AI.

---

## 📊 Overview

Brain tumors can be life-threatening without early diagnosis. Manual diagnosis from MRI images is time-consuming and subjective. This deep learning model aims to **classify brain MRI scans** as either **No Tumor** or one of the major tumor types using a pre-trained VGG16 model.

---

## 🧬 Dataset

Dataset used:  
🔗 [Brain MRI Images for Brain Tumor Detection – Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

**Classes:**
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

**Total Images**: ~3000  
**Image Type**: T1-weighted contrast-enhanced MRI

---

## 🚀 Technologies & Tools

- Python 3.x
- TensorFlow / Keras
- VGG16 (ImageNet weights)
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 🧠 Model Architecture

- Based on **VGG16** pretrained on ImageNet
- Custom fully connected head (FC layers) added for classification
- All convolutional base layers frozen initially, then selectively unfrozen for fine-tuning
- Input image size resized to **224x224**

```python
from tensorflow.keras.applications import VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
