# Lightweight Fruit Classification with MobileNetV2

This project presents a lightweight deep learning approach for fruit image classification using the MobileNetV2 architecture. The model leverages transfer learning and fine-tuning techniques to achieve high accuracy while maintaining computational efficiency.

The solution is designed for real-world computer vision applications, including mobile and edge-based deployment scenarios.

---

## 🚀 Project Overview

The objective of this project is to classify fruit images into predefined categories using a pre-trained MobileNetV2 backbone.

MobileNetV2 was selected due to its:
- Low parameter count
- Reduced computational cost
- Strong performance-to-efficiency ratio
- Suitability for mobile and embedded systems

The final model is optimized for both accuracy and lightweight inference.

---

## 🧠 Model Architecture

- Base Model: **MobileNetV2 (Pre-trained on ImageNet)**
- Technique: Transfer Learning + Fine-Tuning
- Custom classification head added for fruit categories
- Model saved in `.h5` format

---

## 📊 Implementation Platforms

### 🔹 Kaggle Notebook
Full training and experimentation notebook:
https://www.kaggle.com/code/leyuzakoksoken/lightweight-fruit-classification-with-mobilenetv2

### 🔹 Hugging Face Space (Live Demo)
Interactive demo application:
https://huggingface.co/spaces/leyuzak/Date-Fruit-CNN

---

## 📁 Project Structure

- `fruit_recognition_model.h5` – Trained model weights  
- `labels.json` – Class label mappings  
- `.ipynb` notebook – Model training and evaluation  
- `README.md` – Documentation  

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Matplotlib
- Kaggle
- Hugging Face Spaces

---

## 📈 Future Improvements

- Model quantization for further size reduction
- Real-time camera integration
- Deployment as a mobile application
- Dataset expansion for improved generalization

---

This project demonstrates the practical implementation of transfer learning using an efficient convolutional neural network for lightweight computer vision systems.
