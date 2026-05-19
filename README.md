# Pneumonia_Detection_using_XAI_mobilenet
# 🫁 Chest X-Ray Pneumonia Detection with Grad-CAM (XAI)

An interactive Streamlit web application that uses a deep learning model to detect Pneumonia from chest X-rays. To build trust in the AI's predictions, the app integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to highlight exactly which regions of the lungs influenced the network's classification.

## 🚀 Features
* **Automated Diagnosis:** Classifies chest X-rays into **NORMAL** or **PNEUMONIA** categories with a calculated confidence percentage.
* **Explainable AI (XAI):** Generates a visual heat map (using Grad-CAM) superimposed smoothly onto the X-ray to show focal areas of infection or inflammation.
* **Streamlined UI:** A clean, dual-column layout built with Streamlit for quick image uploading and side-by-side analysis.

## 🛠️ How It Works (Code Architecture)

### 1. Model Loading & Optimization
The app loads a fine-tuned deep learning model (built on a MobileNetV2 backbone). It utilizes Streamlit's `@st.cache_resource` decorator to load the model into memory only once, and runs a dummy tensor "warm-up" prediction to eliminate lagging on the user's first upload.

### 2. The Grad-CAM Pipeline
The explainability layer works by capturing the spatial information preserved in the final convolutional layers:
* It extracts the outputs of the last convolutional layer (`out_relu`) alongside the model's final predictions using a custom `tf.keras.Model` graph.
* Utilizing `tf.GradientTape`, the script calculates the gradients of the predicted pneumonia score with respect to the feature maps of that last conv layer.
* These gradients are globally pooled to weigh the importance of each feature channel.
* A matrix multiplication combines the weights and feature maps, creating a 2D heatmap that is normalized and upscaled via OpenCV (`cv2`) to seamlessly match the original input dimensions ($224 \times 224 \times 3$).

## 📦 Installation & Local Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git)
   cd YOUR-REPO-NAME
