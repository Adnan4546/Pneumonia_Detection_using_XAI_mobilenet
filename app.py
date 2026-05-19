import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


# --- MODEL LOADING (With Warm-up) ---
@st.cache_resource
def get_model():
    model = load_model('my_model.keras')
    # Pre-build/Warm-up
    model.predict(np.zeros((1, 224, 224, 3)))
    return model


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="out_relu"):
    # 1. Isolate the base model (MobileNetV2)
    base_model = model.layers[0]

    # 2. Create an "extractor" sub-model from the base model
    base_extractor = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[
            base_model.get_layer(last_conv_layer_name).output,
            base_model.output
        ]
    )

    # 3. Create a brand new Input tensor to map the entire new graph
    inputs = tf.keras.Input(shape=(224, 224, 3))

    # 4. Pass the input through the base extractor
    conv_output, x = base_extractor(inputs)

    # 5. Pass the result 'x' through the REMAINING layers of your Sequential model
    for layer in model.layers[1:]:
        x = layer(x)

    # 6. Build the final, fully-connected unified model
    grad_model = tf.keras.Model(inputs=inputs, outputs=[conv_output, x])

    # 7. Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]

    # 8. Compute gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 9. Process the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]

    # 10. Generate the heatmap
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap securely
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()


# --- UI INTERFACE ---
st.title("🫁 Pneumonia Detection & XAI")
st.write("Upload a Chest X-ray to see the AI's diagnosis and focus areas.")

uploaded_file = st.file_uploader("Choose an X-ray...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess Image
    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_dims = np.expand_dims(img_array, axis=0)

    # Load Model & Predict
    model = get_model()
    prediction = model.predict(img_dims)[0][0]

    # Interpretation
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Visualization
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("AI Prediction")
        color = "red" if label == "PNEUMONIA" else "green"
        st.markdown(f"### Result: :{color}[{label}]")
        st.write(f"Confidence: **{confidence:.2%}**")
        st.image(img, caption="Uploaded X-ray", use_container_width=True)

    with col2:
        st.subheader("XAI: Grad-CAM")
        heatmap = make_gradcam_heatmap(img_dims, model)

        # --- FIX: Safe uint8 processing pipeline ---
        # 1. Cast original image safely to 0-255 format
        img_np = np.array(img, dtype=np.uint8)

        # 2. Resize and scale raw heatmap values to 0-255
        heatmap_rescaled = cv2.resize(heatmap, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap_rescaled)

        # 3. Generate color mapping (Outputs BGR) and convert to RGB
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # 4. Blend the two matching 0-255 arrays together smoothly
        overlay = cv2.addWeighted(img_np, 0.6, heatmap_rgb, 0.4, 0)

        st.image(overlay, caption="Where the AI Focused", use_container_width=True)