import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

@st.cache_resource
def load_trained_model():
    model_path = "Disease_dtct.h5"  # Relative path
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    return load_model(model_path)

model = load_trained_model()
class_labels = ["Bacterial_leaf_blight", "Healthy_leaf", "Rice_Blast", "Tungro"]

st.title("Rice Plant Disease Detection")
st.write("Upload rice plant leaf image")

# Code for uploading an image
upload_file = st.file_uploader("Choose an image from your device", type=['jpg', 'png', 'jpeg', 'bmp'])

if upload_file is not None:
    image_data = Image.open(upload_file)
    st.image(image_data, caption="Uploaded Image", use_column_width=True)

    def preprocess_image(img):
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_arr = image.img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = img_arr / 255.0
        return img_arr

    processed_image = preprocess_image(image_data)

    # Making the predictions
    if model is not None:
        try:
            predictions = model.predict(processed_image)
            predicted_class = class_labels[np.argmax(predictions)]
            st.write(f"Predicted class: {predicted_class}")
            st.write(f"Confidence: {np.max(predictions) * 100:.2f}%")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
    else:
        st.error("Model is not loaded. Please check the model path.")
