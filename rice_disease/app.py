import os
import requests
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing import image
import numpy as np

@st.cache_resource
def load_trained_model():
    model_path = "Disease_dtct.h5"
    if not os.path.exists(model_path):
        url = "https://raw.githubusercontent.com/Ankith15/Rice_plant_Disease/main/rice_disease/Disease_dtct.h5"
        st.write("Downloading model from GitHub...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            st.error("Failed to download the model. Please check the URL.")
            st.stop()
    return load_model(model_path)

# Load the model
model = load_trained_model()

# Class labels
class_labels = ["Bacterial_leaf_blight", "Healthy_leaf", "Rice_Blast", "Tungro"]

st.title("Rice Plant Disease Detection")
st.write("Upload a folder containing rice plant leaf images")

# Upload a folder of images
upload_folder = st.file_uploader("Choose a folder containing images", type=None, accept_multiple_files=True)

if upload_folder:
    st.write("Uploaded images:")
    for uploaded_file in upload_folder:
        try:
            # Load and display each image
            image_data = Image.open(uploaded_file)
            st.image(image_data, caption=f"{uploaded_file.name}", use_column_width=True)
            
            # Preprocess the image
            def preprocess_image(img):
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                img_arr = image.img_to_array(img)
                img_arr = np.expand_dims(img_arr, axis=0)
                img_arr = img_arr / 255.0
                return img_arr
            
            processed_image = preprocess_image(image_data)

            # Make predictions
            predictions = model.predict(processed_image)
            predicted_class = class_labels[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

            # Display results
            st.write(f"Predicted class for {uploaded_file.name}: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}%")

        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
