import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load model
model = tf.keras.models.load_model("animal_classifier.h5")

# Define class names
categories = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin',
              'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion',
              'Panda', 'Tiger', 'Zebra']

# Prediction function
def predict_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    predicted_class = categories[np.argmax(pred)]
    confidence = np.max(pred)
    return predicted_class, confidence

# Streamlit UI
st.title("🐾 Animal Image Classifier")
st.write("Upload an animal image and the model will tell you which animal it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        predicted_class, confidence = predict_image(image)
        st.success(f"Prediction: **{predicted_class}** ({confidence*100:.2f}% confidence)")
