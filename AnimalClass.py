# import streamlit as st
# import tensorflow as tf
# import cv2
# import numpy as np
# from PIL import Image
#
#
# # Load model
# model = tf.keras.models.load_model("animal_classifier.h5")
#
# # Define class names
# categories = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin',
#               'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion',
#               'Panda', 'Tiger', 'Zebra']
#
# # Prediction function
# def predict_image(image):
#     image = image.resize((224, 224))
#     img_array = np.array(image) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     pred = model.predict(img_array)
#     predicted_class = categories[np.argmax(pred)]
#     confidence = np.max(pred)
#     return predicted_class, confidence
#
# # Streamlit UI
# st.title("🐾 Animal Image Classifier")
# st.write("Upload an animal image and the model will tell you which animal it is!")
#
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)
#
#     if st.button("Predict"):
#         predicted_class, confidence = predict_image(image)
#         st.success(f"Prediction: **{predicted_class}** ({confidence*100:.2f}% confidence)")
#
#


import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import tempfile

# Load model
model = tf.keras.models.load_model("animal_classifier.h5")

# Define class names
categories = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin',
              'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion',
              'Panda', 'Tiger', 'Zebra']

# Prediction function using cv2
def predict_image_cv2(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img_array = img / 255.0
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
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Display image using OpenCV format
    file_bytes = np.asarray(bytearray(open(tmp_path, "rb").read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    st.image(opencv_image_rgb, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        predicted_class, confidence = predict_image_cv2(tmp_path)
        st.success(f"Prediction: **{predicted_class}** ({confidence * 100:.2f}% confidence)")
