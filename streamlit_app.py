import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image  # Import the PIL library

# Load the pre-trained model
model = load_model('model_facial_skin_MobileNetv2.h5')

# Define the class labels
class_labels = ['blackhead', 'acne', 'ride0']

# Inside the predict function
def predict(img):
    # Make the prediction
    prediction = model.predict(img)

    # Print the prediction values for debugging
    print("Prediction values:", prediction)

    # Find the class index with the highest probability
    predicted_class_index = np.argmax(prediction)

    # Map the class index to the human-readable class label
    predicted_class_label = class_labels[predicted_class_index]

    print("Predicted class:", predicted_class_label)
    return predicted_class_label

# Streamlit app title
st.title("Welcome To Gerajeune App")
select = st.selectbox("Choose an option", ["Upload a picture", "Use your Webcam"])

if select == "Upload a picture":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image for prediction
        img = image.load_img(uploaded_image, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.0
        predicted_class = predict(img)

        # Display the predicted class
        st.subheader("Predicted Class:")
        st.write(predicted_class)

if select == "Use your Webcam":
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer:
        st.image(img_file_buffer)

        # To read image file buffer as a PIL Image:
        img = Image.open(img_file_buffer)
        img = img.convert('RGB')
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.0
        predicted_class = predict(img)

        # Display the predicted class
        st.subheader("Predicted Class:")
        st.write(predicted_class)
