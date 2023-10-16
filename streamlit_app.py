# Streamlit
#Streamlit interface to visualize the results of pre-trained models.
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


# Charger le modèle pré-entraîné
model = load_model('model_facial_skin_MobileNetv2.h5')  # Assurez-vous que 'weights.h5' pointe vers votre modèle sauvegardé

# Définir les classes de problèmes possibles
# Inside the predict function
def predict(img):
    # Faire la prédiction
    prediction = model.predict(img)

    # Print the prediction values for debugging
    print("Prediction values:", prediction)

    # Trouver la classe avec la plus haute probabilité
    argmax = np.argmax(prediction)
    
    # Afficher les probabilités pour chaque classe
    st.subheader("Probabilités de chaque problème :")
    classes = ['blackhead', 'acne', 'ride0']
    for i in range(len(classes)):
        st.write(f"{classes[i]} : {prediction[0][i]}")

    # Find the class index with the highest probability
    predicted_class_index = np.argmax(prediction)
    print(predicted_class_index)
    # Map the class index to a human-readable class label
    class_labels = ["Class1", "Class2", "Class3"]
    predicted_class_label = class_labels[predicted_class_index]

    print("Predicted class:", predicted_class_label)
            
        
# Titre de l'application Streamlit
st.title("Welcome To Gerajeune App")
select = st.selectbox("Choose an option", ["Upload a picture", "Use your Webcam"])
if select == "Upload a picture":
    uploaded_image = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
    # Afficher l'image téléchargée
       st.image(uploaded_image, caption="Image téléchargée", use_column_width=True)
   
    # Prétraiter l'image pour la prédiction
       img = image.load_img(uploaded_image, target_size=(224, 224))
       img = image.img_to_array(img)
       img = np.expand_dims(img, axis=0)
       img /= 255.0
       predict(img)
if select =="Use your Webcam":
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer:
            st.image(img_file_buffer)
            # To read image file buffer as a PIL Image:
            img = Image.open(img_file_buffer)
            img = img.convert("L")
            img = img.convert('RGB')
            img = image.load_img(uploaded_image, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img /= 255.0
