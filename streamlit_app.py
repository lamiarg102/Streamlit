

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
def predict(img):
 
# Faire la prédiction
    prediction = model.predict(img)

# Afficher les probabilités pour chaque classe
    st.subheader("Welcome To Gerajeune App")
    classes = ['blackhead', 'acne', 'ride0']

    for i in range(len(classes)):
        st.write(f"{classes[i]} : {prediction[0][i]}")

# Afficher la classe prédite
    argmax = np.argmax(prediction)
    
    if 0 <= argmax < len(classes):
        predicted_class = classes[argmax]
        st.write(f"Classe prédite : {predicted_class}")
    else:
        st.write("Classe prédite : Classe inconnue")

# Titre de l'application Streamlit
st.title("Détection de Problèmes de Peau")
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
