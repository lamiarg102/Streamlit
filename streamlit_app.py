
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
classes = ['blackhead', 'acne', 'ride0']

# Titre de l'application Streamlit
st.title("Détection de Problèmes de Peau")

# Ajouter une option pour télécharger une image
uploaded_image = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
 # Afficher l'image téléchargée
    st.image(uploaded_image, caption="Image téléchargée", use_column_width=True)

 # Prétraiter l'image pour la prédiction
    img = image.load_img(uploaded_image, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0

# Faire la prédiction
    prediction = model.predict(img)

# Afficher les probabilités pour chaque classe
    st.subheader("Probabilités de chaque problème :")
    for i in range(len(classes)):
        st.write(f"{classes[i]} : {prediction[0][i]}")

# Afficher la classe prédite
    predicted_class = classes[np.argmax(prediction)]
    st.write(f"Classe prédite : {predicted_class}")
