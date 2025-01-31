import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Charger le modèle
model = load_model('https://drive.google.com/file/d/1-13ZdUcbvSR03heHeCq27fiXkoAsWQfW/view?usp=sharing')

# Fonction pour prétraiter l'image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # VGG16 attend des images de taille 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normaliser l'image
    return img_array

# Fonction pour faire la prédiction
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    return "Pneumonie" if prediction[0][0] > 0.5 else "Normal"

# Interface Streamlit
st.title("🩺 Classification de Radiographie Thoracique")
st.write("Cette application utilise un modèle VGG16 pour classifier les radiographies thoraciques en **Normal** ou **Pneumonie**.")

uploaded_file = st.file_uploader("📤 Téléchargez une image de radiographie thoracique", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Charger l'image
    image_display = Image.open(uploaded_file)
    
    # Sauvegarder temporairement pour la prédiction
    img_path = "temp_image.jpg"
    image_display.save(img_path)
    prediction = predict_image(img_path)
    
    # Définir la couleur en fonction du résultat
    color = "#28a745" if prediction == "Normal" else "#dc3545"
    
    # Affichage en colonne
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image_display, caption='🖼️ Image téléchargée', use_container_width=True)
    
    with col2:
        st.markdown(f"<div style='display: flex; justify-content: center; align-items: center; height: 100%;'><h3 style='text-align: center; color: {color};'>{prediction}</h3></div>", unsafe_allow_html=True)
