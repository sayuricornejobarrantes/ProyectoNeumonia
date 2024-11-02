import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import wget
import zipfile
import os

# Tamaño de entrada de las imágenes
IMG_SIZE = (155, 155)

# Función para descargar y descomprimir el modelo
def download_and_extract_model():
    model_url = 'https://www.dropbox.com/scl/fi/1ze6340627igdnwj8y5fn/inception_v3_final_model.zip?rlkey=3jjnk2wr71ze8opp7y425sb7v&st=hab63yh8&dl=0'
    zip_path = 'inception_v3_final_model.zip'
    extract_folder = 'extracted_files'

    # Descargar el archivo zip si no existe
    if not os.path.exists(zip_path):
        try:
            wget.download(model_url, zip_path)
            st.success("Modelo descargado correctamente.")
        except Exception as e:
            st.error(f"Error al descargar el modelo: {e}")
            return None

    # Descomprimir el archivo
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    
    return os.path.join(extract_folder, 'inception_v3_final_model.keras')

modelo_path = download_and_extract_model()

# Verificar si el archivo del modelo existe
if not modelo_path or not os.path.exists(modelo_path):
    st.error("No se encontró el archivo del modelo.")
else:
    st.success("Archivo del modelo encontrado.")

# Definir el modelo base InceptionV3 sin la capa de clasificación superior
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
base_model.trainable = False

# Añadir capas de clasificación personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Cargar los pesos del modelo desde el archivo .keras
try:
    model.load_weights(modelo_path, skip_mismatch=True)
    st.success("Pesos del modelo cargados correctamente.")
except Exception as e:
    st.error(f"Error al cargar los pesos del modelo: {e}")

# Verificación de carga de archivo
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"], label_visibility="hidden")

if uploaded_file is not None:
    # Mostrar la imagen subida
    st.image(uploaded_file, caption="Imagen cargada")

    # Preprocesamiento de la imagen para hacer la predicción
    img = image.load_img(uploaded_file, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Realizar la predicción
    prediction = model.predict(img_array)

    # Mostrar resultados
    if prediction[0][0] < 0.5:
        st.success('El modelo predice que la imagen es de un **NORMAL**.')
    else:
        st.error('El modelo predice que la imagen es de un **PNEUMONIA**.')

    # f"Confianza de la preadicción: {prediction[0][0]:.4f}"

