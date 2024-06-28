import streamlit as st
import requests
from PIL import Image
import io

st.title('MNIST Digit Recognition')

# Zone de dessin
canvas_result = st.image_input("Draw a digit (0-9)", type="pil")

if canvas_result is not None:
    # Afficher l'image dessinée
    st.image(canvas_result, caption='Drawn Image', use_column_width=True)
    
    # Convertir l'image pour l'envoyer à l'API
    buffered = io.BytesIO()
    canvas_result.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()

    # Bouton pour faire la prédiction
    if st.button('Predict'):
        files = {'file': ('image.png', image_bytes, 'image/png')}
        response = requests.post("http://backend:8000/api/v1/predict", files=files)
        
        if response.status_code == 200:
            prediction = response.json()['prediction']
            st.success(f'The predicted digit is: {prediction}')
        else:
            st.error('Error in prediction. Please try again.')