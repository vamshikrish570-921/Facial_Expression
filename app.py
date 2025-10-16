import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from io import BytesIO

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('expression_model.h5')

model = load_model()
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

st.title("😊 Facial Expression Recognition App")
st.write("Upload an image and detect the facial emotion!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # ✅ Read bytes once and keep them safe
        file_bytes = uploaded_file.getvalue()

        # ✅ Try opening image safely
        image = Image.open(BytesIO(file_bytes)).convert("RGB")

        st.image(image, caption='Uploaded Image', use_column_width=True)

        # ✅ Preprocess image for model
        img = ImageOps.grayscale(image)
        img = img.resize((48, 48))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))

        prediction = model.predict(img_array)
        emotion = labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.markdown(f"### 🧠 Prediction: **{emotion.capitalize()}**")
        st.markdown(f"**Confidence:** {confidence:.2f}")

    except UnidentifiedImageError:
        st.error("⚠️ Could not identify the uploaded image. Please upload a valid JPG or PNG file.")
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {e}")
