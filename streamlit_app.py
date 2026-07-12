import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Potato Disease Detection",
    page_icon="🥔"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "potato_disease_model.keras",
        compile=False
    )

model = load_model()

class_names = ["Early Blight", "Late Blight", "Healthy"]

st.title("🥔 Potato Disease Detection")
st.write("Upload a potato leaf image to detect disease.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = image.resize((224, 224))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
