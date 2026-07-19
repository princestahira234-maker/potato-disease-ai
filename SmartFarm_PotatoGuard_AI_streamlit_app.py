# SmartFarm PotatoGuard AI
# Intelligent Potato Disease Detection System
# Streamlit Cloud Compatible

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="SmartFarm PotatoGuard AI",
    page_icon="🌾",
    layout="wide"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "potato_disease_model.keras",
        compile=False
    )

model = load_model()
class_names = ["Early Blight", "Late Blight", "Healthy"]

st.markdown("""
<style>
.hero{
background:linear-gradient(135deg,#1B5E20,#43A047);
padding:40px;border-radius:20px;color:white;text-align:center;
}
</style>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["Home","Detection","Disease Guide","About"]
)

if page == "Home":
    st.markdown("""
    <div class="hero">
    <h1>🌾 SmartFarm PotatoGuard AI</h1>
    <h3>Intelligent Potato Disease Detection System</h3>
    </div>
    """, unsafe_allow_html=True)

elif page == "Detection":
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        img = image.resize((224,224))
        img = np.array(img).astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))*100

        st.image(image)
        st.success(predicted_class)
        st.metric("Confidence", f"{confidence:.2f}%")

elif page == "Disease Guide":
    st.write("Early Blight, Late Blight and Healthy leaf information.")

elif page == "About":
    st.write("AI Powered Agriculture Solution.")
