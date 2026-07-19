import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="AI Potato Disease Detection",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
.block-container{padding-top:1rem;padding-bottom:2rem;}
[data-testid="stSidebar"]{background:#f7fff7;}
.hero{background:linear-gradient(135deg,#2E7D32,#43A047);color:white;padding:40px;border-radius:20px;text-align:center;margin-bottom:25px;}
.metric-card{background:white;padding:20px;border-radius:18px;text-align:center;}
.section-title{color:#2E7D32;font-weight:bold;}
.prediction-box{background:#f8fff8;border-left:8px solid #2E7D32;padding:25px;border-radius:15px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("potato_disease_model.keras", compile=False)

model = load_model()
class_names = ["Early Blight", "Late Blight", "Healthy"]

st.sidebar.title("🥔 Navigation")
page = st.sidebar.radio("", ["🏠 Home", "🔍 Detection"])

if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
        <h1>🥔 AI-Powered Potato Disease Detection</h1>
        <p>Detect potato leaf diseases instantly using Deep Learning and Computer Vision technology.</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "🔍 Detection":
    st.markdown("<h2 class='section-title'>Upload Potato Leaf Image</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Leaf")

        img = image.resize((224,224))
        img = np.array(img).astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")
