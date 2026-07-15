import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Potato Disease Detection",
    page_icon="🥔",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.hero {
    background: linear-gradient(135deg, #1b5e20, #4caf50);
    padding: 2.5rem;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.feature-card {
    background: #f8fff8;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

.result-card {
    background: #f1fff1;
    padding: 20px;
    border-radius: 15px;
    border-left: 8px solid #2e7d32;
    margin-top: 15px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "potato_disease_model.keras",
        compile=False
    )

model = load_model()

class_names = ["Early Blight", "Late Blight", "Healthy"]

# ---------------- HERO SECTION ----------------
st.markdown("""
<div class="hero">
    <h1>🥔 AI-Powered Potato Disease Detection</h1>
    <h4>Upload a potato leaf image and instantly detect diseases using Deep Learning</h4>
</div>
""", unsafe_allow_html=True)

# ---------------- FEATURES ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>⚡ Fast Detection</h3>
        <p>Instant disease prediction in seconds.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>🤖 AI Powered</h3>
        <p>Deep learning model trained for disease recognition.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>🌱 Farmer Friendly</h3>
        <p>Simple and easy interface for everyone.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# ---------------- UPLOAD SECTION ----------------
st.subheader("📤 Upload Potato Leaf Image")

uploaded_file = st.file_uploader(
    "Choose a potato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
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

    with col2:

        st.markdown("""
        <div class="result-card">
        <h2>🔍 Prediction Result</h2>
        </div>
        """, unsafe_allow_html=True)

        st.success(f"Prediction: {predicted_class}")

        st.progress(int(confidence))

        st.info(f"Confidence: {confidence:.2f}%")

        if confidence < 70:
            st.warning(
                "Low confidence prediction. Please upload a clearer image."
            )

# ---------------- DISEASE INFO ----------------
st.write("")
st.write("")

st.subheader("📚 Disease Information")

tab1, tab2, tab3 = st.tabs(
    ["Early Blight", "Late Blight", "Healthy"]
)

with tab1:
    st.write("""
    **Early Blight**
    
    Symptoms:
    - Brown circular spots
    - Yellowing leaves
    - Reduced crop yield
    """)

with tab2:
    st.write("""
    **Late Blight**
    
    Symptoms:
    - Dark water-soaked lesions
    - Rapid spread
    - Severe crop damage
    """)

with tab3:
    st.write("""
    **Healthy Leaf**
    
    Characteristics:
    - Uniform green color
    - No visible disease spots
    - Healthy plant growth
    """)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    <hr>
    <p>🥔 Potato Disease Detection System</p>
    <p>Built with Streamlit, TensorFlow & Deep Learning</p>
</div>
""", unsafe_allow_html=True)
