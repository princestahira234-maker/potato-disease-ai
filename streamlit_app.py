import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------------------------------

# PAGE CONFIG

# --------------------------------------------------

st.set_page_config(
page_title="AI Potato Disease Detection",
page_icon="🥔",
layout="wide",
initial_sidebar_state="expanded"
)

# --------------------------------------------------

# CUSTOM CSS

# --------------------------------------------------

st.markdown("""

<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}

.block-container{
    padding-top:1rem;
    padding-bottom:2rem;
}

[data-testid="stSidebar"]{
    background:#f7fff7;
}

.hero{
    background: linear-gradient(135deg,#2E7D32,#43A047);
    color:white;
    padding:40px;
    border-radius:20px;
    text-align:center;
    margin-bottom:25px;
}

.hero h1{
    font-size:3rem;
    margin-bottom:10px;
}

.hero p{
    font-size:1.1rem;
}

.metric-card{
    background:white;
    padding:20px;
    border-radius:18px;
    text-align:center;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);
    border:1px solid #E8F5E9;
}

.section-title{
    color:#2E7D32;
    font-weight:bold;
}

.prediction-box{
    background:#f8fff8;
    border-left:8px solid #2E7D32;
    padding:25px;
    border-radius:15px;
}

.info-card{
    background:white;
    padding:20px;
    border-radius:15px;
    border:1px solid #e5e7eb;
    box-shadow:0 2px 8px rgba(0,0,0,0.05);
}

.big-text{
    font-size:24px;
    font-weight:bold;
    color:#2E7D32;
}

.footer-box{
    text-align:center;
    color:#6b7280;
    margin-top:50px;
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "potato_disease_model.keras",
        compile=False
    )

model = load_model()

class_names = [
"Early Blight",
"Late Blight",
"Healthy"
]

# --------------------------------------------------

# SIDEBAR

# --------------------------------------------------

st.sidebar.image(
"https://img.icons8.com/color/96/potato.png",
width=80
)

st.sidebar.title("🥔 Navigation")

page = st.sidebar.radio(
"",
[
"🏠 Home",
"🔍 Detection",
"📚 Disease Guide",
"ℹ️ About"
]
)

st.sidebar.markdown("---")

st.sidebar.success(
"AI Agriculture Assistant"
)

# --------------------------------------------------

# HOME PAGE

# --------------------------------------------------

if page == "🏠 Home":

```
st.markdown("""
<div class="hero">
    <h1>🥔 AI-Powered Potato Disease Detection</h1>
    <p>
    Detect potato leaf diseases instantly using Deep Learning
    and Computer Vision technology.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>🌿 Classes</h3>
        <h2>3</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>🤖 Model</h3>
        <h2>CNN</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>📷 Input Size</h3>
        <h2>224×224</h2>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

left, right = st.columns(2)

with left:
    st.markdown("""
    <div class="info-card">
    <h3>🚀 Why Use This System?</h3>

    ✔ Instant disease detection

    ✔ AI-powered analysis

    ✔ Farmer-friendly interface

    ✔ Fast and reliable predictions

    ✔ Supports healthy and diseased leaves

    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="info-card">
    <h3>🎯 Supported Diseases</h3>

    • Early Blight

    • Late Blight

    • Healthy Leaves

    This system helps farmers identify
    disease symptoms at an early stage.
    </div>
    """, unsafe_allow_html=True)
```

# --------------------------------------------------

# DETECTION PAGE

# --------------------------------------------------

elif page == "🔍 Detection":

```
st.markdown(
    "<h2 class='section-title'>Upload Potato Leaf Image</h2>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:
        st.image(
            image,
            caption="Uploaded Leaf",
            use_container_width=True
        )

    img = image.resize((224,224))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(
        img,
        verbose=0
    )

    predicted_class = class_names[
        np.argmax(prediction)
    ]

    confidence = float(
        np.max(prediction)
    ) * 100

    with col2:

        st.markdown("""
        <div class="prediction-box">
        <h2>🔍 Prediction Result</h2>
        </div>
        """, unsafe_allow_html=True)

        st.success(
            f"Prediction: {predicted_class}"
        )

        st.progress(
            int(confidence)
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        if predicted_class == "Healthy":

            st.success(
                "Leaf appears healthy."
            )

        elif predicted_class == "Early Blight":

            st.warning(
                "Early Blight detected."
            )

            st.markdown("""
            ### Recommended Actions

            • Remove infected leaves

            • Improve air circulation

            • Apply recommended fungicides

            • Monitor nearby plants
            """)

        elif predicted_class == "Late Blight":

            st.error(
                "Late Blight detected."
            )

            st.markdown("""
            ### Immediate Actions

            • Isolate affected plants

            • Apply fungicide quickly

            • Remove severely infected leaves

            • Monitor entire crop
            """)

        if confidence < 70:

            st.warning(
                "Low confidence prediction. Upload a clearer image."
            )
```

# Disease Guide and About page Part 2 mein add kiye jayenge.
