import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="SmartFarm PotatoGuard AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.stApp {
    background-color: #F7FAF7;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}

.hero-section {
    background: linear-gradient(
        135deg,
        #1B5E20,
        #2E7D32,
        #43A047
    );
    padding: 50px;
    border-radius: 25px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
}

.hero-title {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 1.1rem;
    opacity: 0.95;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    border: 1px solid #eef2f7;
}

.metric-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #e8f5e9;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
}

.metric-title {
    color: #6b7280;
    font-size: 14px;
}

.metric-value {
    color: #2E7D32;
    font-size: 28px;
    font-weight: bold;
}

.section-title {
    color: #1B5E20;
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 10px;
}

.feature-card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    border-left: 5px solid #43A047;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
}

.badge {
    background: #E8F5E9;
    color: #1B5E20;
    padding: 6px 14px;
    border-radius: 50px;
    font-size: 13px;
    font-weight: 600;
}

.footer-box {
    text-align: center;
    color: #6b7280;
    margin-top: 50px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODEL
# ==================================================

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

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.image(
    "https://img.icons8.com/color/96/potato.png",
    width=90
)

st.sidebar.title("🌾 SmartFarm")

st.sidebar.markdown(
    """
    <div class='badge'>
    PotatoGuard AI
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
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

# ==================================================
# HOME PAGE
# ==================================================

if page == "🏠 Home":

    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">
            🌾 SmartFarm PotatoGuard AI
        </div>

        <div class="hero-subtitle">
            Intelligent Potato Disease Detection System
            powered by Deep Learning and Computer Vision
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">
                Disease Classes
            </div>
            <div class="metric-value">
                3
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">
                AI Model
            </div>
            <div class="metric-value">
                CNN
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">
                Input Size
            </div>
            <div class="metric-value">
                224×224
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">
                Status
            </div>
            <div class="metric-value">
                Active
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    left, right = st.columns(2)

    with left:

        st.markdown("""
        <div class="feature-card">
        <h3>🚀 Why Use PotatoGuard AI?</h3>

        ✅ Early Disease Detection

        ✅ Faster Decision Making

        ✅ Reduced Crop Loss

        ✅ AI-Based Analysis

        ✅ Farmer Friendly Interface

        ✅ Real-Time Predictions

        </div>
        """, unsafe_allow_html=True)

    with right:

        st.markdown("""
        <div class="feature-card">
        <h3>🎯 Supported Categories</h3>

        • Early Blight

        • Late Blight

        • Healthy Leaves

        Detect diseases at an early stage and
        improve crop productivity.

        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown(
        "<h2 class='section-title'>🌱 Smart Agriculture Solution</h2>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    SmartFarm PotatoGuard AI uses advanced
    Deep Learning techniques to analyze
    potato leaf images and identify diseases
    with high confidence.

    This solution is designed for:

    • Farmers

    • Agriculture Companies

    • Researchers

    • NGOs

    • Government Agriculture Departments

    </div>
    """, unsafe_allow_html=True)
    # ==================================================
# DETECTION PAGE
# ==================================================

elif page == "🔍 Detection":

    st.markdown(
        """
        <h2 class='section-title'>
        🔍 AI Disease Detection Dashboard
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">
    Upload a clear image of a potato leaf and let
    SmartFarm PotatoGuard AI analyze the plant
    health using Deep Learning.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Upload Potato Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        col1, col2 = st.columns([1, 1])

        # ==========================================
        # IMAGE PREVIEW
        # ==========================================

        with col1:

            st.markdown("""
            <div class="card">
            <h3>📷 Uploaded Image</h3>
            </div>
            """, unsafe_allow_html=True)

            st.image(
                image,
                caption="Potato Leaf",
                use_container_width=True
            )

        # ==========================================
        # MODEL PREDICTION
        # ==========================================

        img = image.resize((224, 224))

        img = np.array(
            img
        ).astype("float32") / 255.0

        img = np.expand_dims(
            img,
            axis=0
        )

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

        # ==========================================
        # RESULT PANEL
        # ==========================================

        with col2:

            st.markdown("""
            <div class="card">
            <h3>🤖 AI Prediction Result</h3>
            </div>
            """, unsafe_allow_html=True)

            if predicted_class == "Healthy":

                st.success(
                    "✅ Healthy Plant Detected"
                )

                severity = "Very Low"

            elif predicted_class == "Early Blight":

                st.warning(
                    "⚠️ Early Blight Detected"
                )

                severity = "Moderate"

            else:

                st.error(
                    "🚨 Late Blight Detected"
                )

                severity = "High"

            st.metric(
                "Prediction",
                predicted_class
            )

            st.metric(
                "Confidence Score",
                f"{confidence:.2f}%"
            )

            st.progress(
                int(confidence)
            )
            st.progress(
    int(confidence)
)

# Confidence Quality Indicator

if confidence >= 85:
    st.success("High confidence prediction")

elif confidence >= 70:
    st.warning("Moderate confidence prediction")

else:
    st.error("Low confidence prediction")

st.write("")   # ✅ no extra spaces

            st.markdown("""
            <div class="card">
            <h4>📊 Disease Severity</h4>
            </div>
            """, unsafe_allow_html=True)

            if severity == "Very Low":

                st.success(
                    f"Severity Level: {severity}"
                )

            elif severity == "Moderate":

                st.warning(
                    f"Severity Level: {severity}"
                )

            else:

                st.error(
                    f"Severity Level: {severity}"
                )

        # ==========================================
        # RECOMMENDATIONS
        # ==========================================

        st.write("")

        st.markdown("""
        <h2 class='section-title'>
        🌱 Recommended Actions
        </h2>
        """, unsafe_allow_html=True)

        if predicted_class == "Healthy":

            st.markdown("""
            <div class="card">

            ### ✅ Plant Appears Healthy

            Recommended Practices:

            • Continue regular monitoring

            • Maintain proper irrigation

            • Apply balanced fertilization

            • Monitor for pest activity

            • Ensure proper field sanitation

            </div>
            """, unsafe_allow_html=True)

        elif predicted_class == "Early Blight":

            st.markdown("""
            <div class="card">

            ### ⚠️ Early Blight Management

            Recommended Actions:

            • Remove infected leaves

            • Improve air circulation

            • Avoid overhead irrigation

            • Apply recommended fungicides

            • Monitor nearby plants

            • Follow crop rotation practices

            </div>
            """, unsafe_allow_html=True)

        elif predicted_class == "Late Blight":

            st.markdown("""
            <div class="card">

            ### 🚨 Late Blight Management

            Immediate Actions:

            • Isolate affected plants

            • Apply fungicide immediately

            • Remove severely infected leaves

            • Inspect neighboring plants

            • Improve drainage conditions

            • Monitor crop daily

            </div>
            """, unsafe_allow_html=True)

        # ==========================================
        # AI ANALYSIS SUMMARY
        # ==========================================

        st.write("")

        st.markdown("""
        <h2 class='section-title'>
        📋 AI Analysis Summary
        </h2>
        """, unsafe_allow_html=True)

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:

            st.metric(
                "Detected Class",
                predicted_class
            )

        with summary_col2:

            st.metric(
                "Confidence",
                f"{confidence:.1f}%"
            )

        with summary_col3:

            st.metric(
                "Risk Level",
                severity
            )

        if confidence < 70:

            st.warning(
                """
                Low confidence prediction detected.
                Please upload a clearer image
                with proper lighting conditions.
                """
            )

        else:

            st.success(
                """
                Prediction confidence is high.
                Results are considered reliable.
                """
            )
            # ==================================================
# DISEASE GUIDE PAGE
# ==================================================

elif page == "📚 Disease Guide":

    st.markdown(
        """
        <h2 class='section-title'>
        📚 Potato Disease Guide
        </h2>
        """,
        unsafe_allow_html=True
    )

    # ==========================================
    # EARLY BLIGHT
    # ==========================================

    st.markdown("""
    <div class="card">
    <h2>🟠 Early Blight</h2>

    <h4>Symptoms</h4>

    • Brown circular spots on leaves

    • Concentric ring pattern

    • Yellowing around lesions

    • Premature leaf drop

    <h4>Causes</h4>

    • Alternaria fungus

    • Warm and humid weather

    • Poor crop management

    <h4>Prevention</h4>

    • Crop rotation

    • Proper spacing

    • Field sanitation

    • Balanced fertilization

    <h4>Treatment</h4>

    • Remove infected leaves

    • Use recommended fungicides

    • Improve air circulation

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ==========================================
    # LATE BLIGHT
    # ==========================================

    st.markdown("""
    <div class="card">
    <h2>🔴 Late Blight</h2>

    <h4>Symptoms</h4>

    • Water-soaked lesions

    • Rapid disease spread

    • Dark brown leaf patches

    • White fungal growth

    <h4>Causes</h4>

    • Phytophthora infestans

    • Cool and humid weather

    • Excessive moisture

    <h4>Prevention</h4>

    • Disease-free seed selection

    • Proper drainage

    • Regular monitoring

    • Preventive spraying

    <h4>Treatment</h4>

    • Immediate fungicide application

    • Remove infected plants

    • Isolate affected areas

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ==========================================
    # HEALTHY PLANT
    # ==========================================

    st.markdown("""
    <div class="card">
    <h2>🟢 Healthy Plant</h2>

    <h4>Healthy Indicators</h4>

    • Uniform green color

    • No visible lesions

    • Strong leaf structure

    • Healthy growth pattern

    <h4>Best Practices</h4>

    • Balanced nutrition

    • Proper irrigation

    • Routine field inspection

    • Integrated pest management

    <h4>Maintenance</h4>

    • Monitor crop health regularly

    • Maintain soil fertility

    • Remove weeds promptly

    </div>
    """, unsafe_allow_html=True)

# ==================================================
# ABOUT PAGE
# ==================================================

elif page == "ℹ️ About":

    st.markdown(
        """
        <h2 class='section-title'>
        ℹ️ About SmartFarm PotatoGuard AI
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    <h3>🌾 Project Overview</h3>

    SmartFarm PotatoGuard AI is an intelligent
    agriculture solution designed to detect
    potato leaf diseases using Artificial
    Intelligence, Deep Learning and Computer Vision.

    The platform assists farmers, researchers,
    agricultural organizations and extension
    services in identifying diseases early and
    reducing crop losses.

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="card">

    <h3>🤖 AI Technology</h3>

    The system uses a Convolutional Neural Network (CNN)
    trained on potato leaf images.

    Workflow:

    1. Upload potato leaf image

    2. Image preprocessing

    3. Deep learning analysis

    4. Disease classification

    5. Confidence score generation

    6. Recommendation output

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="card">

        <h3>🚀 Benefits for Farmers</h3>

        • Early disease detection

        • Reduced crop losses

        • Faster decision making

        • Better disease management

        • Increased productivity

        • Cost-effective monitoring

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="card">

        <h3>📈 Future Roadmap</h3>

        • Multi-crop disease detection

        • Mobile application

        • Cloud-based analytics

        • IoT integration

        • Real-time monitoring

        • Advanced reporting

        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="card">

    <h3>🏆 System Features</h3>

    ✅ Deep Learning Based Detection

    ✅ Smart Disease Identification

    ✅ Confidence Score Reporting

    ✅ User Friendly Dashboard

    ✅ Streamlit Cloud Deployment

    ✅ Agriculture-Focused Design

    </div>
    """, unsafe_allow_html=True)

# ==================================================
# FOOTER
# ==================================================

st.write("")
st.write("")

st.markdown("""
<div class="footer-box">

<hr>

<h4>🌾 SmartFarm PotatoGuard AI</h4>

Intelligent Potato Disease Detection System

Powered by Deep Learning & Computer Vision

© 2026 Smart Agriculture Initiative

</div>
""", unsafe_allow_html=True)
