import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="SmartFarm PotatoGuard AI",
    page_icon="🌾",
    layout="wide"
)



# ==============================
# CSS DESIGN
# ==============================

st.markdown("""
<style>

.stApp{
background:#F7FAF7;
}


.card{
background:white;
padding:25px;
border-radius:20px;
box-shadow:0 4px 15px rgba(0,0,0,0.1);
}


.hero{
background:linear-gradient(
135deg,
#1B5E20,
#43A047
);
padding:50px;
border-radius:25px;
color:white;
text-align:center;
}


.title{
font-size:40px;
font-weight:bold;
}


.section-title{
color:#1B5E20;
font-size:30px;
font-weight:bold;
}


.metric-card{

background:white;
padding:20px;
border-radius:15px;
text-align:center;

}


</style>
""", unsafe_allow_html=True)



# ==============================
# LOAD MODEL
# ==============================


@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "potato_disease_model.keras",
        compile=False
    )

    return model



model = load_model()



class_names = [

"Early Blight",
"Late Blight",
"Healthy"

]



# ==============================
# SIDEBAR
# ==============================


st.sidebar.title("🌾 SmartFarm")

st.sidebar.write(
"PotatoGuard AI"
)


page = st.sidebar.radio(

"Navigation",

[
"🏠 Home",
"🔍 Detection",
"📚 Disease Guide",
"ℹ️ About"
]

)



# ==============================
# HOME PAGE
# ==============================


if page == "🏠 Home":



    st.markdown("""
    <div class="hero">

    <div class="title">
    🌾 SmartFarm PotatoGuard AI
    </div>


    <p>
    Intelligent Potato Disease Detection
    using Deep Learning
    </p>


    </div>

    """,
    unsafe_allow_html=True
    )



    st.write("")



    c1,c2,c3,c4 = st.columns(4)



    with c1:

        st.metric(
        "Disease Classes",
        "3"
        )


    with c2:

        st.metric(
        "AI Model",
        "CNN"
        )


    with c3:

        st.metric(
        "Image Size",
        "224x224"
        )


    with c4:

        st.metric(
        "Status",
        "Active"
        )



    st.write("")



    st.markdown(
    """
    <div class="card">

    ## 🚀 Features


    ✅ Early Disease Detection


    ✅ AI Based Prediction


    ✅ Confidence Score


    ✅ Disease Management Guide


    ✅ Farmer Friendly System


    </div>

    """,
    unsafe_allow_html=True
    )
# ==============================
# DETECTION PAGE
# ==============================


elif page == "🔍 Detection":


    st.markdown(
    """
    <h2 class="section-title">
    🔍 AI Disease Detection
    </h2>
    """,
    unsafe_allow_html=True
    )


    st.markdown(
    """
    <div class="card">

    Upload potato leaf image and AI model
    will detect disease automatically.

    </div>
    """,
    unsafe_allow_html=True
    )



    uploaded_file = st.file_uploader(

        "📤 Upload Potato Leaf Image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ]

    )



    if uploaded_file is not None:



        image = Image.open(
            uploaded_file
        ).convert("RGB")



        col1,col2 = st.columns(2)



        # ======================
        # IMAGE DISPLAY
        # ======================


        with col1:


            st.markdown(
            """
            <div class="card">

            📷 Uploaded Image

            </div>
            """,
            unsafe_allow_html=True
            )


            st.image(
                image,
                use_container_width=True
            )




        # ======================
        # MODEL PREDICTION
        # ======================


        img = image.resize(
            (224,224)
        )


        img = np.array(
            img
        )


        img = img.astype(
            "float32"
        ) / 255.0



        img = np.expand_dims(
            img,
            axis=0
        )



        prediction = model.predict(
            img,
            verbose=0
        )



        index = np.argmax(
            prediction
        )


        predicted_class = class_names[index]


        confidence = float(
            np.max(prediction)
        ) * 100





        # ======================
        # RESULT DISPLAY
        # ======================


        with col2:



            st.markdown(
            """
            <div class="card">

            🤖 AI Prediction Result

            </div>
            """,
            unsafe_allow_html=True
            )



            if predicted_class == "Healthy":


                st.success(
                "✅ Healthy Plant Detected"
                )


                severity = "Low"



            elif predicted_class == "Early Blight":


                st.warning(
                "⚠️ Early Blight Detected"
                )


                severity = "Medium"



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

                "Confidence",

                f"{confidence:.2f}%"

            )



            st.progress(
                int(confidence)
            )




            if confidence >= 85:


                st.success(
                "High Confidence Prediction"
                )



            elif confidence >=70:


                st.warning(
                "Moderate Confidence Prediction"
                )



            else:


                st.error(
                "Low Confidence - Upload Clear Image"
                )





            st.write("")



            st.subheader(
            "📊 Disease Severity"
            )



            if severity=="Low":

                st.success(
                severity
                )


            elif severity=="Medium":

                st.warning(
                severity
                )


            else:

                st.error(
                severity
                )




        # ======================
        # RECOMMENDATIONS
        # ======================


        st.write("")

        st.markdown(
        """
        <h2 class="section-title">
        🌱 Recommended Actions
        </h2>
        """,
        unsafe_allow_html=True
        )




        if predicted_class=="Healthy":


            st.info(
            """
            ✅ Plant is healthy.

            • Continue monitoring

            • Maintain irrigation

            • Keep field clean

            """
            )



        elif predicted_class=="Early Blight":


            st.warning(
            """
            ⚠️ Early Blight Management:

            • Remove infected leaves

            • Improve air circulation

            • Apply suitable fungicide

            • Monitor nearby plants

            """
            )



        else:


            st.error(
            """
            🚨 Late Blight Management:

            • Remove infected plants

            • Apply fungicide quickly

            • Check surrounding crops

            """
            )




        # ======================
        # SUMMARY
        # ======================


        st.write("")


        st.markdown(
        """
        <h2 class="section-title">
        📋 AI Analysis Summary
        </h2>
        """,
        unsafe_allow_html=True
        )



        s1,s2,s3 = st.columns(3)



        with s1:

            st.metric(
            "Disease",
            predicted_class
            )



        with s2:

            st.metric(
            "Accuracy",
            f"{confidence:.1f}%"
            )



        with s3:

            st.metric(
            "Status",
            "Completed"
            )
            # ==============================
# DISEASE GUIDE PAGE
# ==============================


elif page == "📚 Disease Guide":


    st.markdown(
    """
    <h2 class="section-title">
    📚 Potato Disease Guide
    </h2>
    """,
    unsafe_allow_html=True
    )



    st.markdown(
    """
    <div class="card">

    ## 🟠 Early Blight


    ### Symptoms

    • Brown circular spots on leaves

    • Yellowing around infected area

    • Leaf dropping



    ### Causes

    • Alternaria fungus

    • Warm humid conditions

    • Poor field management



    ### Prevention

    • Crop rotation

    • Remove infected leaves

    • Maintain plant spacing



    ### Treatment

    • Apply recommended fungicide

    • Improve air circulation


    </div>

    """,
    unsafe_allow_html=True
    )



    st.write("")



    st.markdown(
    """
    <div class="card">


    ## 🔴 Late Blight


    ### Symptoms

    • Dark brown patches

    • Water soaked lesions

    • Fast disease spreading



    ### Causes

    • Phytophthora infestans

    • High humidity

    • Excess moisture



    ### Prevention

    • Healthy seed selection

    • Proper drainage

    • Regular monitoring



    ### Treatment

    • Immediate fungicide application

    • Remove infected plants


    </div>

    """,
    unsafe_allow_html=True
    )



    st.write("")



    st.markdown(
    """
    <div class="card">


    ## 🟢 Healthy Potato Plant


    ### Indicators


    • Green leaves

    • No spots

    • Strong growth


    ### Best Practices


    • Balanced fertilizer

    • Proper irrigation

    • Regular inspection


    </div>

    """,
    unsafe_allow_html=True
    )





# ==============================
# ABOUT PAGE
# ==============================


elif page == "ℹ️ About":


    st.markdown(
    """
    <h2 class="section-title">
    ℹ️ About SmartFarm PotatoGuard AI
    </h2>

    """,
    unsafe_allow_html=True
    )



    st.markdown(
    """
    <div class="card">


    ## 🌾 Project Overview


    SmartFarm PotatoGuard AI is an AI based
    agriculture system designed to detect
    potato leaf diseases using Deep Learning
    and Computer Vision.



    ## 🤖 Technology


    The system uses:

    • Convolutional Neural Network (CNN)

    • Image Processing

    • Deep Learning Classification

    • Confidence Score Analysis



    ## 🚀 Benefits


    ✅ Early disease detection

    ✅ Reduced crop loss

    ✅ Faster decisions

    ✅ Farmer friendly interface



    </div>

    """,
    unsafe_allow_html=True
    )





# ==============================
# FOOTER
# ==============================


st.write("")

st.markdown(
"""
<div style="
text-align:center;
color:#6b7280;
margin-top:50px;
">


<hr>


<h4>
🌾 SmartFarm PotatoGuard AI
</h4>


Intelligent Potato Disease Detection System


<br>


Powered by Deep Learning & Computer Vision


<br><br>


© 2026 Smart Agriculture Initiative


</div>

""",
unsafe_allow_html=True
)
              
            
