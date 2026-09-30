# 🌾 SmartFarm PotatoGuard AI

## Intelligent Potato Disease Detection System using Deep Learning & Computer Vision

SmartFarm PotatoGuard AI is an **AI-powered agricultural computer vision application** designed to identify potato leaf diseases from images using **Deep Learning and Transfer Learning**.

The system analyzes a potato leaf image and classifies it into one of three categories:

* 🟠 **Early Blight**
* 🔴 **Late Blight**
* 🟢 **Healthy**

The trained model is integrated into an interactive **Streamlit application**, providing an accessible interface for image-based disease analysis and agricultural guidance.

---

## 🚀 Live Demo

**Try the deployed application:**

👉 https://potato-disease-ai-myfiddjqngfpmmyegeddit.streamlit.app/

---

## 📌 Project Overview

Potato diseases can significantly affect crop health and productivity when they are not identified early.

SmartFarm PotatoGuard AI demonstrates how **Deep Learning and Computer Vision** can be applied to agricultural image analysis to provide rapid disease classification from potato leaf images.

### The system provides:

* 🌱 AI-based potato disease classification
* 📷 Image-based computer vision analysis
* 📊 Prediction confidence
* 🔎 Disease information
* 💡 Agricultural management guidance
* 🖥️ Interactive Streamlit dashboard
* 🚀 Deployed web application

The project combines **machine learning development, model integration, application development, and deployment** into a complete end-to-end AI solution.

---

# 🎯 Business & Real-World Objective

The goal of SmartFarm PotatoGuard AI is to demonstrate a practical AI workflow that can assist with **early-stage crop disease identification**.

Potential users and use cases include:

* 🌱 Smart farming platforms
* 🚜 Agricultural technology solutions
* 🔬 Agriculture research projects
* 🌾 Crop monitoring systems
* 👨‍🌾 Farmer assistance applications
* 🎓 Agricultural education and AI demonstrations

> **Note:** This application is an AI demonstration and should not replace professional agricultural diagnosis or expert treatment recommendations.

---

# 🚀 Key Features

## 🌱 AI Disease Detection

Users can upload a potato leaf image and receive an AI-generated classification.

The system identifies:

```text
Early Blight
Late Blight
Healthy
```

---

## 📊 Prediction Dashboard

The Streamlit application provides an interactive interface for:

* Uploading a potato leaf image
* Previewing the uploaded image
* Running the trained AI model
* Displaying the predicted class
* Showing prediction confidence
* Presenting relevant disease information
* Providing agricultural guidance

---

## 📚 Disease Knowledge Guide

The application includes information about the major classes supported by the model.

### 🟠 Early Blight

Common indicators include:

* Brown circular or irregular leaf spots
* Yellowing around affected areas
* Progressive leaf damage

General management guidance may include:

* Removing severely affected plant material
* Improving air circulation
* Following appropriate crop-management practices

---

### 🔴 Late Blight

Common indicators include:

* Dark or water-soaked leaf lesions
* Rapid progression under favorable conditions
* Extensive leaf damage

General management guidance may include:

* Removing affected plant material
* Improving field drainage
* Following appropriate disease-management practices

---

### 🟢 Healthy Plant

The healthy class represents leaves without the visual disease patterns targeted by the model.

Typical indicators include:

* Green leaves
* No obvious disease lesions
* Normal-looking leaf structure

---

# 🧠 Deep Learning Model

## MobileNetV2 Transfer Learning

The project uses **MobileNetV2** as the backbone for image feature extraction.

Instead of training a large image-classification network entirely from scratch, the project applies a **transfer learning approach**, adapting a pretrained lightweight architecture to the potato disease classification task.

### Why MobileNetV2?

MobileNetV2 is suitable for deployment-oriented computer vision applications because of its relatively lightweight architecture and efficient feature extraction.

This makes it a practical choice for integrating an image classification model into an interactive web application.

---

## Model Configuration

```text
Architecture: MobileNetV2
Model Type: Keras Functional Model
Input Size: 224 × 224 × 3
Output Classes: 3
Framework: TensorFlow / Keras
```

### Classification Classes

```text
1. Early Blight
2. Late Blight
3. Healthy
```

The majority of the MobileNetV2 backbone is frozen while the classification component is adapted for the potato disease classification task.

---

# 📐 Model Architecture

The model follows a transfer-learning pipeline:

```text
Input Potato Leaf Image
          │
          ▼
Image Resizing
224 × 224
          │
          ▼
Image Preprocessing
          │
          ▼
MobileNetV2
Feature Extraction
          │
          ▼
Global Average Pooling
          │
          ▼
Dense Classification Layer
          │
          ▼
Disease Classification
          │
          ▼
Prediction Confidence
```

---

# 🔢 Model Information

```text
Model Type:
Keras Functional Model

Total Parameters:
2,261,827

Trainable Parameters:
3,843

Non-Trainable Parameters:
2,257,984

Model Size:
~8.63 MB
```

The model configuration reflects a lightweight transfer-learning setup in which most backbone parameters remain frozen while the task-specific classification layer is trained.

---

# 🛠️ Technology Stack

| Category            | Technologies                |
| ------------------- | --------------------------- |
| Programming         | Python 3.12                 |
| Deep Learning       | TensorFlow 2.20, Keras 3.13 |
| Computer Vision     | Pillow, Scikit-image        |
| Machine Learning    | Scikit-learn                |
| Numerical Computing | NumPy                       |
| Visualization       | Matplotlib, Seaborn         |
| Web Application     | Streamlit                   |
| Model Format        | Keras `.keras`              |
| Deployment          | Streamlit Cloud             |

---

# 🔄 End-to-End AI Workflow

The project follows an end-to-end machine learning workflow:

```text
Dataset
   │
   ▼
Image Preparation
   │
   ▼
Image Preprocessing
   │
   ▼
Transfer Learning
   │
   ▼
MobileNetV2 Feature Extraction
   │
   ▼
Task-Specific Classification
   │
   ▼
Model Saving
   │
   ▼
Streamlit Integration
   │
   ▼
User Image Upload
   │
   ▼
AI Prediction
   │
   ▼
Result & Agricultural Guidance
```

This workflow demonstrates the complete transition from **computer vision model development to a deployable AI application**.

---

# 💻 Streamlit Application

The trained model is integrated into a Streamlit web application.

### Application Flow

```text
Home Dashboard
      │
      ▼
Disease Detection
      │
      ▼
Upload Leaf Image
      │
      ▼
AI Prediction
      │
      ▼
Prediction Result
      │
      ▼
Disease Information
      │
      ▼
Agricultural Guidance
```

The interface is designed to make the AI model accessible without requiring users to interact directly with Python code or machine learning infrastructure.

---

# 📂 Project Structure

```text
SmartFarm-PotatoGuard-AI/
│
├── streamlit_app.py
├── potato_disease_model.keras
├── requirements.txt
├── runtime.txt
└── README.md
```

---

# ⚙️ Installation & Local Setup

## 1. Clone the Repository

```bash
git clone your_repository_link
cd SmartFarm-PotatoGuard-AI
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the Application

```bash
streamlit run streamlit_app.py
```

The application will then be available through the local Streamlit server.

---

# 📋 Requirements

The project environment includes:

```text
tensorflow==2.20.0
keras==3.13.2
numpy==2.0.2
pillow==11.3.0
scikit-learn==1.6.1
scikit-image==0.25.2
matplotlib==3.10.0
seaborn==0.13.2
streamlit
```

---

# 🌍 Potential Applications

The underlying approach can be extended to broader agricultural AI solutions such as:

### Smart Agriculture

AI-assisted crop monitoring and disease identification.

### Agricultural Research

Computer vision experiments for plant disease analysis.

### Crop Monitoring

Image-based monitoring systems for agricultural environments.

### Farmer Assistance

Accessible interfaces that provide preliminary AI-based crop insights.

### AI Education

Demonstrating the practical implementation of Deep Learning in agriculture.

---

# 🔮 Future Development

Possible future extensions include:

* 📱 Mobile application integration
* 📷 Real-time camera-based detection
* 🌾 Multi-crop disease classification
* 🌦️ Weather-aware disease risk analysis
* ☁️ Scalable cloud deployment
* 📡 IoT-based crop monitoring
* 🧠 More advanced disease classification models
* 📊 Model monitoring and feedback systems

---

# 🏆 Project Highlights

* 🌾 **Agricultural AI / Computer Vision**
* 🧠 **MobileNetV2 Transfer Learning**
* 🔬 **Deep Learning Image Classification**
* 🖼️ **Image-Based Disease Detection**
* 💻 **Interactive Streamlit Application**
* 🚀 **Live AI Deployment**
* 🔄 **End-to-End ML Workflow**
* 🌱 **Real-World Agriculture Use Case**

---

# 👩‍💻 Project

**SmartFarm PotatoGuard AI**

An end-to-end demonstration of applying **Artificial Intelligence, Deep Learning, Computer Vision, and Web Deployment** to agricultural disease detection.

---

# 📜 License

This project is developed for **educational, research, and demonstration purposes**.

---

# 🙏 Acknowledgement

This project builds upon the open-source **Python, TensorFlow, Keras, Scikit-learn, and Streamlit** ecosystem that enables the development and deployment of practical AI applications.
