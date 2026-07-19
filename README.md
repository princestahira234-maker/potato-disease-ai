# 🌾 SmartFarm PotatoGuard AI  
## Intelligent Potato Disease Detection System using Deep Learning & Computer Vision

![SmartFarm PotatoGuard AI](https://img.shields.io/badge/AI-Agriculture-green)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![Keras](https://img.shields.io/badge/Keras-3.13-red)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-brightgreen)

---

# 📌 Project Overview

**SmartFarm PotatoGuard AI** is an Artificial Intelligence based agriculture solution designed to automatically detect potato leaf diseases using **Deep Learning and Computer Vision**.

The system analyzes uploaded potato leaf images and classifies them into three categories:

- 🟠 Early Blight
- 🔴 Late Blight
- 🟢 Healthy Plant

The goal of this project is to support farmers, agriculture researchers, and crop management teams by providing fast and intelligent disease identification.

This application provides:

✅ AI-based disease prediction  
✅ Confidence score calculation  
✅ Disease severity estimation  
✅ Recommended agricultural actions  
✅ User-friendly Streamlit dashboard  

---

# 🎯 Project Objectives

The main objectives of SmartFarm PotatoGuard AI are:

- Detect potato diseases at an early stage
- Reduce crop losses through faster identification
- Provide AI assistance to farmers
- Demonstrate the application of Deep Learning in agriculture
- Create an easy-to-use disease monitoring platform

---

# 🚀 Application Features

## 🌱 Smart Disease Detection

Users can upload a potato leaf image and the AI model predicts:

- Disease category
- Prediction confidence
- Disease severity level
- Recommended actions


## 📊 AI Prediction Dashboard

The Streamlit interface provides:

- Image preview
- AI prediction result
- Confidence percentage
- Disease analysis
- Farmer recommendations


## 📚 Disease Knowledge Guide

The application includes information about:

### Early Blight

Symptoms:

- Brown circular spots
- Yellowing leaves
- Leaf damage

Management:

- Remove infected leaves
- Improve air circulation
- Apply recommended fungicides


### Late Blight

Symptoms:

- Dark leaf patches
- Rapid disease spreading
- Moisture-related damage

Management:

- Immediate treatment
- Remove infected plants
- Improve field drainage


### Healthy Plant

Indicators:

- Green leaves
- No visible infections
- Strong plant growth

---

# 🧠 Machine Learning Model

## Model Architecture

The project uses a **Convolutional Neural Network (CNN)** based transfer learning approach.

The backbone architecture is:

## MobileNetV2

MobileNetV2 is selected because it provides:

- High accuracy
- Lightweight architecture
- Faster prediction
- Better deployment performance


## Model Type

```
Keras Functional Model
```

## Input Image Size

```
224 × 224 × 3
```

## Output Classes

```
3 Classes

1. Early Blight
2. Late Blight
3. Healthy
```

---

# 📐 Model Summary

Model Information:

```
Model Type:
keras.src.models.functional.Functional


Total Parameters:
2,261,827


Trainable Parameters:
3,843


Non-Trainable Parameters:
2,257,984


Model Size:
8.63 MB
```

The majority of MobileNetV2 layers are frozen, while the final classification layer is trained for potato disease detection.

---

# 🔬 Deep Learning Workflow

The complete workflow:

```
Input Image
      |
      ↓
Image Resize (224x224)
      |
      ↓
Image Normalization
      |
      ↓
MobileNetV2 Feature Extraction
      |
      ↓
Global Average Pooling
      |
      ↓
Dense Classification Layer
      |
      ↓
Disease Prediction
      |
      ↓
Confidence Score
```

---

# 🛠️ Technologies Used

## Programming Language

```
Python 3.12.13
```

## Deep Learning Frameworks

```
TensorFlow 2.20.0
Keras 3.13.2
```

## Data Processing Libraries

```
NumPy 2.0.2
Pillow 11.3.0
Scikit-image 0.25.2
```

## Machine Learning Libraries

```
Scikit-learn 1.6.1
```

## Visualization Libraries

```
Matplotlib 3.10.0
Seaborn 0.13.2
```

## Deployment Framework

```
Streamlit
```

---

# 📦 Installed Environment

Python Environment:

```
Python Version:
3.12.13
```

TensorFlow:

```
2.20.0
```

Keras:

```
3.13.2
```

NumPy:

```
2.0.2
```

Matplotlib:

```
3.10.0
```

Scikit-Learn:

```
1.6.1
```

Pillow:

```
11.3.0
```

---

# 📊 Model Visualization

The model architecture was analyzed using:

- TensorFlow Model Summary
- Training performance graphs
- Prediction visualization
- Confusion matrix analysis


Recommended visualizations:

## 1. Training Accuracy Graph

Shows:

- Learning progress
- Training improvement
- Model convergence


## 2. Training Loss Graph

Shows:

- Error reduction
- Overfitting monitoring


## 3. Confusion Matrix

Shows:

- Correct predictions
- Misclassification patterns


## 4. Class Distribution Chart

Shows:

- Number of images per disease category

---

# 📈 Performance Evaluation

The model can be evaluated using:

## Accuracy

Measures overall correct predictions.

## Precision

Measures correct positive predictions.

## Recall

Measures disease detection ability.

## F1 Score

Balances precision and recall.

## Confusion Matrix

Provides detailed class-wise performance.

---

# 💻 Streamlit Application

The AI model is deployed using Streamlit.

Application features:

```
Home Dashboard
        |
        |
Disease Detection
        |
        |
Disease Guide
        |
        |
About Section
```

---

# 📂 Project Structure

```
SmartFarm-PotatoGuard-AI

│
├── streamlit_app.py
│
├── potato_disease_model.keras
│
├── requirements.txt
│
├── README.md
│
└── assets
    |
    └── images
```

---

# ⚙️ Installation Guide

Clone the repository:

```bash
git clone your_repository_link
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit application:

```bash
streamlit run streamlit_app.py
```

---

# 📋 Requirements

Example requirements.txt:

```
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

# 🌍 Real World Applications

SmartFarm PotatoGuard AI can be used in:

🌱 Smart Farming Systems  
🌱 Agriculture Research Centers  
🌱 Crop Monitoring Platforms  
🌱 Farmer Assistance Applications  
🌱 Agricultural Education Programs  

---

# 🔮 Future Improvements

Future development possibilities:

- Mobile application integration
- Real-time camera detection
- Multi-crop disease classification
- Weather-based disease prediction
- Cloud AI deployment
- IoT-based smart farming integration

---

# 🏆 Project Highlights

⭐ Deep Learning Based Agriculture Solution  
⭐ MobileNetV2 Transfer Learning Model  
⭐ Streamlit Interactive Dashboard  
⭐ Automated Disease Classification  
⭐ Farmer-Friendly AI Assistant  

---

# 👩‍💻 Author

SmartFarm PotatoGuard AI

Artificial Intelligence + Agriculture Innovation Project

---

# 📜 License

This project is developed for educational, research, and demonstration purposes.

---

# 🙏 Acknowledgement

Special thanks to the open-source AI and Machine Learning community for providing powerful frameworks and tools that make intelligent agriculture solutions possible.
