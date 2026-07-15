from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model("potato_disease_model.keras")

class_names = ["Early Blight", "Late Blight", "Healthy"]

@app.route("/")
def home():
    return "Potato Disease API Running"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    image = Image.open(file).convert("RGB")
    image = image.resize((224, 224))

    img = np.array(image).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    idx = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    return jsonify({
        "disease": class_names[idx],
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
