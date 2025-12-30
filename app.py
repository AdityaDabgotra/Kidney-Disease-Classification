import io
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# ------------------ APP SETUP ------------------
app = Flask(__name__)
CORS(app)

# ------------------ CONFIG ------------------
MODEL_PATH = "artifacts/training/model.h5"
IMAGE_SIZE = 224

CLASS_NAMES = ["Normal", "Stone"]

# ------------------ LOAD MODEL ONCE ------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# ------------------ IMAGE PREPROCESSING ------------------
def preprocess_image(image_bytes):
    """
    Converts raw image bytes to model-ready numpy array
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ------------------ ROUTES ------------------
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "Backend running"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Validate request
        if "image" not in request.files:
            return jsonify({"error": "Image file is required"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "Empty file"}), 400

        # Read image
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)

        # Predict
        predictions = model.predict(processed_image)
        confidence = float(np.max(predictions))
        class_index = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[class_index]

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------ RUN SERVER ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
