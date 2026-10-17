
import os
import uuid
import joblib
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.metrics import confusion_matrix
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

CLASSES = ["bus", "car", "motorcycle", "truck"]
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

cnn_path = os.path.join(MODEL_DIR, "vehiclevision_cnn_best.keras")
svm_path = os.path.join(MODEL_DIR, "vehiclevision_hog_svm.joblib")

cnn = tf.keras.models.load_model(cnn_path)
svm_data = joblib.load(svm_path)
svm = svm_data["model"]
svm_classes = svm_data["classes"]

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_hog(image):
    image = image.convert("RGB").resize((64, 64))
    array = np.asarray(image)
    gray = rgb2gray(array)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    ).reshape(1, -1)

def predict_vehicle(image_path, model_choice):
    with Image.open(image_path) as im:
        image = im.convert("RGB")

        if model_choice == "CNN":
            resized = image.resize((128, 128))
            array = np.asarray(resized, dtype=np.float32)
            probabilities = cnn.predict(np.expand_dims(array, axis=0), verbose=0)[0]
            index = int(np.argmax(probabilities))
            return {
                "label": CLASSES[index],
                "confidence": float(probabilities[index]) * 100,
                "scores": [
                    {"label": name, "score": float(probabilities[i]) * 100}
                    for i, name in enumerate(CLASSES)
                ]
            }

        features = extract_hog(image)
        prediction = int(svm.predict(features)[0])
        label = svm_classes[prediction]
        return {
            "label": label,
            "confidence": None,
            "scores": None
        }

def load_metrics(filename):
    import json
    path = os.path.join(RESULTS_DIR, filename)
    try:
        with open(path, "r") as file:
            return json.load(file)
    except (OSError, ValueError):
        return {}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    image_url = None
    selected_model = "CNN"

    if request.method == "POST":
        selected_model = request.form.get("model", "CNN")
        uploaded = request.files.get("image")

        if selected_model not in {"CNN", "HOG_SVM"}:
            error = "Please select a valid model."
        elif not uploaded or not uploaded.filename:
            error = "Please choose an image first."
        elif not allowed_file(uploaded.filename):
            error = "Please upload a JPG, JPEG, PNG, WEBP, or BMP image."
        else:
            filename = secure_filename(uploaded.filename)
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            save_path = os.path.join(UPLOAD_DIR, unique_name)

            try:
                uploaded.save(save_path)
                if os.path.getsize(save_path) == 0:
                    raise ValueError("The uploaded file is empty.")

                with Image.open(save_path) as im:
                    im.verify()

                result = predict_vehicle(save_path, selected_model)
                image_url = f"/static/uploads/{unique_name}"

            except (UnidentifiedImageError, OSError, ValueError):
                if os.path.exists(save_path):
                    os.remove(save_path)
                error = "The uploaded file could not be read as a valid image."
            except Exception:
                if os.path.exists(save_path):
                    os.remove(save_path)
                app.logger.exception("Prediction failed")
                error = "Prediction failed. Please try another image."

    cnn_metrics = load_metrics("cnn_test_metrics.json")
    svm_metrics = load_metrics("hog_svm_test_metrics.json")

    return render_template(
        "index.html",
        result=result,
        error=error,
        image_url=image_url,
        selected_model=selected_model,
        cnn_accuracy=cnn_metrics.get("test_accuracy", 0.79692) * 100,
        svm_accuracy=svm_metrics.get("test_accuracy", 0.6894) * 100
    )

@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        error="Image is too large. Maximum upload size is 10 MB.",
        result=None,
        image_url=None,
        selected_model="CNN",
        cnn_accuracy=79.692,
        svm_accuracy=68.94
    ), 413

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
