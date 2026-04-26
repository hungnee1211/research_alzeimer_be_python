from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
import cv2
import os
import uuid

from gradcam import make_gradcam_heatmap, save_gradcam

# =========================
# FastAPI APP
# =========================
app = FastAPI(title="Alzheimer MRI Prediction API")

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Folder results
# =========================
os.makedirs("results", exist_ok=True)

app.mount(
    "/results",
    StaticFiles(directory="results"),
    name="results"
)

# =========================
# Load model
# =========================
model = tf.keras.models.load_model("adni_resnet50_best_100.keras")

# =========================
# Class labels
# =========================
classes = [
    "Non Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented"
]

# =========================
# Home route
# =========================
@app.get("/")
def home():
    return {
        "message": "Alzheimer MRI API Running"
    }

# =========================
# Predict Route
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # temp image name
    temp_name = f"temp_{uuid.uuid4()}.jpg"

    # save uploaded file
    with open(temp_name, "wb") as f:
        f.write(await file.read())

    try:
        # =====================
        # Read image
        # =====================
        img = cv2.imread(temp_name)

        if img is None:
            return {
                "error": "Cannot read image"
            }

        img = cv2.resize(img, (224, 224))
        img = img.astype("float32") / 255.0

        img_array = np.expand_dims(img, axis=0)

        # =====================
        # Predict
        # =====================
        preds = model.predict([img_array], verbose=0)

        # nếu model output list
        if isinstance(preds, list):
            preds = preds[0]

        preds = preds[0]

        idx = np.argmax(preds)

        predicted_class = classes[idx]
        confidence = float(preds[idx])

        # =====================
        # GradCAM
        # =====================
        heatmap = make_gradcam_heatmap(
            [img_array],
            model,
            last_conv_layer_name="conv5_block3_out"
        )

        # =====================
        # Save heatmap image
        # =====================
        result_name = f"{uuid.uuid4()}.jpg"
        result_path = f"results/{result_name}"

        save_gradcam(
            temp_name,
            heatmap,
            result_path
        )

        # =====================
        # Delete temp
        # =====================
        os.remove(temp_name)

        return {
            "success": True,
            "class": predicted_class,
            "confidence": confidence,
            "heatmap_url": f"http://localhost:8000/results/{result_name}"
        }

    except Exception as e:

        if os.path.exists(temp_name):
            os.remove(temp_name)

        return {
            "success": False,
            "error": str(e)
        }