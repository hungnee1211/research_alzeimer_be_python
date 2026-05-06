from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import tensorflow as tf
import numpy as np
import cv2
import os
import uuid

from gradcam import preprocess_image, gradcam_plus_plus, build_grad_model

# ===== APP =====
app = FastAPI()

# Fix lỗi CORS: Đổi allow_credentials thành False khi allow_origins là "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== TẠO THƯ MỤC ẢNH =====
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# Mount static để truy cập ảnh qua URL
app.mount("/images", StaticFiles(directory="outputs"), name="images")

# ===== CONFIG =====
MODEL_PATH = "best_final.keras"
LAST_CONV_LAYER = "conv5_block3_out"

classes = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented",

]

# ===== LOAD MODEL =====
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Gỡ hàm Softmax ở lớp cuối (nếu có) để Grad-CAM++ lấy đạo hàm được chính xác nhất
if model.layers[-1].activation.__name__ == 'softmax':
    model.layers[-1].activation = tf.keras.activations.linear

grad_model = build_grad_model(model, LAST_CONV_LAYER)
print("Model loaded successfully!")

# ===== SAVE HEATMAP =====
def save_overlay(img_path, heatmap):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Khôi phục kỹ thuật mặt nạ: Lọc nền đen, chỉ lấy vùng có mô não
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_img, 15, 255, cv2.THRESH_BINARY)
    
    # Trộn ảnh và giữ nguyên vùng nền đen
    blended = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    overlay = np.where(mask[:, :, None] == 255, blended, img)

    filename = f"{uuid.uuid4()}.jpg"
    output_path = os.path.join("outputs", filename)

    cv2.imwrite(output_path, overlay)

    return filename

# ===== API =====
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    temp_path = f"temp_{file_id}.jpg"

    try:
        # 1. Lưu file nhận được
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # 2. Tiền xử lý ảnh
        img = preprocess_image(temp_path)

        # 3. Chạy AI và sinh Grad-CAM (chỉ chạy inference 1 lần duy nhất)
        heatmap, idx, raw_preds = gradcam_plus_plus(grad_model, img)

        # 4. Tính toán độ tin cậy (Confidence)
        probs = tf.nn.softmax(raw_preds).numpy()
        conf = float(probs[idx])

        # 5. Sinh và lưu ảnh Heatmap
        filename = save_overlay(temp_path, heatmap)

        return {
"success": True,
            "class": classes[idx],
            "confidence": round(conf, 4),
            "image_url": f"{BASE_URL}/images/{filename}"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

    finally:
        # Xóa file ảnh tạm sau khi xong việc
        if os.path.exists(temp_path):
            os.remove(temp_path)