from typing import List

import io
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO
import easyocr


app = FastAPI(
    title="License Plate Detection API",
    description="API untuk deteksi dan pembacaan plat nomor menggunakan Ultralytics YOLO dan EasyOCR",
    version="1.0.0",
)

# Atur CORS jika perlu diakses dari frontend lain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==== INISIALISASI MODEL ====
# Ganti path berikut dengan path model YOLO Anda sendiri
# Misalnya: "models/plate_yolov8n.pt" atau "best.pt"
YOLO_MODEL_PATH = "models/best.pt"

CROPS_DIR = Path("outputs") / "plate_crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)

try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    # Jika model belum ada, beri error yang jelas saat API dipanggil
    yolo_model = None
    yolo_load_error = str(e)
else:
    yolo_load_error = None

# EasyOCR untuk membaca teks plat
reader = easyocr.Reader(["en"])


def read_imagefile_as_cv2(data: bytes) -> np.ndarray:
    """Konversi bytes gambar menjadi array OpenCV (BGR)."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_bgr


@app.get("/")
def root():
    return {
        "message": "License Plate Detection API aktif",
        "yolo_model_loaded": yolo_load_error is None,
        "model_path": YOLO_MODEL_PATH,
    }


@app.post("/detect-plate")
async def detect_plate(file: UploadFile = File(...)):
    """
    Endpoint untuk deteksi dan pembacaan plat nomor.

    Cara pakai di Postman:
    - Method: POST
    - URL: http://localhost:8000/detect-plate
    - Body: form-data
      - key: file (type = File) -> pilih gambar kendaraan
    """
    if yolo_model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model YOLO belum bisa diload. Periksa file weight di '{YOLO_MODEL_PATH}'. Error: {yolo_load_error}",
        )

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File yang dikirim harus berupa gambar (image/*).")

    # Baca bytes file
    image_bytes = await file.read()
    img_bgr = read_imagefile_as_cv2(image_bytes)

    # Jalankan deteksi dengan YOLO
    try:
        results = yolo_model(img_bgr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menjalankan YOLO: {e}")

    if not results:
        return {"plates": [], "message": "Tidak ada hasil deteksi dari YOLO."}

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return {"plates": [], "message": "Tidak ada plat nomor yang terdeteksi."}

    h, w, _ = img_bgr.shape
    plates: List[dict] = []

    # Loop setiap bounding box
    for box in result.boxes:
        xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        conf = float(box.conf[0].item()) if box.conf is not None else None
        cls_id = int(box.cls[0].item()) if box.cls is not None else None

        x1, y1, x2, y2 = map(int, xyxy)

        # Pastikan koordinat masih dalam batas gambar
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        plate_crop = img_bgr[y1:y2, x1:x2]

        crop_saved_path = None
        if plate_crop.size > 0:
            crop_filename = f"{uuid4().hex}.jpg"
            crop_path = CROPS_DIR / crop_filename
            cv2.imwrite(str(crop_path), plate_crop)
            crop_saved_path = str(crop_path).replace("\\", "/")

        plate_text = ""
        ocr_confidence = None

        if plate_crop.size > 0:
            crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
            ocr_results = reader.readtext(crop_rgb, detail=1)

            texts = []
            confs = []
            for (_bbox, text, score) in ocr_results:
                texts.append(text)
                confs.append(float(score))

            if texts:
                plate_text = " ".join(texts)
                ocr_confidence = float(np.mean(confs))

        plates.append(
            {
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                "confidence": conf,
                "class_id": cls_id,
                "crop_path": crop_saved_path,
                "plate_text": plate_text,
                "ocr_confidence": ocr_confidence,
            }
        )

    return {
        "plates": plates,
        "count": len(plates),
    }


# Untuk menjalankan langsung dengan: python main.py
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

