## API Deteksi & Pembacaan Plat Nomor (YOLO + Ultralytics + EasyOCR)

### 1. Persiapan lingkungan

1. Pastikan Python 3.9+ sudah terinstall.
2. Buka terminal / PowerShell di:

   ```bash
   cd C:\Tracking_number
   ```

3. Install dependency:

   ```bash
   pip install -r requirements.txt
   ```

### 2. Siapkan model YOLO plat nomor

- Letakkan file weight YOLO Anda di folder:

  ```text
  C:\Tracking_number\models\plate_best.pt
  ```

- Jika nama/path berbeda, ubah variabel `YOLO_MODEL_PATH` di file `main.py`.

### 3. Menjalankan API

```bash
cd C:\Tracking_number
python main.py
```

API akan berjalan di: `http://localhost:8000`

Anda bisa cek dokumentasi interaktif di:

- `http://localhost:8000/docs`

### 4. Mengirim request dari Postman

- **Method**: `POST`
- **URL**: `http://localhost:8000/detect-plate`
- **Body**: pilih `form-data`
  - key: `file`
  - type: `File`
  - value: pilih gambar kendaraan (jpg/png)

**Response contoh (format JSON)**:

```json
{
    "plates": [
        {
            "bbox": {
                "x1": 1086,
                "y1": 1354,
                "x2": 1313,
                "y2": 1437
            },
            "confidence": 0.8520758152008057,
            "class_id": 1,
            "crop_path": "outputs/plate_crops/b1e678d0707443199a0459ec26b574fb.jpg",
            "plate_text": "",
            "ocr_confidence": null
        }
    ],
    "count": 1,
    "annotated_image_path": "outputs/annotated/acc62802eb3c4d6e8852373a9de63ffe.jpg"
}
```

### 5. Contoh hasil deteksi

Berikut contoh hasil deteksi plat nomor yang dihasilkan model:

![Contoh hasil deteksi](runs/detect/predict/macet_pondok_indah_ruf.jpg)

