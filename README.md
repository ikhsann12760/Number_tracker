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
      "bbox": { "x1": 100, "y1": 200, "x2": 300, "y2": 260 },
      "confidence": 0.92,
      "class_id": 0,
      "plate_text": "B 1234 CD",
      "ocr_confidence": 0.81
    }
  ],
  "count": 1
}
```

### 5. Contoh hasil deteksi

Berikut contoh hasil deteksi plat nomor yang dihasilkan model:

![Contoh hasil deteksi](runs/detect/predict/macet_pondok_indah_ruf.jpg)

