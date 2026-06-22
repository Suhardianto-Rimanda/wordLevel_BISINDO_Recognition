# Penerjemah Bahasa Isyarat Indonesia (BISINDO) Real-time

Aplikasi penerjemah BISINDO real-time berbasis webcam. **Word-level recognition** —
sistem memprediksi **satu kata per isyarat**, lalu merangkai kata-kata stabil menjadi
kalimat dengan aturan sederhana (rule-based). Ini **bukan** klasifikasi kalimat utuh dan
**bukan** koreksi tata bahasa. Tugas Akhir D-III Teknik Informatika.

> Status: **berjalan end-to-end** (webcam → teks → audio) untuk inference lokal & web.
> Training model dilakukan terpisah di Colab/Kaggle.

---

## Pipeline

Alur per frame (komponen **opsional** ditandai — default OFF):

```
1. Webcam capture frame (OpenCV, 640x480)
2. [OPSIONAL] YOLOv8 deteksi tangan → crop ROI          (USE_YOLO_ROI, default False)
3. MediaPipe → landmark TANGAN saja: 2 tangan x 21 titik x (x,y,z) = 126 fitur
   - default "holistic"; [OPSIONAL] "hands" (lebih ringan)  (FEATURE_EXTRACTOR)
   - pose & wajah TIDAK dipakai
4. FrameBuffer 30 frame → normalisasi (wrist-relative + scale, sama seperti training)
5. LSTM → label KATA + skor; di bawah PROBABILITY_THRESHOLD dianggap noise
6. NLP rule-based → smoothing (commit bila stabil) + dedup beruntun + rakit kalimat
   - finalisasi otomatis saat idle IDLE_RESET_SECONDS → kalimat dikosongkan utk kalimat baru
   - batas MAX_WORDS mencegah kalimat menumpuk tak terbatas
7. gTTS → audio (bahasa Indonesia), diputar SEKALI saat kalimat difinalisasi
8. Flask web UI → video ber-anotasi + panel teks + kontrol audio
```

**YOLO di sini hanya deteksi + cropping, bukan klasifikasi** — pengenalan kata tetap LSTM.

---

## Spesifikasi

| Item        | Nilai                                                |
|-------------|------------------------------------------------------|
| Bahasa      | Python 3.10+                                         |
| Web         | Flask (server-side OpenCV, stream MJPEG)             |
| Inference   | Lokal (laptop). TensorFlow **CPU-only** di Windows (lihat Catatan GPU) |
| Training    | Terpisah di Colab/Kaggle (LSTM & YOLO)              |
| Dataset     | "BISINDO 40 Kata mp4" — 40 kelas, ~50 video/kelas, **single-signer** |

---

## Struktur Folder

```
project_08/
├── config.py            # SEMUA parameter terpusat (frame, threshold, flag, path, label)
├── app.py               # entry point Flask
├── requirements.txt
├── data/
│   ├── raw/             # dataset mp4 (gitignored) — 1 subfolder per kata
│   └── processed/       # dataset TER-SPLIT level-video (X_/y_ train/val/test, label_map, manifest)
│                        #   (data/landmarks/ = lokasi lama, DEPRECATED)
├── src/
│   ├── preprocessing/   # extract_landmarks (MediaPipe), build_sequences (mp4→split npy)
│   ├── training/        # model.py (arsitektur LSTM, 1 sumber kebenaran), train.py (Colab)
│   ├── inference/       # predictor, frame_buffer, inference_engine, yolo_roi (opsional)
│   ├── nlp/             # smoother (PredictionSmoother), sentence_builder
│   ├── web/             # routes, camera (MJPEG pipeline), tts, templates, static
│   └── utils/           # logger, visualisasi landmark
├── models/              # bisindo_lstm.weights.h5 + label_map.json (+ yolo_hand.pt opsional)
├── notebooks/           # train_lstm.ipynb + train_yolo_hand.ipynb (Colab/Kaggle)
└── scripts/             # extract_landmarks, run_inference, compare_extractors (CLI)
```

---

## Setup

```bash
# 1. Virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt
```

**Catatan dependency penting** (lihat `requirements.txt`):
- **`numpy<2`** wajib (1.26.4) — TensorFlow 2.15 + MediaPipe belum kompatibel numpy 2.x.
- **`protobuf>=4.25.3,<5`** — protobuf 5.x membuat MediaPipe Holistic gagal init.
- `ultralytics` (YOLO) **opsional** — hanya perlu di-install bila `USE_YOLO_ROI=True`.

---

## Menjalankan

### Web app (utama)

```bash
python app.py
```

Buka **http://127.0.0.1:5000**, lalu:

1. **Mulai Kamera** → stream video real-time + landmark tangan.
2. Peragakan isyarat → panel **Deteksi Gerakan (LSTM)** menampilkan kata + skor;
   **Terjemahan Kalimat (NLP)** merangkai kata stabil menjadi kalimat.
3. **Kontrol Audio (gTTS)**: centang *Auto-play* → audio diputar **sekali** tiap kalimat
   difinalisasi (setelah jeda idle), atau klik **Putar Ulang** untuk kalimat yang tampil.
4. **Reset** → kosongkan buffer kata/kalimat (tidak memicu audio).

**Prasyarat:** dependency terpasang, webcam aktif, model di `models/`
(`bisindo_lstm.weights.h5` + `label_map.json`), dan **internet** untuk gTTS.
*Tanpa model, web tetap jalan:* webcam + landmark tampil, panel deteksi menampilkan
"Model belum ada — latih dulu".

**Endpoint:**

| Method | Route         | Fungsi                                               |
|--------|---------------|------------------------------------------------------|
| GET    | `/`           | Halaman utama                                        |
| GET    | `/video_feed` | Stream MJPEG webcam ber-anotasi                      |
| GET    | `/state`      | Kata mentah + skor + kalimat (+ `final_id`) — JSON   |
| POST   | `/speak`      | gTTS kalimat → URL mp3                               |
| POST   | `/reset`      | Kosongkan buffer kata/kalimat                        |

> Server dijalankan `threaded=True` → gTTS (`/speak`) berjalan di thread request
> terpisah, tidak memblokir streaming video.

### Inference tanpa web (debug)

```bash
python scripts/run_inference.py
```

Jendela OpenCV langsung: landmark + kata + **FPS aktual** (tekan `q` untuk keluar).

---

## Konfigurasi

Semua parameter terpusat di [`config.py`](config.py). Yang penting:

### Aktif (perilaku default)

| Flag | Default | Arti |
|------|---------|------|
| `SEQUENCE_LENGTH` | `30` | jumlah frame per prediksi (buffer LSTM) |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | `640` / `480` | resolusi capture webcam |
| `FEATURE_DIM` | `126` | fitur per frame (2 tangan × 21 × 3, hands-only) |
| `NUM_CLASSES` | `40` | jumlah kata |
| `PROBABILITY_THRESHOLD` | `0.7` | prediksi di bawah ini diabaikan (noise) |
| `SMOOTHING_WINDOW` | `5` | kata di-commit setelah muncul konsisten N kali |
| `IDLE_RESET_SECONDS` | `2.5` | idle tanpa kata baru → finalisasi kalimat & reset |
| `MAX_WORDS` / `ENFORCE_MAX_WORDS` | `20` / `True` | batas panjang kalimat (anti menumpuk) |

### Opsional (di belakang toggle — default OFF)

| Flag | Default | Arti |
|------|---------|------|
| `FEATURE_EXTRACTOR` | `"holistic"` | `"holistic"` (identik training) atau `"hands"` (lebih ringan). **Validasi parity dulu** via `scripts/compare_extractors.py` sebelum pakai `"hands"` |
| `USE_YOLO_ROI` | `False` | aktifkan YOLO crop ROI sebelum MediaPipe. Butuh `ultralytics` + `models/yolo_hand.pt` |
| `YOLO_DEVICE` | `"cuda"` | auto-fallback ke `"cpu"` bila GPU tak ada (tak pernah hardcode) |
| `YOLO_CONF` / `YOLO_ROI_MARGIN` / `YOLO_IMGSZ` | `0.25` / `0.20` / `640` | confidence, padding crop, ukuran input YOLO |

> Dengan `USE_YOLO_ROI=False` dan `FEATURE_EXTRACTOR="holistic"`, sistem berperilaku
> **persis seperti baseline MediaPipe-only** — jaring pengaman saat demo.

---

## Catatan Model

- **LSTM dimuat via WEIGHTS, bukan `.h5` penuh.** Arsitektur direkonstruksi oleh
  `build_lstm_model()` ([src/training/model.py](src/training/model.py) — satu sumber
  kebenaran), lalu bobot dimuat dari `models/bisindo_lstm.weights.h5`. Loader
  (`_load_weights_cross_version` di [predictor.py](src/inference/predictor.py)) menangani
  bobot Keras 3 (Colab TF 2.20) agar bisa dimuat di Keras 2.15 lokal. `bisindo_lstm.h5`
  hanya konstanta legacy — tidak dipakai jalur pemuatan.
- **Model dilatih terpisah** (Colab/Kaggle, lihat `notebooks/`). Export bobot via
  `model.save_weights('bisindo_lstm.weights.h5')`, lalu salin ke `models/` bersama
  `label_map.json`.
- **YOLO (opsional):** sediakan `models/yolo_hand.pt` (hasil training Tahap A,
  `notebooks/train_yolo_hand.ipynb`) bila ingin `USE_YOLO_ROI=True`.

---

## Catatan GPU (jujur)

- **TensorFlow lokal jalan CPU-only.** Pada Windows native, TF ≥ 2.11 tidak menyertakan
  CUDA — `tf.config.list_physical_devices('GPU')` kosong. Versi (TF 2.15) **sengaja
  dipertahankan** demi kompatibilitas MediaPipe; **jangan diubah** demi mengejar GPU.
- LSTM sangat ringan (~187K parameter); beban utama FPS = MediaPipe di CPU.
- **YOLO (opsional) bisa pakai GPU** lewat torch-CUDA yang **terpisah** dari TF (TF tetap
  CPU). Tanpa GPU, `yolo_roi.py` otomatis fallback ke CPU.
- **Trade-off:** mengaktifkan YOLO = dua model per frame → **FPS turun**. Ini disengaja
  untuk eksperimen pembanding (baseline vs YOLO) di laporan.

---

## Status & Keterbatasan

- **Signer-dependent:** dataset single-signer → akurasi bisa turun untuk penanda lain.
- **Bukan koreksi grammar:** NLP hanya menggabung + kapitalisasi + spasi, tanpa parsing
  S-P-O bahasa Indonesia.
- **YOLO ROI & ekstraktor Hands opsional** (default OFF); harus divalidasi sebelum dipakai
  sebagai default. mAP YOLO dievaluasi terpisah pada test set ber-anotasi (bukan webcam).
- **gTTS butuh internet** (sintesis audio online).

---

## Alur Kerja (training terpisah)

1. **Taruh dataset** di `data/raw/` — satu subfolder per kata (40 folder), isi `.mp4`.
2. **Bangun dataset landmark ter-split:** `python scripts/extract_landmarks.py`
   (membungkus `src/preprocessing/build_sequences.py`) → `data/processed/`
   (`X_/y_ train/val/test` + `label_map.json` + `manifest.csv`, split 70/15/15 level-video).
3. **Training** LSTM di Colab/Kaggle (`notebooks/train_lstm.ipynb`) → export
   `bisindo_lstm.weights.h5` + `label_map.json`.
4. **Taruh** kedua file ke `models/`.
5. **Jalankan** `python app.py`.

(YOLO opsional: `notebooks/train_yolo_hand.ipynb` → `best.pt` → `models/yolo_hand.pt`.)
