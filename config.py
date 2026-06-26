"""
config.py — Parameter terpusat aplikasi penerjemah BISINDO real-time.

Semua konstanta (path, ukuran frame, dimensi landmark, threshold, hyperparameter
model, daftar label) diletakkan di sini supaya modul lain tinggal import.
Tidak ada logika ML di file ini — hanya nilai dan satu helper pembaca label.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# PATHS
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"              # "BISINDO 40 Kata mp4": 40 folder kelas
LANDMARKS_DIR = DATA_DIR / "landmarks"  # (legacy) lokasi lama X.npy/y.npy gabungan
# DEPRECATED: output gabungan tidak lagi dihasilkan. Pipeline kini menyimpan
# data TER-SPLIT (level video) ke PROCESSED_DIR. Konstanta dibiarkan agar import
# lama tidak pecah, tapi build_sequences.py TIDAK menulis ke sini lagi.
X_PATH = LANDMARKS_DIR / "X.npy"        # (legacy) tensor fitur (n_samples, 30, 126)
Y_PATH = LANDMARKS_DIR / "y.npy"        # (legacy) label index (n_samples,)

# --- Dataset ter-split (level video) -> input training satu-satunya ---------- #
PROCESSED_DIR = DATA_DIR / "processed"
X_TRAIN_PATH = PROCESSED_DIR / "X_train.npy"
Y_TRAIN_PATH = PROCESSED_DIR / "y_train.npy"
X_VAL_PATH = PROCESSED_DIR / "X_val.npy"
Y_VAL_PATH = PROCESSED_DIR / "y_val.npy"
X_TEST_PATH = PROCESSED_DIR / "X_test.npy"
Y_TEST_PATH = PROCESSED_DIR / "y_test.npy"
PROCESSED_LABEL_MAP_PATH = PROCESSED_DIR / "label_map.json"  # {index: kata}
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"  # video_id,label,split,jumlah_sequence

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "bisindo_lstm.h5"     # (legacy) .h5 penuh — fallback opsional
# Sumber pemuatan UTAMA: bobot saja. Portabel lintas versi TF/Keras (angka murni,
# tanpa metadata versi) sehingga bobot hasil Colab (TF 2.20) bisa dimuat di TF 2.15.
WEIGHTS_PATH = MODELS_DIR / "bisindo_lstm.weights.h5"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"  # {index: kata}

# --------------------------------------------------------------------------- #
# FRAME / SEQUENCE
# --------------------------------------------------------------------------- #
SEQUENCE_LENGTH = 30   # jumlah frame berurutan per prediksi (buffer LSTM)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 20        # target fps capture webcam

# --------------------------------------------------------------------------- #
# DIMENSI LANDMARK (MediaPipe Holistic)
# --------------------------------------------------------------------------- #
# Tangan: 21 titik x (x, y, z) per tangan, 2 tangan.
# Pose  : 33 titik x (x, y, z, visibility)  -> TIDAK dipakai (hands-only).
HAND_LANDMARKS = 21
HAND_DIMS = 3                                   # x, y, z
POSE_LANDMARKS = 33
POSE_DIMS = 4                                   # x, y, z, visibility

HANDS_FEATURE_DIM = 2 * HAND_LANDMARKS * HAND_DIMS   # 2 * 21 * 3 = 126
POSE_FEATURE_DIM = POSE_LANDMARKS * POSE_DIMS        # 33 * 4    = 132 (tak dipakai)

# Feature set = TANGAN SAJA. Asal 126: 2 tangan x 21 titik x 3 (x,y,z).
# Pose sengaja dibuang untuk fokus sinyal utama BISINDO & model lebih ringan.
FEATURE_DIM = HANDS_FEATURE_DIM                      # 126 fitur per frame

# MediaPipe Holistic confidence
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Ekstraktor landmark realtime: "holistic" (default, identik training) atau "hands"
# (lebih ringan/cepat). JANGAN set "hands" sebelum scripts/compare_extractors.py
# memastikan fitur cocok dengan training — kalau beda, prediksi LSTM bisa rusak.
FEATURE_EXTRACTOR = "holistic"   # "holistic" | "hands"

# --------------------------------------------------------------------------- #
# INFERENCE / NLP
# --------------------------------------------------------------------------- #
PROBABILITY_THRESHOLD = 0.7   # prediksi di bawah ini diabaikan
SMOOTHING_WINDOW = 5          # jumlah prediksi terakhir untuk voting/smoothing
PREDICTION_COOLDOWN = 15      # min. frame jeda sebelum kata sama dicatat ulang

# --- State management kalimat (reset & anti-tumpuk) -------------------------- #
IDLE_RESET_SECONDS = 2.5   # idle tanpa kata baru -> finalisasi kalimat & kosongkan buffer
MAX_WORDS = 20             # batas kata per kalimat (cegah menumpuk tak terbatas)
ENFORCE_MAX_WORDS = True   # toggle penegakan batas panjang kalimat

# --------------------------------------------------------------------------- #
# YOLO ROI (opsional, Tahap B) — deteksi tangan + cropping SEBELUM MediaPipe
# --------------------------------------------------------------------------- #
# Toggle pengaman: False = baseline MediaPipe-only PERSIS seperti sekarang (YOLO
# tak disentuh, ultralytics/torch tak perlu ter-install). True = YOLO crop ROI →
# MediaPipe pada crop. YOLO HANYA deteksi+crop; klasifikasi tetap LSTM.
USE_YOLO_ROI = True
YOLO_MODEL_PATH = MODELS_DIR / "yolo_hand.pt"  # bobot best.pt dari Tahap A (Colab)
YOLO_DEVICE = "cuda"          # auto-fallback ke "cpu" bila GPU tak tersedia (TAK hardcode)
YOLO_CONF = 0.25              # confidence threshold deteksi tangan
YOLO_ROI_MARGIN = 0.20        # padding fraksi di sekeliling union bbox sebelum crop
YOLO_IMGSZ = 640              # ukuran input inferensi YOLO

# --------------------------------------------------------------------------- #
# MODEL / TRAINING (dipakai di Colab/Kaggle, bukan di laptop)
# --------------------------------------------------------------------------- #
NUM_CLASSES = 40
LSTM_UNITS = 128
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TRAIN_TEST_SPLIT = 0.2   # DEPRECATED: tak dipakai. Lihat SPLIT_*_RATIO di bawah.
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# SPLIT DATASET (level VIDEO, stratified per kelas)
# --------------------------------------------------------------------------- #
# Split dilakukan di tahap preprocessing (build_sequences.py), BUKAN di notebook.
# Satu video utuh hanya masuk ke SATU split. Ketiga rasio harus berjumlah 1.0.
SPLIT_TRAIN_RATIO = 0.70
SPLIT_VAL_RATIO = 0.15
SPLIT_TEST_RATIO = 0.15

# --------------------------------------------------------------------------- #
# LABELS
# --------------------------------------------------------------------------- #
# Placeholder 40 label. Nama kata asli di-derive dari nama folder dataset
# (lihat get_labels_from_dataset). Placeholder dipakai bila dataset belum ada.
LABELS = [f"LABEL_{i:02d}" for i in range(NUM_CLASSES)]


def get_labels_from_dataset(raw_dir: Path = RAW_DIR):
    """Kembalikan daftar label dari nama subfolder di ``raw_dir`` (terurut).

    Tiap subfolder pada dataset "BISINDO 40 Kata mp4" mewakili satu kata.
    Bila folder tidak ada atau kosong, fallback ke ``LABELS`` placeholder.
    """
    if not raw_dir.exists():
        return LABELS
    names = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())
    return names if names else LABELS


# --------------------------------------------------------------------------- #
# TTS / WEB
# --------------------------------------------------------------------------- #
TTS_LANG = "id"        # gTTS bahasa Indonesia
TTS_OUTPUT_DIR = BASE_DIR / "src" / "web" / "static" / "audio"

FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
DEBUG = True
