# notebooks/

Notebook training LSTM dijalankan di **Colab/Kaggle** (bukan di laptop).

Alur notebook:

1. Upload / mount folder **`data/processed/`** (data sudah TER-SPLIT di level
   video: `X_train/X_val/X_test` + `y_*` + `label_map.json` + `manifest.csv`).
   Notebook **tidak** melakukan split sendiri.
2. Muat 6 array split, tegaskan ringkasan jumlah sample per split.
3. Train, evaluasi (akurasi, confusion matrix). Lihat sel **10b** untuk panduan
   membaca hasil (homogenitas vs generalisasi; angka bersifat signer-dependent).
4. Export bobot `model.save_weights('bisindo_lstm.weights.h5')` + `label_map.json` →
   unduh ke `models/` lokal. (Loader lokal memuat via **weights**, bukan `.h5` penuh —
   lihat `src/inference/predictor.py`.)

File `train_lstm.ipynb` ada di sini.
