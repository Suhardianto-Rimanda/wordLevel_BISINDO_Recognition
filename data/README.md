# data/

Dataset & hasil ekstraksi. **Isi folder ini gitignored** (lihat `.gitignore`).

## `raw/`
Letakkan dataset **"BISINDO 40 Kata mp4"** di sini. Struktur: satu subfolder per
kata (40 folder), tiap folder berisi ~50 file `.mp4`.

```
data/raw/
├── Apa/
│   ├── BISINDO_Apa_001.mp4
│   └── ...
├── Apa Kabar/
│   └── BISINDO_Apa Kabar_001.mp4
│   └── ...
└── ...   (40 kelas)
```

Nama subfolder = label kata. `config.get_labels_from_dataset()` membaca nama-nama
ini secara terurut untuk membentuk daftar label.

## `processed/`  ← input training (satu-satunya)
Dataset **ter-split di LEVEL VIDEO** hasil `python scripts/extract_landmarks.py`
(membungkus `src/preprocessing/build_sequences.py`). Split 70/15/15 stratified per
kelas dengan seed tetap (semua dari `config.py`); satu video utuh hanya masuk satu
split sehingga **tidak ada kebocoran data antar split**.

Isi:
```
data/processed/
├── X_train.npy / y_train.npy
├── X_val.npy   / y_val.npy
├── X_test.npy  / y_test.npy
├── label_map.json      # {index: kata}
└── manifest.csv        # video_id, label, split, jumlah_sequence
```
`manifest.csv` = bukti audit anti-leakage: tiap `video_id` muncul di **satu** split.
Saat run, build mencetak `LEAKAGE CHECK (video overlap): PASS/FAIL`.

Upload/mount folder ini untuk training di Colab/Kaggle.

## `landmarks/`  ← DEPRECATED
Lokasi lama output gabungan (`X.npy`, `y.npy`, `(2000, 30, 126)`). **Tidak lagi
dihasilkan** dan tidak boleh dipakai training (split per-sample atasnya berisiko
kebocoran bila kelak satu video menghasilkan >1 sequence). Gunakan `processed/`.
