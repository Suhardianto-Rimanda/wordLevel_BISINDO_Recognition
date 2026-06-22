"""
predictor.py — Load model LSTM & prediksi kata dari sequence landmark.

Pemuatan via BOBOT (bukan .h5 penuh): bangun arsitektur lewat
``build_lstm_model`` (src/training/model.py — satu sumber kebenaran) lalu muat
``config.WEIGHTS_PATH``. Bobot = angka murni tanpa metadata versi, jadi portabel
lintas versi TF/Keras (bobot Colab TF 2.20 muat di TF 2.15 lokal). Ini menghindari
error deserialisasi ``quantization_config`` dari load_model(.h5) penuh.

Terima array (SEQUENCE_LENGTH, FEATURE_DIM) → kembalikan (kata, probabilitas) dari
kelas ber-probabilitas tertinggi. Threshold noise diterapkan di pemanggil
(InferenceEngine), bukan di sini.
"""

import json
from pathlib import Path

import h5py
import numpy as np

import config


def _load_weights_cross_version(model, path):
    """Muat bobot ke ``model`` lintas versi Keras (3 → 2).

    File ``.weights.h5`` dari Keras 3 (Colab TF 2.20) memakai layout HDF5 baru
    (root ``layers/`` dgn bobot di ``layers/<nama>/vars/<i>``, RNN di ``cell/vars``).
    Keras 2.15 ``load_weights()`` mengharap layout legacy (``model_weights`` +
    attr ``layer_names``) → gagal "0 variables" pada file Keras 3.

    Strategi: deteksi format. Bila file LEGACY → pakai ``load_weights`` native.
    Bila file KERAS 3 → baca bobot per layer via h5py lalu ``set_weights`` (numpy
    murni, bebas format kontainer). Pencocokan per ``layer.name`` (urutan iterasi
    h5 alfabetis, tak bisa diandalkan).
    """
    with h5py.File(str(path), "r") as f:
        is_keras3 = (
            "layers" in f
            and "model_weights" not in f
            and "layer_names" not in f.attrs
        )
        if not is_keras3:
            # Format legacy Keras 2 → jalur native tetap didukung.
            model.load_weights(str(path))
            return

        layers_grp = f["layers"]
        flat = []
        for layer in model.layers:                 # urutan topologis model
            n = len(layer.weights)
            if n == 0:                             # Dropout / Input: tanpa bobot
                continue
            if layer.name not in layers_grp:
                raise KeyError(
                    f"Layer '{layer.name}' tak ada di file bobot. Arsitektur lokal "
                    "tidak sinkron dengan saat training (cek build_lstm_model)."
                )
            g = layers_grp[layer.name]
            vars_grp = g["cell"]["vars"] if "cell" in g else g["vars"]  # RNN di cell/
            for i in range(n):
                flat.append(np.asarray(vars_grp[str(i)]))
        model.set_weights(flat)


def _load_labels(label_map_path):
    """Baca label_map.json {index: kata}; fallback ke config bila kosong/hilang."""
    try:
        with open(label_map_path, encoding="utf-8") as f:
            m = json.load(f)
        if m:  # bukan {}
            return [m[str(i)] for i in range(len(m))]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    # fallback: nama folder dataset, lalu placeholder LABEL_xx
    return config.get_labels_from_dataset()


class Predictor:
    """Pembungkus model LSTM untuk prediksi kata real-time."""

    def __init__(self, weights_path=None, label_map_path=None):
        # Catatan: InferenceEngine memanggil Predictor(model_path, label_map_path)
        # secara posisional dengan model_path=None default → jatuh ke WEIGHTS_PATH.
        weights_path = Path(weights_path or config.WEIGHTS_PATH)
        label_map_path = label_map_path or config.LABEL_MAP_PATH

        if not weights_path.exists():
            raise FileNotFoundError(
                f"File weights tidak ditemukan: {weights_path}\n"
                "Bobot harus diekspor dari Colab, BUKAN .h5 penuh. Di Colab:\n"
                "  model.save_weights('bisindo_lstm.weights.h5')\n"
                "lalu salin file itu ke folder models/ (sejajar label_map.json)."
            )

        # import lokal: TensorFlow berat, hanya dibutuhkan saat inference aktif.
        # build_lstm_model = satu sumber kebenaran arsitektur (src/training/model.py).
        from src.training.model import build_lstm_model

        self.model = build_lstm_model()
        try:
            _load_weights_cross_version(self.model, weights_path)
        except Exception as e:
            raise RuntimeError(
                "Gagal load_weights — arsitektur lokal kemungkinan TIDAK identik "
                "dengan saat training di Colab.\n"
                f"Cek dimensi: SEQUENCE_LENGTH={config.SEQUENCE_LENGTH}, "
                f"FEATURE_DIM={config.FEATURE_DIM}, NUM_CLASSES={config.NUM_CLASSES}.\n"
                "Pastikan urutan/jenis/units layer build_lstm_model() sama persis "
                "dengan notebook.\n"
                f"Error asli: {e}"
            ) from e

        # Konfirmasi arsitektur cocok dengan bobot (jumlah layer, shape, params).
        print(
            f"[Predictor] weights OK | layers={len(self.model.layers)} "
            f"in={self.model.input_shape} out={self.model.output_shape} "
            f"params={self.model.count_params():,}"
        )

        self.labels = _load_labels(label_map_path)

    def predict(self, sequence):
        """Array (SEQUENCE_LENGTH, FEATURE_DIM) → (kata, probabilitas).

        Mengembalikan kata dengan probabilitas softmax tertinggi (argmax) dan
        nilai probabilitasnya. Tanpa threshold (urusan pemanggil).
        """
        seq = np.asarray(sequence, dtype=np.float32)[None, ...]  # (1, 30, 126)
        probs = self.model.predict(seq, verbose=0)[0]
        idx = int(np.argmax(probs))
        prob = float(probs[idx])
        word = self.labels[idx] if idx < len(self.labels) else f"LABEL_{idx:02d}"
        return word, prob
