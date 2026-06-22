"""
model.py — Definisi arsitektur LSTM untuk klasifikasi kata BISINDO.

Input  : (SEQUENCE_LENGTH, FEATURE_DIM)
Output : softmax atas NUM_CLASSES.

SATU SUMBER KEBENARAN arsitektur. Dipakai dua sisi:
  * Training (Colab/Kaggle) — build → fit → save_weights.
  * Inference (laptop)       — build → load_weights (lihat src/inference/predictor.py).

PENTING: arsitektur di sini WAJIB identik dengan yang dilatih di notebook Colab.
Beda satu layer/units → load_weights gagal shape-mismatch.
"""

import config


def build_lstm_model(sequence_length=None, feature_dim=None, num_classes=None):
    """Bangun & kompilasi model LSTM. Kembalikan keras.Model.

    Arsitektur (identik dengan notebook training):
        Input(SEQUENCE_LENGTH, FEATURE_DIM)
        LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
        Dropout(0.4)
        LSTM(64)
        Dropout(0.4)
        Dense(64, relu)
        Dropout(0.3)
        Dense(NUM_CLASSES, softmax)
        compile: Adam(lr=1e-3), categorical_crossentropy.

    Default dimensi diambil dari config (SEQUENCE_LENGTH/FEATURE_DIM/NUM_CLASSES).
    """
    # import lokal: TensorFlow berat, hanya dibutuhkan saat build model aktif.
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
    from tensorflow.keras.optimizers import Adam

    sequence_length = sequence_length or config.SEQUENCE_LENGTH   # 30
    feature_dim = feature_dim or config.FEATURE_DIM               # 126
    num_classes = num_classes or config.NUM_CLASSES               # 40

    model = Sequential([
        Input(shape=(sequence_length, feature_dim)),
        LSTM(config.LSTM_UNITS, return_sequences=True,
             dropout=0.3, recurrent_dropout=0.2),          # 128
        Dropout(0.4),
        LSTM(config.LSTM_UNITS // 2),                       # 64
        Dropout(0.4),
        Dense(config.LSTM_UNITS // 2, activation="relu"),  # 64
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
