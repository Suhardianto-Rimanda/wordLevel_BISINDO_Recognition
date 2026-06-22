"""
extract_landmarks.py — Ekstraksi landmark tangan via MediaPipe Holistic.

Per frame: deteksi tangan kiri & kanan, ambil 21 titik (x, y, z) tiap tangan.
Tangan yang tidak terdeteksi diisi zero-array agar dimensi selalu konsisten.

n_features per frame = 2 tangan x 21 titik x 3 (x,y,z) = 126.
Urutan vektor: [ tangan_kiri(63) | tangan_kanan(63) ].

Kelas HolisticLandmarkExtractor dipakai ULANG untuk dua hal:
  - offline  : build_sequences.py (proses dataset video)
  - realtime : src/web/camera.py (frame webcam)
"""

from types import SimpleNamespace

import numpy as np
import mediapipe as mp

import config

# 21 titik x 3 koordinat = 63 nilai per tangan
HAND_FEATURE_LEN = config.HAND_LANDMARKS * config.HAND_DIMS   # 63
N_FEATURES = config.FEATURE_DIM                               # 126

_mp_holistic = mp.solutions.holistic
_mp_hands = mp.solutions.hands


def _hand_to_array(hand_landmarks):
    """Konversi NormalizedLandmarkList satu tangan → np.ndarray (63,).

    Bila ``hand_landmarks`` None (tangan tak terdeteksi) → zero-array (63,)
    sehingga dimensi vektor frame tetap konsisten.
    """
    if hand_landmarks is None:
        return np.zeros(HAND_FEATURE_LEN, dtype=np.float32)
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )
    return coords.flatten()  # (21, 3) -> (63,)


class HolisticLandmarkExtractor:
    """Wrapper MediaPipe Holistic untuk ekstraksi landmark tangan.

    Pakai sebagai context manager:

        with HolisticLandmarkExtractor() as ex:
            vec = ex.extract(frame_bgr)   # np.ndarray (126,)
    """

    def __init__(
        self,
        static_image_mode=False,
        min_detection_confidence=None,
        min_tracking_confidence=None,
    ):
        self.holistic = _mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            min_detection_confidence=(
                min_detection_confidence
                if min_detection_confidence is not None
                else config.MIN_DETECTION_CONFIDENCE
            ),
            min_tracking_confidence=(
                min_tracking_confidence
                if min_tracking_confidence is not None
                else config.MIN_TRACKING_CONFIDENCE
            ),
        )

    def extract_with_results(self, frame_bgr):
        """Frame BGR → (vektor fitur (126,), results MediaPipe mentah).

        ``results`` dibutuhkan untuk menggambar landmark saat inference
        real-time. Tangan tak terdeteksi → blok 63 di-isi zero.
        """
        import cv2  # lokal: hindari dependency saat hanya butuh kelas

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.holistic.process(frame_rgb)

        left = _hand_to_array(results.left_hand_landmarks)
        right = _hand_to_array(results.right_hand_landmarks)
        return np.concatenate([left, right]), results  # (126,), results

    def extract(self, frame_bgr):
        """Frame BGR (H, W, 3) → vektor fitur np.ndarray (126,).

        Dipakai offline (build dataset) saat ``results`` tak diperlukan.
        """
        vec, _ = self.extract_with_results(frame_bgr)
        return vec

    def close(self):
        """Lepas resource MediaPipe."""
        self.holistic.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class HandsLandmarkExtractor:
    """Wrapper MediaPipe Hands (hands-only) — lebih ringan dari Holistic.

    Menghasilkan layout fitur IDENTIK dgn ``HolisticLandmarkExtractor``:
    ``[ tangan_kiri(63) | tangan_kanan(63) ]`` (126). Slot kiri/kanan diisi
    menurut label handedness MediaPipe ("Left"/"Right") — konvensi yang sama
    dipakai Holistic secara internal. Tangan tak terdeteksi → zero (63).

    OPSIONAL & di belakang flag ``config.FEATURE_EXTRACTOR``. Holistic menurunkan
    ROI tangan dari pose, jadi nilai landmark Hands TAK dijamin identik dgn data
    training Holistic — validasi dulu via ``scripts/compare_extractors.py``.

    Antarmuka SAMA dgn HolisticLandmarkExtractor: ``extract_with_results`` membalas
    results-adapter ber-atribut ``left_hand_landmarks``/``right_hand_landmarks``
    sehingga ``_hand_to_array`` & ``visualization.draw_landmarks`` jalan tanpa diubah.
    """

    def __init__(
        self,
        static_image_mode=False,
        min_detection_confidence=None,
        min_tracking_confidence=None,
    ):
        self.hands = _mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            min_detection_confidence=(
                min_detection_confidence
                if min_detection_confidence is not None
                else config.MIN_DETECTION_CONFIDENCE
            ),
            min_tracking_confidence=(
                min_tracking_confidence
                if min_tracking_confidence is not None
                else config.MIN_TRACKING_CONFIDENCE
            ),
        )

    def extract_with_results(self, frame_bgr):
        """Frame BGR → (vektor fitur (126,), results-adapter).

        Adapter meniru atribut Holistic (``left_hand_landmarks``/``right_hand_landmarks``)
        agar layer hilir (drawing) kompatibel.
        """
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)

        left = right = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for lm, handed in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handed.classification[0].label  # "Left" / "Right"
                if label == "Left" and left is None:
                    left = lm
                elif label == "Right" and right is None:
                    right = lm

        adapter = SimpleNamespace(
            left_hand_landmarks=left, right_hand_landmarks=right
        )
        vec = np.concatenate([_hand_to_array(left), _hand_to_array(right)])
        return vec, adapter  # (126,), adapter

    def extract(self, frame_bgr):
        """Frame BGR → vektor fitur np.ndarray (126,)."""
        vec, _ = self.extract_with_results(frame_bgr)
        return vec

    def close(self):
        """Lepas resource MediaPipe."""
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def make_extractor(static_image_mode=False, **kwargs):
    """Factory ekstraktor landmark realtime sesuai ``config.FEATURE_EXTRACTOR``.

    "holistic" (default) → HolisticLandmarkExtractor (identik training).
    "hands"             → HandsLandmarkExtractor (lebih ringan; validasi parity dulu).
    """
    if config.FEATURE_EXTRACTOR == "hands":
        return HandsLandmarkExtractor(static_image_mode=static_image_mode, **kwargs)
    return HolisticLandmarkExtractor(static_image_mode=static_image_mode, **kwargs)


def extract_sequence(frames, extractor):
    """Daftar frame BGR → np.ndarray (T, 126).

    Args:
        frames: list frame BGR.
        extractor: instance Holistic/Hands LandmarkExtractor (sudah dibuka).
    """
    seq = [extractor.extract(f) for f in frames]
    return np.array(seq, dtype=np.float32)
