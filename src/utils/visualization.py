"""
visualization.py — Overlay landmark & info prediksi/FPS di atas frame.

Lapisan TAMPILAN (terpisah dari logika inference). Dipakai ulang oleh runner
laptop (run_webcam) maupun modul web (camera.py) untuk menggambar di frame
sebelum ditampilkan / di-encode MJPEG.
"""

import cv2
import mediapipe as mp

_mp_drawing = mp.solutions.drawing_utils
_mp_hands = mp.solutions.hands

# style koneksi tangan (hijau)
_HAND_STYLE = _mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
_CONN_STYLE = _mp_drawing.DrawingSpec(color=(0, 180, 0), thickness=2)


def draw_landmarks(frame, results):
    """Overlay landmark tangan kiri+kanan dari ``results`` Holistic ke frame.

    Mengubah frame in-place dan mengembalikannya. Tangan yang None dilewati.
    """
    if results is None:
        return frame
    for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
        if hand is not None:
            _mp_drawing.draw_landmarks(
                frame, hand, _mp_hands.HAND_CONNECTIONS,
                _HAND_STYLE, _CONN_STYLE,
            )
    return frame


def draw_prediction(frame, word, score):
    """Tulis kata + probabilitas di kiri-atas frame.

    word None (di bawah threshold / noise) → tampil '...' warna abu.
    """
    if word:
        text = f"{word}  ({score:.2f})"
        color = (0, 255, 0)
    else:
        text = "..."
        color = (160, 160, 160)

    # latar gelap agar teks terbaca
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return frame


def draw_fps(frame, fps):
    """Tulis FPS aktual di kanan-atas frame (ukur latency real-time)."""
    text = f"FPS: {fps:4.1f}"
    (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x = frame.shape[1] - w - 12
    cv2.putText(frame, text, (x, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return frame
