"""
extract_frames.py — Baca file video .mp4 → daftar frame BGR.

Langkah pertama pipeline preprocessing: buka video dengan OpenCV dan kumpulkan
seluruh frame. Standardisasi panjang (ke 30 frame) dilakukan terpisah di
build_sequences.py, jadi di sini diambil SEMUA frame apa adanya.

Menangani video gagal (tak bisa dibuka / kosong) dengan return None + warning.
"""

from pathlib import Path
import cv2
from src.utils.logger import get_logger

logger = get_logger(__name__)

def read_frames(video_path):
    """Baca semua frame dari ``video_path``.
    Args:
        video_path: path file video (.mp4).
    Returns:
        list[np.ndarray] berisi frame BGR (H, W, 3), atau ``None`` bila video
        gagal dibuka atau tidak punya frame sama sekali.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.warning("Gagal membuka video: %s", video_path)
        cap.release()
        return None

    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        logger.warning("Video kosong / tak ada frame terbaca: %s", video_path)
        return None

    logger.debug("%s -> %d frame", Path(video_path).name, len(frames))
    return frames
