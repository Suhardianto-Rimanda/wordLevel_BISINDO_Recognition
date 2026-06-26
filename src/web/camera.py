"""
camera.py — Capture webcam (server-side OpenCV) + pipeline + MJPEG.

Membungkus pipeline real-time untuk web. REUSE penuh modul yang sudah ada:
  InferenceEngine (Holistic+LSTM+threshold) → PredictionSmoother → SentenceBuilder
  → annotate_frame (gambar landmark+label+FPS) → encode JPEG → stream MJPEG.

Tidak ada logika ML baru di sini — hanya orkestrasi + state untuk UI.
Model belum ada → graceful: webcam tetap stream + pesan, tanpa prediksi.
"""

import threading
import time

import cv2

import config
from src.inference.inference_engine import InferenceEngine, annotate_frame
from src.nlp.smoother import PredictionSmoother
from src.nlp.sentence_builder import SentenceBuilder
from src.utils import visualization as viz
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VideoCamera:
    """Sumber video webcam + pipeline inference real-time untuk web."""

    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.opened = self.cap.isOpened()

        self._lock = threading.Lock()
        self.smoother = PredictionSmoother()
        self.sentence_builder = SentenceBuilder()
        self.raw_word = None
        self.raw_score = 0.0

        # State finalisasi kalimat (reset berbasis idle timeout).
        self._last_activity = time.perf_counter()  # waktu kata terakhir di-commit
        self._final_sentence = ""                  # kalimat terakhir yang difinalisasi
        self._final_id = 0                         # naik tiap finalisasi -> sinyal TTS sekali

        # FPS (EMA)
        self._fps = 0.0
        self._prev = time.perf_counter()

        # Engine berat — hanya dibangun bila kamera benar terbuka.
        self.engine = None
        if self.opened:
            try:
                self.engine = InferenceEngine()
            except FileNotFoundError as e:
                logger.warning("Model belum ada, jalan tanpa prediksi: %s", e)
                self.engine = None

    @property
    def model_loaded(self):
        return self.engine is not None

    def _update_fps(self):
        now = time.perf_counter()
        dt = now - self._prev
        self._prev = now
        if dt > 0:
            inst = 1.0 / dt
            self._fps = inst if self._fps == 0.0 else 0.9 * self._fps + 0.1 * inst
        return self._fps

    def get_frame(self):
        """Baca 1 frame, jalankan pipeline, kembalikan JPEG bytes (atau None)."""
        ok, frame = self.cap.read()
        if not ok:
            return None

        fps = self._update_fps()

        if self.engine is not None:
            result = self.engine.process_frame(frame)
            committed = self.smoother.update(result.word, result.score)
            now = time.perf_counter()
            with self._lock:
                self.raw_word = result.word
                self.raw_score = result.score
                if committed:
                    self.sentence_builder.add_word(committed)
                    self._last_activity = now
                self._maybe_finalize(now)
            annotate_frame(frame, result, fps)
        else:
            # graceful: tampilkan webcam + pesan, tanpa prediksi
            viz.draw_fps(frame, fps)
            cv2.putText(frame, "Model belum ada - latih dulu", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

        ok, jpeg = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return jpeg.tobytes()

    def _maybe_finalize(self, now):
        """Finalisasi kalimat bila idle melewati ambang. ASUMSI lock dipegang.

        Bila ada kata & tak ada kata baru selama ``IDLE_RESET_SECONDS``: simpan
        kalimat sebagai final (naikkan ``_final_id`` -> sinyal gTTS sekali) lalu
        kosongkan buffer + smoother agar kalimat berikutnya mulai dari nol.
        Buffer kosong -> kondisi gagal -> tak refinalize berulang.
        """
        if self.sentence_builder.words and (
            now - self._last_activity >= config.IDLE_RESET_SECONDS
        ):
            self._final_sentence = self.sentence_builder.get_sentence()
            self._final_id += 1
            self.sentence_builder.clear()
            self.smoother.reset()

    def get_state(self):
        """State terkini untuk polling UI."""
        with self._lock:
            live = self.sentence_builder.get_sentence()
            return {
                "raw_word": self.raw_word,
                "raw_score": round(float(self.raw_score), 3),
                "fps": round(float(self._fps), 1),
                # tampilkan kalimat berjalan; bila sudah difinalisasi (buffer kosong),
                # tetap tampilkan kalimat final terakhir sampai kata baru muncul.
                "sentence": live if live else self._final_sentence,
                "final_sentence": self._final_sentence,
                "final_id": self._final_id,
                "model_loaded": self.model_loaded,
            }

    def reset(self):
        """Kosongkan buffer kata + state (tombol Reset Model).

        Reset manual TIDAK menaikkan ``_final_id`` -> tak memicu autoplay gTTS.
        """
        with self._lock:
            self.smoother.reset()
            self.sentence_builder.clear()
            if self.engine is not None:
                self.engine.reset()
            self.raw_word = None
            self.raw_score = 0.0
            self._final_sentence = ""
            self._last_activity = time.perf_counter()

    def release(self):
        """Lepas kamera & resource."""
        if self.cap is not None:
            self.cap.release()
        if self.engine is not None:
            self.engine.close()


def gen_frames(camera):
    """Generator multipart MJPEG untuk Flask Response."""
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


# --------------------------------------------------------------------------- #
# Singleton kamera (lazy): dibuat saat /video_feed pertama (tombol Mulai Kamera)
# --------------------------------------------------------------------------- #
_camera = None
_camera_lock = threading.Lock()


def get_camera(source=0):
    """Kembalikan singleton VideoCamera. Tak cache bila kamera gagal dibuka."""
    global _camera
    with _camera_lock:
        if _camera is not None and _camera.opened:
            return _camera
        cam = VideoCamera(source)
        if not cam.opened:
            cam.release()
            return cam  # caller cek .opened → balas 503
        _camera = cam
        return _camera


def current_camera():
    """Kembalikan singleton kamera yang ADA (tanpa membuat/membuka baru).

    Dipakai /state & /reset agar polling tak memicu pembukaan webcam.
    """
    return _camera


def release_camera():
    """Lepas singleton kamera (dipanggil saat shutdown)."""
    global _camera
    with _camera_lock:
        if _camera is not None:
            _camera.release()
            _camera = None
